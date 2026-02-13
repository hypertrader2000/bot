from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger("bot")


# ============================================================
# Position
# ============================================================
@dataclass
class Position:
    qty: float = 0.0
    entry_price: float = 0.0
    entry_ts: int = 0

    tp1: float = 0.0
    tp2: float = 0.0
    stop: float = 0.0
    trail: float = 0.0  # PRICE LEVEL

    tp1_done: bool = False
    peak_price: float = 0.0

    entry_reason: str = ""
    exit_reason: str = ""
    regime: str = ""  # "DOWN_SCALP" | "UP_MOMO" | etc


# ============================================================
# Bot
# ============================================================
class KaspaBot:
    """
    Stable bot logic for your current project.

    EXPECTED dependencies passed from run.py:
      - exchange: has get_klines_1m(symbol,limit), market_buy(symbol, spend_usdt), market_sell(symbol, qty)
      - db: your app/db.py module (connect(), set_state(), get_state(), insert_trade(), upsert_equity())
      - con: sqlite3 connection from db.connect()

    OPTIONAL: DB knobs
      If your UI writes state key "knobs_json" with JSON like {"risk_mode":"aggressive", "min_rr":1.1, ...}
      this bot will apply them live.
    """

    def __init__(self, s, exchange, con, db):
        self.s = s
        self.ex = exchange
        self.con = con
        self.db = db

        self.position = Position()

        # SIM balances (dashboard expects these in state)
        start_usdt = float(getattr(self.s, "sim_start_usdt", getattr(self.s, "sim_start_quote_usdt", 10000.0)))
        self.sim_quote = start_usdt
        self.sim_base = 0.0

        self.pause_until_ms = 0
        self.last_trade_ms = 0
        self.loss_streak = 0

        # cache for debug
        self._last_knob_log_ms = 0

        log.info("KaspaBot initialized")

    # ----------------------------
    # Settings + Knobs
    # ----------------------------
    def _get_knobs(self) -> Dict[str, Any]:
        """
        Optional live overrides from DB state key "knobs_json".
        If not present, returns empty dict.
        """
        try:
            raw = self.db.get_state(self.con, "knobs_json")
            if not raw:
                return {}
            v = json.loads(raw)
            return v if isinstance(v, dict) else {}
        except Exception:
            return {}

    def _g(self, name: str, default: Any) -> Any:
        """
        Get config value from (1) knobs_json override, else (2) Settings object, else default.
        """
        knobs = getattr(self, "_knobs_cache", None)
        if knobs is None:
            knobs = self._get_knobs()
            self._knobs_cache = knobs
        if name in knobs:
            return knobs[name]
        return getattr(self.s, name, default)

    def _log_knobs_periodically(self, now_ms: int) -> None:
        if now_ms - self._last_knob_log_ms < 10_000:
            return
        self._last_knob_log_ms = now_ms
        log.info(
            "KNOBS "
            f"enable_dip_entry={self._g('enable_dip_entry', True)} "
            f"enable_micro_mom_entry={self._g('enable_micro_mom_entry', True)} "
            f"enable_fallback_entry={self._g('enable_fallback_entry', False)} "
            f"trend_filter={self._g('trend_filter', True)} trend_sma_len={self._g('trend_sma_len', 40)} "
            f"sma_len={self._g('sma_len', 8)} dip_entry_pct={self._g('dip_entry_pct', 0.0008)} "
            f"mom_up_bars={self._g('mom_up_bars', 2)} "
            f"min_rr={self._g('min_rr', 1.1)} cooldown_seconds={self._g('cooldown_seconds', 5)}"
        )

    # ----------------------------
    # Indicators
    # ----------------------------
    @staticmethod
    def _sf(df: pd.DataFrame, col: str) -> pd.Series:
        return pd.to_numeric(df[col], errors="coerce").astype(float)

    @staticmethod
    def _rsi(closes: pd.Series, period: int = 14) -> float:
        closes = pd.to_numeric(closes, errors="coerce").astype(float)
        if len(closes) < period + 2:
            return 50.0
        d = closes.diff()
        gain = d.clip(lower=0.0)
        loss = (-d).clip(lower=0.0)
        avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
        rs = avg_gain / (avg_loss.replace(0, np.nan))
        rsi = 100.0 - (100.0 / (1.0 + rs))
        v = float(rsi.iloc[-1])
        return 50.0 if not np.isfinite(v) else float(np.clip(v, 0.0, 100.0))

    @staticmethod
    def _atr(highs: pd.Series, lows: pd.Series, closes: pd.Series, period: int = 14) -> float:
        highs = pd.to_numeric(highs, errors="coerce").astype(float)
        lows = pd.to_numeric(lows, errors="coerce").astype(float)
        closes = pd.to_numeric(closes, errors="coerce").astype(float)
        if len(closes) < period + 2:
            return 0.0
        prev = closes.shift(1)
        tr = pd.concat([(highs - lows).abs(), (highs - prev).abs(), (lows - prev).abs()], axis=1).max(axis=1)
        v = tr.ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
        return float(v) if np.isfinite(v) else 0.0

    # ----------------------------
    # Vol regime (simple + practical)
    # ----------------------------
    def _vol_regime(self, df1: pd.DataFrame) -> str:
        if df1 is None or len(df1) < 60:
            return "MID"
        closes = self._sf(df1, "close").tail(60)
        highs = self._sf(df1, "high").tail(60)
        lows = self._sf(df1, "low").tail(60)
        price = float(closes.iloc[-1])
        if price <= 0:
            return "MID"
        atr = self._atr(highs, lows, closes, int(self._g("atr_len", 14)))
        atr_pct = (atr / price) * 100.0 if atr > 0 else 0.0
        rets = closes.pct_change().dropna()
        hist_vol = float(rets.std() * 100.0) if len(rets) > 10 else 0.0
        combined = (atr_pct + hist_vol) / 2.0

        # thresholds tuned for 1m scalping
        if combined < 0.35:
            return "EXTREME_LOW"
        if combined < 0.90:
            return "LOW"
        if combined < 1.80:
            return "MID"
        if combined < 3.20:
            return "HIGH"
        return "EXTREME_HIGH"

    # ----------------------------
    # Trend regime -> controls how we trade
    # ----------------------------
    def _trend_regime(self, closes: pd.Series, price: float) -> str:
        """
        Goal:
          - downtrend: scalp mean-reversion with smaller targets (HIGH frequency)
          - uptrend: momentum entries (higher win rate)
        """
        if not bool(self._g("trend_filter", True)):
            return "NEUTRAL"

        n = int(self._g("trend_sma_len", 40))
        if len(closes) < n + 2:
            return "NEUTRAL"
        sma = float(closes.tail(n).mean())
        if sma <= 0:
            return "NEUTRAL"

        # 0.10% buffer avoids flip-flopping
        if price < sma * 0.999:
            return "DOWN"
        if price > sma * 1.001:
            return "UP"
        return "NEUTRAL"

    # ----------------------------
    # Risk / RR
    # ----------------------------
    def _validate_rr(self, entry: float, tp: float, stop: float) -> Tuple[bool, float]:
        if entry <= 0 or tp <= 0 or stop <= 0:
            return False, 0.0
        risk = entry - stop
        reward = tp - entry
        if risk <= 0 or reward <= 0:
            return False, 0.0
        rr = reward / risk
        min_rr = float(self._g("min_rr", 1.1))  # lower default so it actually trades
        return rr >= min_rr, float(rr)

    # ----------------------------
    # Adaptive targets (different for downtrend scalps vs uptrend momo)
    # ----------------------------
    def _levels(self, price: float, atr: float, trend: str) -> Tuple[float, float, float, float]:
        """
        Returns (tp1, tp2, stop, trail_pct).
        """
        if price <= 0:
            return 0.0, 0.0, 0.0, 0.0

        # Base scalping profile from your Settings
        stop_pct = float(self._g("scalp_stop", 0.0018))
        tp1_pct = float(self._g("scalp_tp1", 0.0035))
        tp2_pct = float(self._g("scalp_tp2", 0.0065))
        trail_pct = float(self._g("scalp_trail", 0.0022))

        # Make DOWN mode higher frequency:
        # smaller profits, faster exits, tighter trail
        if trend == "DOWN":
            tp1_pct *= 0.55
            tp2_pct *= 0.55
            stop_pct *= 0.85
            trail_pct *= 0.65

        # Make UP mode slightly larger winners, better R:R
        if trend == "UP":
            tp1_pct *= 1.05
            tp2_pct *= 1.10
            stop_pct *= 0.95

        # ATR blend for stability
        atr = max(float(atr), price * 0.00015)

        # ATR multipliers (smaller in DOWN to keep fast)
        tp_mult = 2.0 if trend != "DOWN" else 1.3
        stop_mult = 1.6 if trend != "DOWN" else 1.2

        base_tp1 = price * (1.0 + tp1_pct)
        base_tp2 = price * (1.0 + tp2_pct)
        base_stop = price * (1.0 - stop_pct)

        atr_tp1 = price + atr * tp_mult
        atr_tp2 = price + atr * (tp_mult * 1.8)
        atr_stop = price - atr * stop_mult

        tp1 = 0.6 * base_tp1 + 0.4 * atr_tp1
        tp2 = 0.6 * base_tp2 + 0.4 * atr_tp2
        stop = 0.6 * base_stop + 0.4 * atr_stop

        return float(tp1), float(tp2), float(stop), float(trail_pct)

    # ----------------------------
    # Entry logic (trades more)
    # ----------------------------
    def _entry_signal(self, df1: pd.DataFrame, price: float, trend: str, vol: str) -> Optional[str]:
        if df1 is None or len(df1) < 50:
            return None

        closes = self._sf(df1, "close")
        highs = self._sf(df1, "high")
        opens = self._sf(df1, "open")
        volumes = self._sf(df1, "volume")

        sma_len = int(self._g("sma_len", 8))
        sma = float(closes.tail(sma_len).mean()) if len(closes) >= sma_len else float(closes.mean())
        rsi = self._rsi(closes, int(self._g("rsi_len", 14)))

        avg_vol = float(volumes.tail(20).mean()) if len(volumes) >= 20 else float(volumes.mean())
        vol_ok = avg_vol > 0 and volumes.iloc[-1] >= avg_vol * (1.05 if trend == "DOWN" else 1.10)

        # Avoid trading in EXTREME_HIGH unless you explicitly allow
        allow_extreme_high = bool(self._g("allow_extreme_high", False))
        if vol == "EXTREME_HIGH" and not allow_extreme_high:
            return None

        # -----------------------------------
        # DOWN trend: mean reversion scalps
        # -----------------------------------
        if trend == "DOWN":
            if bool(self._g("enable_dip_entry", True)):
                dip_pct = float(self._g("dip_entry_pct", 0.0008))
                below_sma = sma > 0 and price <= sma * (1.0 - dip_pct)

                # "bounce" + candle confirmation
                bounce = closes.iloc[-1] > closes.iloc[-2]
                green = closes.iloc[-1] > opens.iloc[-1]

                # loosen RSI so it trades
                oversold = rsi < 48

                if below_sma and (bounce or green) and oversold and vol_ok:
                    return "DOWN_DIP_BOUNCE"

            # fallback tiny micro-mom even in downtrend (small, quick)
            if bool(self._g("enable_micro_mom_entry", True)):
                # allow 1 strong green candle
                strong_green = (closes.iloc[-1] > opens.iloc[-1]) and ((closes.iloc[-1] - opens.iloc[-1]) / max(opens.iloc[-1], 1e-9) > 0.0008)
                if strong_green and rsi < 60 and vol_ok:
                    return "DOWN_MICRO_POP"

            return None

        # -----------------------------------
        # UP / NEUTRAL: momentum & breakouts
        # -----------------------------------
        if bool(self._g("enable_micro_mom_entry", True)):
            mom_up = int(self._g("mom_up_bars", 2))
            ups = 0
            for i in range(1, mom_up + 1):
                if closes.iloc[-i] > closes.iloc[-i - 1]:
                    ups += 1
            above_sma = sma > 0 and price > sma
            if ups == mom_up and above_sma and (35 < rsi < 72) and vol_ok:
                return "MOMO_CONT"

        # breakout
        recent_high = float(highs.tail(20).max())
        breakout = recent_high > 0 and price > recent_high * 1.001
        strong_vol = avg_vol > 0 and volumes.iloc[-1] > avg_vol * 1.25
        if breakout and strong_vol and rsi < 74:
            return "BREAKOUT"

        # fallback (optional)
        if bool(self._g("enable_fallback_entry", False)):
            if sma > 0 and price > sma * 1.0015 and (45 < rsi < 68) and vol_ok and vol != "EXTREME_LOW":
                return "TREND_FOLLOW"

        return None

    # ----------------------------
    # Exits (fast in DOWN, let winners run in UP)
    # ----------------------------
    def _update_trail(self, price: float, trail_pct: float) -> None:
        if self.position.qty <= 0 or trail_pct <= 0:
            return
        if self.position.peak_price <= 0:
            self.position.peak_price = price
        if price > self.position.peak_price:
            self.position.peak_price = price
        level = self.position.peak_price * (1.0 - trail_pct)
        self.position.trail = max(self.position.trail, level) if self.position.trail > 0 else level

    def _should_emergency_exit(self, df1: pd.DataFrame) -> Tuple[bool, str]:
        if df1 is None or len(df1) < 25:
            return False, ""

        closes = self._sf(df1, "close")
        opens = self._sf(df1, "open")
        volumes = self._sf(df1, "volume")

        avg_vol = float(volumes.tail(20).mean())
        if avg_vol > 0 and volumes.iloc[-1] > avg_vol * 4.0 and closes.iloc[-1] < opens.iloc[-1]:
            return True, "BEAR_VOL_SPIKE"

        # 3 consecutive lower closes
        if all(closes.iloc[-i] < closes.iloc[-i - 1] for i in range(1, 4)):
            return True, "3_RED"

        return False, ""

    def _check_exit(self, now_ms: int, price: float, df1: pd.DataFrame, trend: str, trail_pct: float) -> Tuple[bool, str, float]:
        if self.position.qty <= 0:
            return False, "", 0.0

        self._update_trail(price, trail_pct)

        emerg, ereason = self._should_emergency_exit(df1)
        if emerg:
            return True, ereason, self.position.qty

        max_hold = int(self._g("max_hold_sec", 120))
        if trend == "DOWN":
            # faster time exit in downtrend
            max_hold = int(max(20, max_hold * 0.55))

        if now_ms - self.position.entry_ts >= max_hold * 1000:
            return True, "TIME", self.position.qty

        stop_level = max(self.position.stop, self.position.trail)
        if stop_level > 0 and price <= stop_level:
            return True, "STOP", self.position.qty

        if (not self.position.tp1_done) and self.position.tp1 > 0 and price >= self.position.tp1:
            frac = float(self._g("tp1_sell_fraction", 0.60))
            return True, "TP1", self.position.qty * frac

        if self.position.tp2 > 0 and price >= self.position.tp2:
            return True, "TP2", self.position.qty

        return False, "", 0.0

    # ----------------------------
    # Execution helpers
    # ----------------------------
    def _cooldown_ok(self, now_ms: int) -> bool:
        cd = int(self._g("cooldown_seconds", 5))
        return self.last_trade_ms == 0 or (now_ms - self.last_trade_ms) >= cd * 1000

    def _in_pause(self, now_ms: int) -> bool:
        return now_ms < self.pause_until_ms

    def _pause_after_loss(self, now_ms: int) -> None:
        pause_s = int(self._g("pause_on_loss_streak_sec", 1200))
        self.pause_until_ms = now_ms + pause_s * 1000

    def _trade_frac(self, trend: str, vol: str) -> float:
        min_f = float(self._g("min_trade_frac", 0.04))
        max_f = float(self._g("max_trade_frac", 0.12))

        # Default mid
        frac = (min_f + max_f) / 2.0

        # downtrend: smaller but more frequent
        if trend == "DOWN":
            frac *= 0.75

        # extreme high vol: reduce
        if vol in ("HIGH", "EXTREME_HIGH"):
            frac *= 0.65

        return float(np.clip(frac, min_f, max_f))

    # ----------------------------
    # DB state helpers for dashboard
    # ----------------------------
    def _write_state(self, price: float) -> None:
        # position_json expected by your webapp
        pos = {
            "qty": self.position.qty,
            "cost_usdt": self.position.qty * self.position.entry_price,
            "stop": self.position.stop,
            "trail": self.position.trail,
            "tp1": self.position.tp1,
            "tp2": self.position.tp2,
        }
        try:
            self.db.set_state(self.con, "position_json", json.dumps(pos))
            self.db.set_state(self.con, "sim_quote", f"{self.sim_quote:.10f}")
            self.db.set_state(self.con, "sim_base", f"{self.sim_base:.10f}")
        except Exception:
            pass

        # equity table
        equity = self.sim_quote + self.sim_base * price
        unreal = 0.0
        if self.position.qty > 0:
            unreal = (price - self.position.entry_price) * self.position.qty
        realized = 0.0  # your DB schema tracks realized separately in equity; keep 0 unless you want to compute daily

        try:
            self.db.upsert_equity(
                self.con,
                ts=int(time.time() * 1000),
                equity_usdt=float(equity),
                realized_pnl_usdt=float(realized),
                unrealized_pnl_usdt=float(unreal),
                price=float(price),
            )
        except Exception:
            pass

    # ----------------------------
    # Buy/Sell (SIM or LIVE)
    # ----------------------------
    def _buy(self, now_ms: int, price: float, spend_usdt: float, reason: str, trend: str, tp1: float, tp2: float, stop: float) -> None:
        if spend_usdt <= float(self._g("min_notional_usdt", 5.0)):
            return
        qty = spend_usdt / max(price, 1e-12)

        live = bool(self._g("live_trading", False))
        if live:
            self.ex.market_buy(str(self._g("symbol", "KASUSDT")), float(spend_usdt))

        # sim balances updated either way (dashboard)
        self.sim_quote -= spend_usdt
        self.sim_base += qty

        self.position = Position(
            qty=float(qty),
            entry_price=float(price),
            entry_ts=int(now_ms),
            tp1=float(tp1),
            tp2=float(tp2),
            stop=float(stop),
            trail=0.0,
            tp1_done=False,
            peak_price=float(price),
            entry_reason=reason,
            regime=("DOWN_SCALP" if trend == "DOWN" else "UP_MOMO"),
        )
        self.last_trade_ms = now_ms

        log.info(f"BUY {qty:.2f} @ {price:.6f} spend={spend_usdt:.2f} reason={reason} trend={trend}")

        # minimal trade insert compatible with your db.py schema
        try:
            self.db.insert_trade(
                self.con,
                ts=now_ms,
                symbol=str(self._g("symbol", "KASUSDT")),
                side="BUY",
                qty=float(qty),
                price=float(price),
                fee=0.0,
                fee_asset=None,
                mexc_trade_id=None,
                mexc_order_id=None,
                raw_json=json.dumps({"reason": reason, "trend": trend}),
            )
        except Exception:
            pass

    def _sell(self, now_ms: int, price: float, qty: float, reason: str) -> None:
        qty = float(min(qty, self.position.qty))
        if qty <= 0:
            return

        live = bool(self._g("live_trading", False))
        if live:
            self.ex.market_sell(str(self._g("symbol", "KASUSDT")), float(qty))

        proceeds = qty * price
        self.sim_base -= qty
        self.sim_quote += proceeds

        self.position.qty -= qty
        if self.position.qty <= 1e-12:
            self.position.exit_reason = reason
            self.position = Position()
        else:
            self.position.tp1_done = True

        log.info(f"SELL {qty:.2f} @ {price:.6f} reason={reason}")

        try:
            self.db.insert_trade(
                self.con,
                ts=now_ms,
                symbol=str(self._g("symbol", "KASUSDT")),
                side="SELL",
                qty=float(qty),
                price=float(price),
                fee=0.0,
                fee_asset=None,
                mexc_trade_id=None,
                mexc_order_id=None,
                raw_json=json.dumps({"reason": reason}),
            )
        except Exception:
            pass

    # ----------------------------
    # Main step loop
    # ----------------------------
    def step(self) -> Dict[str, Any]:
        now_ms = int(time.time() * 1000)

        # refresh knob cache every step
        self._knobs_cache = self._get_knobs()
        self._log_knobs_periodically(now_ms)

        symbol = str(self._g("symbol", "KASUSDT"))

        klines = self.ex.get_klines_1m(symbol, limit=160)
        df1 = pd.DataFrame(klines)
        if df1.empty:
            return {"ts": now_ms, "symbol": symbol, "error": "no_klines"}

        price = float(df1["close"].iloc[-1])
        closes = self._sf(df1, "close")
        highs = self._sf(df1, "high")
        lows = self._sf(df1, "low")

        vol = self._vol_regime(df1)
        trend = self._trend_regime(closes, price)

        atr = self._atr(highs, lows, closes, int(self._g("atr_len", 14)))
        tp1, tp2, stop, trail_pct = self._levels(price, atr, trend)

        # write state for dashboard
        self._write_state(price)

        # -------------- EXITS --------------
        if self.position.qty > 0:
            do_exit, ex_reason, qty = self._check_exit(now_ms, price, df1, trend, trail_pct)
            if do_exit:
                self._sell(now_ms, price, qty, ex_reason)

        # -------------- ENTRIES --------------
        if self.position.qty <= 0:
            sig = self._entry_signal(df1, price, trend, vol)

            if not sig:
                log.info(f"NO_ENTRY signal=None trend={trend} vol={vol}")
            else:
                # blockers with explicit logs
                if self._in_pause(now_ms):
                    log.info("ENTRY_BLOCKED paused")
                    sig = None
                elif not self._cooldown_ok(now_ms):
                    log.info("ENTRY_BLOCKED cooldown")
                    sig = None
                elif vol == "EXTREME_LOW":
                    log.info("ENTRY_BLOCKED extreme_low_vol")
                    sig = None

            if sig:
                ok_rr, rr = self._validate_rr(price, tp1, stop)
                if not ok_rr:
                    log.info(f"ENTRY_BLOCKED rr_low rr={rr:.2f} min_rr={self._g('min_rr', 1.1)}")
                    sig = None

            if sig:
                frac = self._trade_frac(trend, vol)
                spend = self.sim_quote * frac
                self._buy(now_ms, price, spend, sig, trend, tp1, tp2, stop)

        equity = self.sim_quote + self.sim_base * price

        return {
            "ts": now_ms,
            "symbol": symbol,
            "price": price,
            "equity": equity,
            "sim_quote": self.sim_quote,
            "sim_base": self.sim_base,
            "trend": trend,
            "vol": vol,
            "pos_qty": self.position.qty,
            "pos_entry": self.position.entry_price,
            "pos_tp1": self.position.tp1,
            "pos_tp2": self.position.tp2,
            "pos_stop": self.position.stop,
            "pos_trail": self.position.trail,
        }

    def run_forever(self) -> None:
        log.info("BOT STARTED")
        poll = float(getattr(self.s, "poll_seconds", 2))
        while True:
            try:
                self.step()
            except Exception as e:
                log.exception(f"loop error: {e}")
            time.sleep(poll)
