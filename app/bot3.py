from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

import app.db as dbm

log = logging.getLogger("bot")


# ======================
# Data structures
# ======================
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
    entry_reason: str = ""
    exit_reason: str = ""

    vol_regime: str = ""
    trend_regime: str = ""

    rsi_entry: float = 50.0
    sma_entry: float = 0.0
    atr_entry: float = 0.0
    rr_entry: float = 0.0

    peak_price: float = 0.0


# ======================
# Runtime profiles ("knobs")
# ======================
PRESETS: Dict[str, Dict[str, Any]] = {
    # safest: fewer trades, tighter risk
    "conservative": {
        "min_trade_frac": 0.02,
        "max_trade_frac": 0.06,
        "max_risk_per_trade_frac": 0.0015,
        "min_rr": 1.8,
        "cooldown_sec": 12,
        "enable_fallback": False,
        "trail_pct": 0.0016,
        "tp1_base_pct": 0.0014,
        "tp2_base_pct": 0.0024,
        "stop_base_pct": 0.0018,
    },
    # default balanced
    "balanced": {
        "min_trade_frac": 0.04,
        "max_trade_frac": 0.12,
        "max_risk_per_trade_frac": 0.0020,
        "min_rr": 1.6,
        "cooldown_sec": 8,
        "enable_fallback": False,
        "trail_pct": 0.0013,
        "tp1_base_pct": 0.0016,
        "tp2_base_pct": 0.0028,
        "stop_base_pct": 0.0022,
    },
    # higher frequency + size
    "aggressive": {
        "min_trade_frac": 0.06,
        "max_trade_frac": 0.18,
        "max_risk_per_trade_frac": 0.0030,
        "min_rr": 1.4,
        "cooldown_sec": 4,
        "enable_fallback": True,
        "trail_pct": 0.0011,
        "tp1_base_pct": 0.0018,
        "tp2_base_pct": 0.0032,
        "stop_base_pct": 0.0026,
    },
    # goal: high-frequency in downtrend to clip small profits
    "scalp_downtrend": {
        "min_trade_frac": 0.03,
        "max_trade_frac": 0.10,
        "max_risk_per_trade_frac": 0.0018,
        "min_rr": 1.4,
        "cooldown_sec": 3,
        "enable_dip": True,
        "enable_mom": True,
        "enable_breakout": False,
        "enable_fallback": False,
        "tp1_base_pct": 0.0011,
        "tp2_base_pct": 0.0019,
        "stop_base_pct": 0.0014,
        "trail_pct": 0.0010,
        "dip_entry_pct": 0.0012,
        "rsi_oversold": 40.0,
    },
    # goal: higher win-rate when momentum picks up (trend-follow / breakout)
    "momentum": {
        "min_trade_frac": 0.04,
        "max_trade_frac": 0.14,
        "max_risk_per_trade_frac": 0.0022,
        "min_rr": 1.7,
        "cooldown_sec": 6,
        "enable_dip": False,
        "enable_mom": True,
        "enable_breakout": True,
        "enable_fallback": False,
        "tp1_base_pct": 0.0020,
        "tp2_base_pct": 0.0045,
        "stop_base_pct": 0.0026,
        "trail_pct": 0.0014,
        "mom_up_bars": 3,
        "rsi_overbought": 75.0,
    },
    # "market maker" proxy (no limit orders here): smaller + tighter + only on strong mean-reversion
    "maker_like": {
        "min_trade_frac": 0.02,
        "max_trade_frac": 0.06,
        "max_risk_per_trade_frac": 0.0015,
        "min_rr": 1.5,
        "cooldown_sec": 6,
        "enable_dip": True,
        "enable_mom": False,
        "enable_breakout": False,
        "enable_fallback": False,
        "tp1_base_pct": 0.0009,
        "tp2_base_pct": 0.0014,
        "stop_base_pct": 0.0012,
        "trail_pct": 0.0009,
        "dip_entry_pct": 0.0014,
        "require_vol_increase": False,
    },
}


class KaspaBot:
    """
    Stable bot that matches your project wiring:

    - run.py constructs:
        exchange = Exchange.from_settings(s)
        con = dbm.connect()
        bot = KaspaBot(s, exchange, con)

    - exchange interface expected:
        get_klines_1m(symbol, limit) -> list[dict(open,high,low,close,volume,ts)]
        market_buy(symbol, spend_usdt)
        market_sell(symbol, qty)

    - DB interface expected (app/db.py):
        set_state(con, key, value)
        get_state(con, key)
        insert_trade(con, ts, symbol, side, qty, price, fee, fee_asset, mexc_trade_id, mexc_order_id, raw_json)
        insert_order(...) (optional)
        upsert_equity(con, ts, equity_usdt, realized_pnl_usdt, unrealized_pnl_usdt, price)
    """

    def __init__(self, settings, exchange, con):
        self.s = settings
        self.ex = exchange
        self.con = con

        self.position: Position = Position()
        self.usdt: float = float(getattr(self.s, "sim_start_quote_usdt", getattr(self.s, "sim_start_usdt", 10000.0)))
        self.base: float = 0.0  # KAS

        self.equity: float = self.usdt
        self.peak: float = self.equity
        self.dd: float = 0.0
        self.realized_pnl: float = 0.0

        self.last_trade_ms: int = 0
        self.pause_until_ms: int = 0
        self.loss_streak: int = 0

        self.vol_regime: str = "MID"
        self.trend_regime: str = "NEUTRAL"

        self._runtime_overrides: Dict[str, Any] = {}
        self._runtime_mode: str = str(getattr(self.s, "default_mode", "balanced"))

        # initialize state for dashboard
        dbm.set_state(self.con, "sim_quote", f"{self.usdt:.10f}")
        dbm.set_state(self.con, "sim_base", f"{self.base:.10f}")
        dbm.set_state(self.con, "mode", self._runtime_mode)
        dbm.set_state(self.con, "runtime_overrides_json", json.dumps(self._runtime_overrides))
        dbm.set_state(self.con, "position_json", json.dumps(asdict(self.position)))
        dbm.set_state(self.con, "halted", "0")
        dbm.set_state(self.con, "halt_reason", "")

        log.info("KaspaBot initialized")

    # ---------------------------
    # Config helpers (settings + runtime overrides + presets)
    # ---------------------------
    def _load_runtime_controls(self) -> None:
        """
        Pulls runtime mode/overrides from DB state so the web UI can change them live.
        Keys:
            - mode: preset name (string)
            - runtime_overrides_json: json object {param: value}
        """
        mode = dbm.get_state(self.con, "mode")
        if mode:
            self._runtime_mode = str(mode).strip()

        raw = dbm.get_state(self.con, "runtime_overrides_json")
        if raw:
            try:
                obj = json.loads(raw)
                if isinstance(obj, dict):
                    self._runtime_overrides = obj
            except Exception:
                pass

    def cfg(self, name: str, default: Any) -> Any:
        """
        Resolution order:
            1) runtime_overrides_json
            2) preset(mode)
            3) settings attribute
            4) default
        """
        if name in self._runtime_overrides:
            return self._runtime_overrides[name]
        preset = PRESETS.get(self._runtime_mode, {})
        if name in preset:
            return preset[name]
        return getattr(self.s, name, default)

    # ---------------------------
    # Indicators
    # ---------------------------
    @staticmethod
    def _sf(df: pd.DataFrame, col: str) -> pd.Series:
        return pd.to_numeric(df[col], errors="coerce").astype(float)

    @staticmethod
    def _rsi(closes: pd.Series, period: int = 14) -> float:
        closes = pd.to_numeric(closes, errors="coerce").astype(float)
        if len(closes) < period + 2:
            return 50.0
        delta = closes.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
        rs = avg_gain / (avg_loss.replace(0, np.nan))
        rsi = 100.0 - (100.0 / (1.0 + rs))
        v = float(rsi.iloc[-1])
        if not np.isfinite(v):
            return 50.0
        return float(np.clip(v, 0.0, 100.0))

    @staticmethod
    def _atr(highs: pd.Series, lows: pd.Series, closes: pd.Series, period: int = 14) -> float:
        highs = pd.to_numeric(highs, errors="coerce").astype(float)
        lows = pd.to_numeric(lows, errors="coerce").astype(float)
        closes = pd.to_numeric(closes, errors="coerce").astype(float)
        if len(closes) < period + 2:
            return 0.0
        prev = closes.shift(1)
        tr = pd.concat([(highs - lows).abs(), (highs - prev).abs(), (lows - prev).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
        return float(atr) if np.isfinite(atr) else 0.0

    def _compute_vol_regime(self, df1: pd.DataFrame) -> str:
        if df1 is None or len(df1) < 60:
            return "MID"

        closes = self._sf(df1, "close").tail(60)
        highs = self._sf(df1, "high").tail(60)
        lows = self._sf(df1, "low").tail(60)

        price = float(closes.iloc[-1])
        if price <= 0:
            return "MID"

        atr = self._atr(highs, lows, closes, int(self.cfg("atr_len", 14)))
        atr_pct = (atr / price) * 100.0 if atr > 0 else 0.0

        rets = closes.pct_change().dropna()
        hist_vol = float(rets.std() * 100.0) if len(rets) > 10 else 0.0

        combined = (atr_pct + hist_vol) / 2.0

        if combined < 0.35:
            return "EXTREME_LOW"
        if combined < 0.90:
            return "LOW"
        if combined < 1.80:
            return "MID"
        if combined < 3.20:
            return "HIGH"
        return "EXTREME_HIGH"

    def _compute_trend_regime(self, df1: pd.DataFrame) -> str:
        """
        Trend regime used to bias behavior:
          - DOWNTREND: price < SMA(trend) and SMA slope negative
          - UPTREND:   price > SMA(trend) and slope positive
          - NEUTRAL:   otherwise
        """
        if df1 is None or len(df1) < 80:
            return "NEUTRAL"
        closes = self._sf(df1, "close")
        price = float(closes.iloc[-1])
        trend_len = int(self.cfg("trend_sma_len", 40))
        sma = float(closes.tail(trend_len).mean())
        if sma <= 0:
            return "NEUTRAL"

        # slope: compare SMA now vs 15 bars ago
        if len(closes) < trend_len + 20:
            return "NEUTRAL"
        sma_now = float(closes.tail(trend_len).mean())
        sma_prev = float(closes.iloc[-15:-15 + trend_len].mean()) if len(closes) >= (trend_len + 15) else sma_now
        slope = (sma_now - sma_prev) / sma_prev if sma_prev > 0 else 0.0

        if price < sma and slope < -0.00015:
            return "DOWNTREND"
        if price > sma and slope > 0.00015:
            return "UPTREND"
        return "NEUTRAL"

    # ---------------------------
    # Risk/Reward validation
    # ---------------------------
    def _validate_rr(self, entry: float, tp: float, stop: float) -> Tuple[bool, float]:
        if entry <= 0 or tp <= 0 or stop <= 0:
            return False, 0.0
        risk = entry - stop
        reward = tp - entry
        if risk <= 0 or reward <= 0:
            return False, 0.0
        rr = reward / risk
        return rr >= float(self.cfg("min_rr", 1.6)), float(rr)

    # ---------------------------
    # Entry signals
    # ---------------------------
    def _entry_signal(self, df1: pd.DataFrame, price: float) -> Optional[str]:
        if df1 is None or len(df1) < 60:
            return None

        closes = self._sf(df1, "close")
        highs = self._sf(df1, "high")
        volumes = self._sf(df1, "volume")

        sma_len = int(self.cfg("sma_len", 10))
        sma = float(closes.tail(sma_len).mean())
        rsi = self._rsi(closes, int(self.cfg("rsi_len", 14)))

        # block extremes
        if self.vol_regime in ("EXTREME_LOW", "EXTREME_HIGH"):
            return None

        avg_vol = float(volumes.tail(20).mean()) if len(volumes) >= 20 else float(volumes.mean())
        require_vol_increase = bool(self.cfg("require_vol_increase", True))
        vol_increase = (not require_vol_increase) or (avg_vol > 0 and volumes.iloc[-1] > avg_vol * 1.2)

        dip_entry_pct = float(self.cfg("dip_entry_pct", 0.0009))
        mom_up = int(self.cfg("mom_up_bars", 2))

        # --- 1) Oversold bounce (mean reversion) ---
        if bool(self.cfg("enable_dip", True)):
            below_sma = sma > 0 and price <= sma * (1.0 - dip_entry_pct)
            bouncing = closes.iloc[-1] > closes.iloc[-2]
            rsi_oversold = float(self.cfg("rsi_oversold", 35.0))
            if below_sma and bouncing and (rsi < rsi_oversold) and vol_increase:
                return "OVERSOLD_BOUNCE"

        # --- 2) Momentum continuation ---
        if bool(self.cfg("enable_mom", True)):
            consecutive_ups = 0
            for i in range(1, mom_up + 1):
                if closes.iloc[-i] > closes.iloc[-i - 1]:
                    consecutive_ups += 1
            above_sma = sma > 0 and price > sma
            rsi_overbought = float(self.cfg("rsi_overbought", 70.0))
            rsi_ok = 35 < rsi < rsi_overbought
            if consecutive_ups == mom_up and above_sma and rsi_ok:
                return "MOMENTUM_CONTINUATION"

        # --- 3) Breakout ---
        if bool(self.cfg("enable_breakout", True)):
            recent_high = float(highs.tail(20).max())
            breakout = recent_high > 0 and price > recent_high * 1.001
            strong_vol = avg_vol > 0 and volumes.iloc[-1] > avg_vol * 1.5
            if breakout and strong_vol and rsi < float(self.cfg("rsi_overbought", 70.0)):
                return "BREAKOUT"

        return None

    # ---------------------------
    # Adaptive levels (ATR + base blend)
    # ---------------------------
    def _adaptive_levels(self, price: float, atr: float) -> Tuple[float, float, float]:
        if price <= 0:
            return 0.0, 0.0, 0.0

        base_tp1_pct = float(self.cfg("tp1_base_pct", 0.0016))
        base_tp2_pct = float(self.cfg("tp2_base_pct", 0.0028))
        base_stop_pct = float(self.cfg("stop_base_pct", 0.0022))

        base_tp1 = price * (1.0 + base_tp1_pct)
        base_tp2 = price * (1.0 + base_tp2_pct)
        base_stop = price * (1.0 - base_stop_pct)

        tp1_mult = float(self.cfg("tp1_atr", 2.0))
        tp2_mult = float(self.cfg("tp2_atr", 3.2))
        stop_mult = float(self.cfg("stop_atr", 1.6))

        if self.vol_regime in ("LOW", "EXTREME_LOW"):
            tp1_mult *= 0.7
            tp2_mult *= 0.7
            stop_mult *= 0.85
        elif self.vol_regime in ("HIGH", "EXTREME_HIGH"):
            tp1_mult *= 1.3
            tp2_mult *= 1.4
            stop_mult *= 1.25

        atr = max(float(atr), price * 0.0002)
        atr_tp1 = price + atr * tp1_mult
        atr_tp2 = price + atr * tp2_mult
        atr_stop = price - atr * stop_mult

        blend = float(self.cfg("atr_blend", 0.5))  # 0..1
        blend = float(np.clip(blend, 0.0, 1.0))
        tp1 = blend * atr_tp1 + (1.0 - blend) * base_tp1
        tp2 = blend * atr_tp2 + (1.0 - blend) * base_tp2
        stop = blend * atr_stop + (1.0 - blend) * base_stop

        return float(tp1), float(tp2), float(stop)

    # ---------------------------
    # Exits (TP/SL + emergency)
    # ---------------------------
    def _emergency_exit(self, df1: pd.DataFrame) -> Tuple[bool, str]:
        if df1 is None or len(df1) < 25:
            return False, ""

        closes = self._sf(df1, "close")
        opens = self._sf(df1, "open")
        volumes = self._sf(df1, "volume")

        avg_vol = float(volumes.tail(20).mean())
        if avg_vol > 0 and volumes.iloc[-1] > avg_vol * float(self.cfg("bear_vol_spike_mult", 4.0)):
            if closes.iloc[-1] < opens.iloc[-1]:
                return True, "BEARISH_VOL_SPIKE"

        if all(closes.iloc[-i] < closes.iloc[-i - 1] for i in range(1, 4)):
            return True, "MOMENTUM_REVERSAL"

        # after tp1, protect profits if RSI extreme
        if self.position.tp1_done:
            rsi = self._rsi(closes, int(self.cfg("rsi_len", 14)))
            if rsi > float(self.cfg("rsi_extreme", 82.0)):
                return True, "RSI_EXTREME"

        return False, ""

    def _update_peak_trail(self, price: float) -> None:
        trail_pct = float(self.cfg("trail_pct", 0.0013))
        if trail_pct <= 0:
            return

        if self.position.peak_price <= 0:
            self.position.peak_price = price
        if price > self.position.peak_price:
            self.position.peak_price = price

        new_trail = self.position.peak_price * (1.0 - trail_pct)
        self.position.trail = max(self.position.trail, new_trail) if self.position.trail > 0 else new_trail

    def _check_exits(self, now_ms: int, price: float, df1: pd.DataFrame) -> Tuple[bool, str, float]:
        if self.position.qty <= 0:
            return False, "", 0.0

        self._update_peak_trail(price)

        emerg, reason = self._emergency_exit(df1)
        if emerg:
            return True, reason, self.position.qty

        max_hold = int(self.cfg("max_hold_sec", 45))
        if (now_ms - self.position.entry_ts) >= max_hold * 1000:
            return True, "TIME_EXIT", self.position.qty

        stop_level = max(self.position.stop, self.position.trail)
        if stop_level > 0 and price <= stop_level:
            return True, "STOP_HIT", self.position.qty

        if (not self.position.tp1_done) and self.position.tp1 > 0 and price >= self.position.tp1:
            frac = float(self.cfg("tp1_sell_fraction", 0.50))
            frac = float(np.clip(frac, 0.05, 0.95))
            return True, "TP1", self.position.qty * frac

        if self.position.tp2 > 0 and price >= self.position.tp2:
            return True, "TP2", self.position.qty

        return False, "", 0.0

    # ---------------------------
    # Position sizing (risk cap + clamps)
    # ---------------------------
    def _risk_cap_fraction(self, price: float, stop: float) -> float:
        max_risk_frac = float(self.cfg("max_risk_per_trade_frac", 0.002))
        max_risk_usdt = max(0.0, self.equity * max_risk_frac)
        risk_per_unit = max(price - stop, 1e-12)  # risk per 1 KAS
        max_qty = max_risk_usdt / risk_per_unit
        max_spend = max_qty * price
        return float(max_spend / self.equity) if self.equity > 0 else 0.0

    def _entry_fraction(self, price: float, stop: float) -> float:
        min_f = float(self.cfg("min_trade_frac", 0.04))
        max_f = float(self.cfg("max_trade_frac", 0.12))
        # base fraction (midpoint)
        frac = float(self.cfg("default_trade_frac", (min_f + max_f) / 2.0))

        # trend bias:
        # downtrend -> slightly smaller + more frequent,
        # uptrend -> allow slightly bigger
        if self.trend_regime == "DOWNTREND":
            frac *= float(self.cfg("downtrend_size_mult", 0.80))
        elif self.trend_regime == "UPTREND":
            frac *= float(self.cfg("uptrend_size_mult", 1.10))

        # high vol -> reduce
        if self.vol_regime == "HIGH":
            frac *= float(self.cfg("high_vol_size_mult", 0.60))

        # cap by risk
        frac = min(frac, self._risk_cap_fraction(price, stop))

        return float(np.clip(frac, min_f, max_f))

    # ---------------------------
    # Cooldowns / pauses
    # ---------------------------
    def _cooldown_ok(self, now_ms: int) -> bool:
        cd = int(self.cfg("cooldown_sec", getattr(self.s, "cooldown_seconds", 8)))
        return self.last_trade_ms == 0 or (now_ms - self.last_trade_ms) >= cd * 1000

    def _in_pause(self, now_ms: int) -> bool:
        return now_ms < self.pause_until_ms

    def _pause_after_loss(self, now_ms: int) -> None:
        pause_s = int(self.cfg("pause_after_losses_sec", 20))
        self.pause_until_ms = now_ms + pause_s * 1000

    # ---------------------------
    # Trade execution (SIM or LIVE)
    # ---------------------------
    def _live(self) -> bool:
        return bool(getattr(self.s, "live_trading", False))

    def _buy(self, now_ms: int, price: float, spend_usdt: float, reason: str, meta: Dict[str, Any]) -> None:
        if spend_usdt <= float(self.cfg("min_notional_usdt", 5.0)):
            return
        qty = spend_usdt / price

        if self._live():
            self.ex.market_buy(self.s.symbol, spend_usdt)

        self.usdt -= spend_usdt
        self.base += qty

        self.position = Position(
            qty=float(qty),
            entry_price=float(price),
            entry_ts=int(now_ms),
            tp1=float(meta["tp1"]),
            tp2=float(meta["tp2"]),
            stop=float(meta["stop"]),
            trail=0.0,
            tp1_done=False,
            entry_reason=str(reason),
            vol_regime=str(self.vol_regime),
            trend_regime=str(self.trend_regime),
            rsi_entry=float(meta["rsi"]),
            sma_entry=float(meta["sma"]),
            atr_entry=float(meta["atr"]),
            rr_entry=float(meta["rr"]),
            peak_price=float(price),
        )

        self.last_trade_ms = now_ms

        # DB trade log (schema is minimal; raw_json holds extras)
        dbm.insert_trade(
            self.con,
            ts=now_ms,
            symbol=self.s.symbol,
            side="BUY",
            qty=float(qty),
            price=float(price),
            fee=0.0,
            fee_asset=None,
            mexc_trade_id=None,
            mexc_order_id=None,
            raw_json=json.dumps({"reason": reason, "meta": meta}),
        )

        log.info(f"BUY {qty:.2f} @ {price:.6f} reason={reason} mode={self._runtime_mode} trend={self.trend_regime} vol={self.vol_regime}")

    def _sell(self, now_ms: int, price: float, qty: float, reason: str) -> None:
        qty = min(float(qty), float(self.position.qty))
        if qty <= 0:
            return

        if self._live():
            self.ex.market_sell(self.s.symbol, qty)

        proceeds = qty * price
        cost = qty * self.position.entry_price
        realized = proceeds - cost

        self.base -= qty
        self.usdt += proceeds
        self.realized_pnl += realized

        remaining = self.position.qty - qty

        dbm.insert_trade(
            self.con,
            ts=now_ms,
            symbol=self.s.symbol,
            side="SELL",
            qty=float(qty),
            price=float(price),
            fee=0.0,
            fee_asset=None,
            mexc_trade_id=None,
            mexc_order_id=None,
            raw_json=json.dumps({"reason": reason, "realized": realized, "entry_reason": self.position.entry_reason}),
        )

        if remaining <= 1e-12:
            if realized < 0:
                self.loss_streak += 1
                self._pause_after_loss(now_ms)
            else:
                self.loss_streak = 0
            log.info(f"SELL {qty:.2f} @ {price:.6f} reason={reason} realized={realized:.4f}")
            self.position = Position()
            self.last_trade_ms = now_ms
        else:
            self.position.qty = remaining
            self.position.tp1_done = True
            log.info(f"PARTIAL SELL {qty:.2f} @ {price:.6f} reason={reason} realized={realized:.4f}")

    # ---------------------------
    # Step loop
    # ---------------------------
    def step(self) -> Dict[str, Any]:
        now_ms = int(time.time() * 1000)

        # refresh runtime knobs
        self._load_runtime_controls()

        # "auto mode": if enabled, bot flips between scalp_downtrend and momentum based on trend_regime
        if bool(self.cfg("auto_mode", True)):
            # compute using the last bars; if switching modes, do it gently (only when flat)
            pass

        klines = self.ex.get_klines_1m(self.s.symbol, limit=180)
        df1 = pd.DataFrame(klines)
        if df1.empty:
            return {}

        price = float(df1["close"].iloc[-1])

        self.vol_regime = self._compute_vol_regime(df1)
        self.trend_regime = self._compute_trend_regime(df1)

        # Auto switch between the two main behaviors
        if bool(self.cfg("auto_mode", True)) and self.position.qty <= 0:
            if self.trend_regime == "DOWNTREND":
                self._runtime_mode = str(self.cfg("downtrend_mode_name", "scalp_downtrend"))
            elif self.trend_regime == "UPTREND":
                self._runtime_mode = str(self.cfg("uptrend_mode_name", "momentum"))
            else:
                # keep whatever the UI selected
                pass

        # mark-to-market
        eq = self.usdt + self.base * price
        self.equity = float(eq)
        self.peak = max(self.peak, self.equity)
        self.dd = (self.peak - self.equity) / self.peak if self.peak > 0 else 0.0
        unreal = (price - self.position.entry_price) * self.position.qty if self.position.qty > 0 else 0.0

        # upsert equity row for dashboard
        dbm.upsert_equity(
            self.con,
            ts=now_ms,
            equity_usdt=float(self.equity),
            realized_pnl_usdt=float(self.realized_pnl),
            unrealized_pnl_usdt=float(unreal),
            price=float(price),
        )

        # update state for dashboard
        dbm.set_state(self.con, "sim_quote", f"{self.usdt:.10f}")
        dbm.set_state(self.con, "sim_base", f"{self.base:.10f}")
        dbm.set_state(self.con, "position_json", json.dumps(asdict(self.position)))
        dbm.set_state(self.con, "vol_regime", self.vol_regime)
        dbm.set_state(self.con, "trend_regime", self.trend_regime)
        dbm.set_state(self.con, "mode", self._runtime_mode)

        # exits
        if self.position.qty > 0:
            should_exit, reason, qty = self._check_exits(now_ms, price, df1)
            if should_exit:
                self._sell(now_ms, price, qty, reason)

        # entries
        if self.position.qty <= 0:
            if self._cooldown_ok(now_ms) and (not self._in_pause(now_ms)):
                sig = self._entry_signal(df1, price)
                if sig:
                    closes = self._sf(df1, "close")
                    highs = self._sf(df1, "high")
                    lows = self._sf(df1, "low")

                    sma_len = int(self.cfg("sma_len", 10))
                    sma = float(closes.tail(sma_len).mean())
                    rsi = self._rsi(closes, int(self.cfg("rsi_len", 14)))
                    atr = self._atr(highs, lows, closes, int(self.cfg("atr_len", 14)))

                    tp1, tp2, stop = self._adaptive_levels(price, atr)
                    ok_rr, rr = self._validate_rr(price, tp1, stop)
                    if not ok_rr:
                        sig = None

                    # additional safety: don't trade extremes
                    if sig and self.vol_regime in ("EXTREME_LOW", "EXTREME_HIGH"):
                        sig = None

                    if sig:
                        frac = self._entry_fraction(price, stop)
                        spend = min(self.usdt, self.equity * frac)
                        meta = {"tp1": tp1, "tp2": tp2, "stop": stop, "rsi": rsi, "sma": sma, "atr": atr, "rr": rr}
                        self._buy(now_ms, price, spend, sig, meta)

        return {
            "ts": now_ms,
            "symbol": self.s.symbol,
            "price": price,
            "equity": self.equity,
            "peak": self.peak,
            "dd": self.dd,
            "usdt": self.usdt,
            "kas": self.base,
            "vol_regime": self.vol_regime,
            "trend_regime": self.trend_regime,
            "mode": self._runtime_mode,
            "pos_qty": self.position.qty,
            "pause_s": max(0, int((self.pause_until_ms - now_ms) / 1000)),
            "loss_streak": self.loss_streak,
        }

    def run_forever(self) -> None:
        log.info("BOT STARTED")
        sleep_s = float(getattr(self.s, "poll_seconds", getattr(self.s, "loop_sleep_sec", 1.0)))
        while True:
            try:
                self.step()
            except Exception as e:
                log.exception(f"Loop error: {e}")
            time.sleep(sleep_s)
