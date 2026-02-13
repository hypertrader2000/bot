from __future__ import annotations

import json
import time
import logging
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd

log = logging.getLogger("bot")


# ----------------------------
# Position model
# ----------------------------
@dataclass
class Position:
    qty: float = 0.0
    entry_price: float = 0.0
    entry_ts: int = 0

    tp1: float = 0.0
    tp2: float = 0.0
    stop: float = 0.0
    trail: float = 0.0

    tp1_done: bool = False
    entry_reason: str = ""
    vol_regime: str = ""

    peak_price: float = 0.0


# ----------------------------
# Bot
# ----------------------------
class KaspaBot:
    """
    Stable bot that:
    - Works with: KaspaBot(s, exchange, con, db)
    - Trades EXTREME_LOW using a CHOP_SCALP entry (tight targets)
    - Prints WHY it didn't enter (block reasons)
    - Stores state to DB (position_json, knobs, halted flags)
    """

    def __init__(self, settings, exchange, con, db):
        self.s = settings
        self.ex = exchange
        self.con = con
        self.db = db

        # balances (SIM by default)
        self.usdt = float(getattr(self.s, "sim_start_quote_usdt", getattr(self.s, "sim_start_usdt", 10000.0)))
        self.kas = 0.0

        self.position = Position()

        self.last_trade_ms = 0
        self.pause_until_ms = 0

        self._last_knobs_log_ms = 0

        log.info("KaspaBot initialized")

    # ----------------------------
    # Indicator helpers
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
        g = d.clip(lower=0.0)
        l = (-d).clip(lower=0.0)
        ag = g.ewm(alpha=1 / period, adjust=False).mean()
        al = l.ewm(alpha=1 / period, adjust=False).mean()
        rs = ag / (al.replace(0, np.nan))
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

    # ----------------------------
    # Vol regime + trend
    # ----------------------------
    def _compute_vol_regime(self, df1: pd.DataFrame) -> str:
        if df1 is None or len(df1) < 60:
            return "MID"

        closes = self._sf(df1, "close").tail(60)
        highs = self._sf(df1, "high").tail(60)
        lows = self._sf(df1, "low").tail(60)

        p = float(closes.iloc[-1])
        if p <= 0:
            return "MID"

        atr = self._atr(highs, lows, closes, 14)
        atr_pct = (atr / p) * 100.0 if atr > 0 else 0.0

        rets = closes.pct_change().dropna()
        hist_vol = float(rets.std() * 100.0) if len(rets) > 10 else 0.0

        combined = (atr_pct + hist_vol) / 2.0

        # Slightly looser EXTREME_LOW so we don't get stuck there forever
        if combined < 0.20:
            return "EXTREME_LOW"
        if combined < 0.90:
            return "LOW"
        if combined < 1.80:
            return "MID"
        if combined < 3.20:
            return "HIGH"
        return "EXTREME_HIGH"

    def _trend_dir(self, closes: pd.Series, sma_len: int) -> str:
        if len(closes) < sma_len + 2:
            return "FLAT"
        sma = float(closes.tail(sma_len).mean())
        p = float(closes.iloc[-1])
        if sma <= 0:
            return "FLAT"
        if p > sma * 1.0005:
            return "UP"
        if p < sma * 0.9995:
            return "DOWN"
        return "FLAT"

    # ----------------------------
    # Knobs / config (reads from DB state)
    # ----------------------------
    def _get_knob(self, key: str, default: Any) -> Any:
        v = self.db.get_state(self.con, key)
        if v is None:
            return default
        # try parse booleans/nums
        s = str(v).strip().lower()
        if isinstance(default, bool):
            return s in ("1", "true", "yes", "y", "on")
        if isinstance(default, int):
            try:
                return int(float(s))
            except Exception:
                return default
        if isinstance(default, float):
            try:
                return float(s)
            except Exception:
                return default
        return v

    def _log_knobs_periodically(self, now_ms: int) -> Dict[str, Any]:
        if now_ms - self._last_knobs_log_ms < 10_000:
            return {}

        knobs = {
            "enable_dip_entry": self._get_knob("enable_dip_entry", True),
            "enable_micro_mom_entry": self._get_knob("enable_micro_mom_entry", True),
            "enable_fallback_entry": self._get_knob("enable_fallback_entry", False),
            "trend_filter": self._get_knob("trend_filter", True),
            "trend_sma_len": self._get_knob("trend_sma_len", 40),
            "sma_len": self._get_knob("sma_len", 8),
            "dip_entry_pct": self._get_knob("dip_entry_pct", 0.0008),
            "mom_up_bars": self._get_knob("mom_up_bars", 2),
            "min_rr": self._get_knob("min_rr", 1.1),
            "cooldown_seconds": self._get_knob("cooldown_seconds", 5),

            # NEW: low-vol scalper toggle
            "enable_chop_scalp": self._get_knob("enable_chop_scalp", True),
        }

        log.info(
            "KNOBS "
            + " ".join(f"{k}={v}" for k, v in knobs.items())
        )
        self._last_knobs_log_ms = now_ms
        return knobs

    # ----------------------------
    # Entry logic
    # ----------------------------
    def _validate_rr(self, entry: float, tp: float, stop: float, min_rr: float) -> Tuple[bool, float]:
        risk = entry - stop
        reward = tp - entry
        if entry <= 0 or risk <= 0 or reward <= 0:
            return False, 0.0
        rr = reward / risk
        return rr >= float(min_rr), float(rr)

    def _levels_for_signal(self, price: float, atr: float, vol: str, signal: str) -> Tuple[float, float, float]:
        """
        Two profiles:
        - CHOP_SCALP (EXTREME_LOW / LOW): tiny targets, tiny stops
        - NORMAL (MID/HIGH): your existing-ish style
        """
        atr = max(float(atr), price * 0.00015)

        if signal == "CHOP_SCALP":
            # tight and fast
            tp1 = price + atr * 0.9
            tp2 = price + atr * 1.4
            stop = price - atr * 0.9
            return float(tp1), float(tp2), float(stop)

        # normal
        tp1 = price + atr * (1.6 if vol in ("HIGH", "EXTREME_HIGH") else 1.2)
        tp2 = price + atr * (2.4 if vol in ("HIGH", "EXTREME_HIGH") else 1.8)
        stop = price - atr * (1.2 if vol in ("HIGH", "EXTREME_HIGH") else 1.0)
        return float(tp1), float(tp2), float(stop)

    def _entry_signal(
        self,
        df1: pd.DataFrame,
        price: float,
        knobs: Dict[str, Any],
        trend: str,
        vol: str,
    ) -> Tuple[Optional[str], str]:

        if df1 is None or len(df1) < 50:
            return None, "not_enough_bars"

        closes = self._sf(df1, "close")
        volumes = self._sf(df1, "volume")

        sma_len = int(knobs.get("sma_len", 8))
        sma = float(closes.tail(sma_len).mean()) if len(closes) >= sma_len else float(closes.mean())
        rsi = self._rsi(closes, 14)

        # Hard block only EXTREME_HIGH (crazy candles)
        if vol == "EXTREME_HIGH":
            return None, "blocked_extreme_high_vol"

        enable_chop = bool(knobs.get("enable_chop_scalp", True))

        # Trend filter: in DOWN, only allow chop scalps (low vol only)
        if bool(knobs.get("trend_filter", True)) and trend == "DOWN":
            if not (enable_chop and vol in ("EXTREME_LOW", "LOW")):
                return None, "blocked_by_trend_filter_down"

        # -------------------------------------------------
        # CHOP SCALP (original bounce)
        # -------------------------------------------------
        if enable_chop and vol in ("EXTREME_LOW", "LOW"):
            dip_pct = float(knobs.get("dip_entry_pct", 0.0008))

            below_sma = sma > 0 and price <= sma * (1.0 - dip_pct * 0.6)
            bouncing = closes.iloc[-1] > closes.iloc[-2]
            rsi_ok = rsi < 58  # slightly looser

            if below_sma and bouncing and rsi_ok:
                return "CHOP_SCALP", "chop_scalp_bounce_ok"

            # -------------------------------------------------
            # NEW: CHOP_PULSE entry (more frequent)
            # Mean reversion: allow entry when slightly below SMA even without bounce
            # but require "some movement" (avoid buying dead-flat ticks)
            # -------------------------------------------------
            pulse_pct = float(knobs.get("chop_pulse_pct", 0.00025))  # 0.025%
            min_move_pct = float(knobs.get("chop_min_move_pct", 0.00010))  # 0.01%

            # how much did price move over last 3 bars?
            if len(closes) >= 4:
                move3 = abs(float(closes.iloc[-1]) - float(closes.iloc[-4]))
                move3_pct = move3 / price if price > 0 else 0.0
            else:
                move3_pct = 0.0

            slightly_below = sma > 0 and price <= sma * (1.0 - pulse_pct)
            rsi_mid = 40 <= rsi <= 62

            if slightly_below and rsi_mid and move3_pct >= min_move_pct:
                return "CHOP_PULSE", "chop_pulse_ok"

        # -------------------------------------------------
        # Dip entry (normal)
        # -------------------------------------------------
        if bool(knobs.get("enable_dip_entry", True)):
            dip_pct = float(knobs.get("dip_entry_pct", 0.0008))
            below_sma = sma > 0 and price <= sma * (1.0 - dip_pct)
            bouncing = closes.iloc[-1] > closes.iloc[-2]
            rsi_oversold = rsi < 40

            avg_vol = float(volumes.tail(20).mean()) if len(volumes) >= 20 else float(volumes.mean())
            vol_ok = (avg_vol <= 0) or (volumes.iloc[-1] >= avg_vol * 1.03)  # looser

            if below_sma and bouncing and rsi_oversold and vol_ok:
                return "DIP_BOUNCE", "dip_ok"

        # -------------------------------------------------
        # Micro momentum
        # -------------------------------------------------
        if bool(knobs.get("enable_micro_mom_entry", True)):
            mom_up = int(knobs.get("mom_up_bars", 2))
            if len(closes) >= mom_up + 2:
                ups = 0
                for i in range(1, mom_up + 1):
                    if closes.iloc[-i] > closes.iloc[-i - 1]:
                        ups += 1
                above_sma = sma > 0 and price > sma
                rsi_ok = 38 < rsi < 72  # slightly looser
                if ups == mom_up and above_sma and rsi_ok:
                    return "MICRO_MOM", "mom_ok"

        # -------------------------------------------------
        # Fallback trend
        # -------------------------------------------------
        if bool(knobs.get("enable_fallback_entry", False)):
            if trend in ("UP", "FLAT") and sma > 0 and price > sma * 1.0006 and 45 < rsi < 70:
                return "FALLBACK_TREND", "fallback_ok"

        return None, "no_conditions_met"


    # ----------------------------
    # Execution + DB state
    # ----------------------------
    def _cooldown_ok(self, now_ms: int, cooldown_seconds: int) -> bool:
        if self.last_trade_ms == 0:
            return True
        return (now_ms - self.last_trade_ms) >= int(cooldown_seconds) * 1000

    def _update_position_state(self) -> None:
        self.db.set_state(self.con, "position_json", json.dumps(asdict(self.position)))

    def _buy(self, now_ms: int, price: float, spend_usdt: float, signal: str) -> None:
        if spend_usdt <= 0:
            return
        if spend_usdt > self.usdt:
            spend_usdt = self.usdt
        if spend_usdt < float(getattr(self.s, "min_notional_usdt", 5.0)):
            return

        qty = spend_usdt / price

        # Live trading toggle (from settings)
        if bool(getattr(self.s, "live_trading", False)):
            self.ex.market_buy(self.s.symbol, spend_usdt)

        self.usdt -= spend_usdt
        self.kas += qty

        self.position.entry_ts = now_ms
        self.position.entry_price = price
        self.position.qty = qty
        self.position.entry_reason = signal
        self.position.peak_price = price

        self._update_position_state()

        log.info(f"BUY {qty:.2f} @ {price:.6f} signal={signal}")

        # write trade
        self.db.insert_trade(
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
            raw_json="{}",
        )

        self.last_trade_ms = now_ms

    def _sell_all(self, now_ms: int, price: float, reason: str) -> None:
        if self.position.qty <= 0:
            return

        qty = float(self.position.qty)

        if bool(getattr(self.s, "live_trading", False)):
            self.ex.market_sell(self.s.symbol, qty)

        proceeds = qty * price
        self.kas -= qty
        self.usdt += proceeds

        log.info(f"SELL {qty:.2f} @ {price:.6f} reason={reason}")

        self.db.insert_trade(
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
            raw_json="{}",
        )

        self.position = Position()
        self._update_position_state()
        self.last_trade_ms = now_ms

    def _update_trail(self, price: float, trail_dist: float) -> None:
        if self.position.qty <= 0:
            return
        if price > self.position.peak_price:
            self.position.peak_price = price
        trail_level = self.position.peak_price * (1.0 - trail_dist)
        if self.position.trail <= 0:
            self.position.trail = trail_level
        else:
            self.position.trail = max(self.position.trail, trail_level)

    # ----------------------------
    # Main loop step
    # ----------------------------
    def step(self) -> None:
        now_ms = int(time.time() * 1000)

        knobs = self._log_knobs_periodically(now_ms)
        if not knobs:
            # still need latest knobs for logic even if we don't log
            knobs = {
                "enable_dip_entry": self._get_knob("enable_dip_entry", True),
                "enable_micro_mom_entry": self._get_knob("enable_micro_mom_entry", True),
                "enable_fallback_entry": self._get_knob("enable_fallback_entry", False),
                "trend_filter": self._get_knob("trend_filter", True),
                "trend_sma_len": self._get_knob("trend_sma_len", 40),
                "sma_len": self._get_knob("sma_len", 8),
                "dip_entry_pct": self._get_knob("dip_entry_pct", 0.0008),
                "mom_up_bars": self._get_knob("mom_up_bars", 2),
                "min_rr": self._get_knob("min_rr", 1.1),
                "cooldown_seconds": self._get_knob("cooldown_seconds", 5),
                "enable_chop_scalp": self._get_knob("enable_chop_scalp", True),
            }

        # fetch data
        rows = self.ex.get_klines_1m(self.s.symbol, limit=160)
        df1 = pd.DataFrame(rows)
        price = float(df1["close"].iloc[-1])

        closes = self._sf(df1, "close")
        highs = self._sf(df1, "high")
        lows = self._sf(df1, "low")

        vol = self._compute_vol_regime(df1)
        trend = self._trend_dir(closes, int(knobs.get("trend_sma_len", 40)))
        atr = self._atr(highs, lows, closes, 14)

        # exits if in position
        if self.position.qty > 0:
            # tight trailing in chop
            trail_dist = 0.0010 if vol in ("EXTREME_LOW", "LOW") else 0.0016
            self._update_trail(price, trail_dist)

            # exit rules
            stop_level = max(self.position.stop, self.position.trail)
            if stop_level > 0 and price <= stop_level:
                self._sell_all(now_ms, price, "STOP/TRAIL")
                return

            # time exit: shorter in chop
            max_hold = 60 if vol in ("EXTREME_LOW", "LOW") else 150
            if (now_ms - self.position.entry_ts) >= max_hold * 1000:
                self._sell_all(now_ms, price, "TIME_EXIT")
                return

            # tp2 exit
            if self.position.tp2 > 0 and price >= self.position.tp2:
                self._sell_all(now_ms, price, "TP2")
                return

            # tp1 -> tighten stop (acts like partial without partial complexity)
            if (not self.position.tp1_done) and self.position.tp1 > 0 and price >= self.position.tp1:
                self.position.tp1_done = True
                # raise stop to near breakeven
                self.position.stop = max(self.position.stop, self.position.entry_price * 0.9995)
                self._update_position_state()
                log.info("TP1 reached -> tightened stop")
            return

        # entries if flat
        if not self._cooldown_ok(now_ms, int(knobs.get("cooldown_seconds", 5))):
            log.info(f"NO_ENTRY signal=None trend={trend} vol={vol} reason=cooldown")
            return

        signal, why = self._entry_signal(df1, price, knobs, trend, vol)
        if signal is None:
            log.info(f"NO_ENTRY signal=None trend={trend} vol={vol} reason={why}")
            return

        tp1, tp2, stop = self._levels_for_signal(price, atr, vol, signal)
        ok, rr = self._validate_rr(price, tp1, stop, float(knobs.get("min_rr", 1.1)))
        if not ok:
            log.info(f"NO_ENTRY signal={signal} trend={trend} vol={vol} reason=rr_low rr={rr:.2f}")
            return

        # size
        min_frac = float(getattr(self.s, "min_trade_frac", 0.04))
        max_frac = float(getattr(self.s, "max_trade_frac", 0.12))
        frac = min(max_frac, max(min_frac, 0.06))
        # smaller in chop
        if signal == "CHOP_SCALP":
            frac *= 0.6

        spend = self.usdt * frac

        # set position levels before buy
        self.position.tp1 = tp1
        self.position.tp2 = tp2
        self.position.stop = stop
        self.position.trail = 0.0

        self._buy(now_ms, price, spend, signal)

    def run_forever(self) -> None:
        log.info("BOT STARTED")
        poll = int(getattr(self.s, "poll_seconds", 2))
        while True:
            try:
                self.step()
            except Exception as e:
                log.exception(f"Loop error: {e}")
            time.sleep(poll)
