from __future__ import annotations

import json
import logging
import random
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd

from .config import Settings
from .mexc_api import MexcClient
from .strategy import build_df
from . import db as dbm

log = logging.getLogger("bot")


# -------------------------
# Strategy: MOMO + Volume Z
# -------------------------
class MomoVolumeStrategy:
    """
    KAS 1m scalper logic (works in LOW/MID regimes too)

    BUY if EITHER:
      A) Breakout: ret5 > +min_ret5 AND trend > trend_floor AND vol_z > vol_floor
      B) Dip scalp: ret1 < -dip_ret1 AND trend > dip_trend_floor AND vol_z <= dip_volz_max (panic filter)
      C) Momentum flip: ret1 > flip_ret1 AND ret5 > 0 AND trend > trend_floor

    SELL if:
      - trend < sell_trend_ceiling, OR
      - high-volume down impulse while in position
    """

    def __init__(
        self,
        # breakout
        min_ret5_buy: float = 0.00018,
        trend_floor: float = -0.00008,
        vol_floor: float = -0.80,
        # dip scalp
        dip_ret1: float = 0.00035,         # buy when ret1 < -0.00035
        dip_trend_floor: float = -0.00010,
        dip_volz_max: float = 1.40,        # NEW: block dip buys on panic volume spikes
        # momentum flip
        flip_ret1: float = 0.00018,
        # exits
        sell_trend_ceiling: float = -0.00018,
        sell_vol_z: float = 1.6,
        sell_ret1: float = -0.00070,
    ):
        self.min_ret5_buy = float(min_ret5_buy)
        self.trend_floor = float(trend_floor)
        self.vol_floor = float(vol_floor)

        self.dip_ret1 = float(dip_ret1)
        self.dip_trend_floor = float(dip_trend_floor)
        self.dip_volz_max = float(dip_volz_max)

        self.flip_ret1 = float(flip_ret1)

        self.sell_trend_ceiling = float(sell_trend_ceiling)
        self.sell_vol_z = float(sell_vol_z)
        self.sell_ret1 = float(sell_ret1)

    def decide(self, features: Dict[str, float], position_open: bool, regime: str = "MID") -> str:
        if not features:
            return "HOLD"

        trend = float(features["trend"])
        vol_z = float(features["vol_z"])
        ret5 = float(features["ret_5"])
        ret1 = float(features["ret_1"])

        if not position_open:
            # A) Breakout
            if ret5 > self.min_ret5_buy and trend > self.trend_floor and vol_z > self.vol_floor:
                return "BUY"

            # B) Dip scalp (mean reversion) + falling-knife filter
            if (ret1 < -self.dip_ret1) and (trend > self.dip_trend_floor) and (vol_z <= self.dip_volz_max):
                return "BUY"

            # C) Momentum flip
            if ret1 > self.flip_ret1 and ret5 > 0.0 and trend > self.trend_floor:
                return "BUY"

            return "HOLD"

        # In position: exits
        if trend < self.sell_trend_ceiling:
            return "SELL"

        if vol_z > self.sell_vol_z and ret1 < self.sell_ret1:
            return "SELL"

        return "HOLD"


# -------------------------
# Position + Bot
# -------------------------
@dataclass
class Position:
    qty: float = 0.0
    cost_usdt: float = 0.0
    entry_price: float = 0.0
    entry_ts: int = 0
    last_mark_ts: int = 0

    # NEW: risk + trailing
    tp_pct: float = 0.0
    sl_pct: float = 0.0
    trail_active: bool = False
    trail_high: float = 0.0
    trail_stop: float = 0.0

    @property
    def avg_price(self) -> float:
        return (self.cost_usdt / self.qty) if self.qty > 0 else 0.0


class KaspaBot:
    def __init__(self, settings: Settings):
        self.s = settings
        self.client = MexcClient(settings.api_key, settings.api_secret)
        self.con = dbm.connect()

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self.base_asset, self.quote_asset = self._split_symbol(settings.symbol)

        # strategy
        self.momo_enabled = bool(getattr(self.s, "momo_enabled", True))
        self.momo_window = int(getattr(self.s, "momo_window", 50))

        # New knobs (safe defaults)
        self.momo_exit_grace_sec = int(getattr(self.s, "momo_exit_grace_sec", 8))

        self.momo = MomoVolumeStrategy(
            min_ret5_buy=float(getattr(self.s, "min_ret5_buy", 0.00018)),
            trend_floor=float(getattr(self.s, "trend_floor", -0.00008)),
            vol_floor=float(getattr(self.s, "vol_floor", -0.80)),
            dip_ret1=float(getattr(self.s, "dip_ret1", 0.00035)),
            dip_trend_floor=float(getattr(self.s, "dip_trend_floor", -0.00010)),
            dip_volz_max=float(getattr(self.s, "dip_volz_max", 1.40)),
            flip_ret1=float(getattr(self.s, "flip_ret1", 0.00018)),
            sell_trend_ceiling=float(getattr(self.s, "sell_trend_ceiling", -0.00018)),
            sell_vol_z=float(getattr(self.s, "sell_vol_z", 1.6)),
            sell_ret1=float(getattr(self.s, "sell_ret1", -0.00070)),
        )

        log.info("MOMO enabled=%s window=%d", self.momo_enabled, self.momo_window)

        # state
        self.position = self._load_position()

        self.last_trade_ts = int(dbm.get_state(self.con, "last_trade_ts") or "0")
        self._last_any_trade_ms = int(dbm.get_state(self.con, "last_any_trade_ms") or "0")

        sim_q = dbm.get_state(self.con, "sim_quote")
        sim_b = dbm.get_state(self.con, "sim_base")
        self.sim_quote = float(sim_q) if sim_q is not None else float(getattr(self.s, "sim_start_quote_usdt", 10000.0))
        self.sim_base = float(sim_b) if sim_b is not None else 0.0

        base_eq = dbm.get_state(self.con, "sim_base_equity")
        self.base_equity = float(base_eq) if base_eq is not None else float(self.sim_quote)

        peak_eq = dbm.get_state(self.con, "sim_peak_equity")
        self.peak_equity = float(peak_eq) if peak_eq is not None else self.base_equity

        self.loss_streak = int(dbm.get_state(self.con, "loss_streak") or "0")
        self.pause_until_ms = int(dbm.get_state(self.con, "pause_until_ms") or "0")
        self.pause_reason = dbm.get_state(self.con, "pause_reason") or ""

        # sizing
        self.min_trade_frac = float(getattr(self.s, "min_trade_frac", 0.04))
        self.max_trade_frac = float(getattr(self.s, "max_trade_frac", 0.12))
        self.max_position_fraction = float(getattr(self.s, "max_position_fraction", 0.25))

        # exits (baseline percent)
        self.quick_tp = float(getattr(self.s, "quick_tp", 0.0012))         # +0.12%
        self.stop_loss = float(getattr(self.s, "stop_loss", 0.0016))       # -0.16%

        # NEW: dynamic TP/SL scaling by volatility
        self.tp_mult = float(getattr(self.s, "tp_mult", 1.15))
        self.sl_mult = float(getattr(self.s, "sl_mult", 1.35))

        # NEW: trailing stop
        self.trail_activate = float(getattr(self.s, "trail_activate", 0.0015))  # activate after +0.15%
        self.trail_dist = float(getattr(self.s, "trail_dist", 0.0010))          # trail by 0.10%

        # breakeven protect (keep yours)
        self.breakeven_arm = float(getattr(self.s, "breakeven_arm", 0.0008))    # arm BE after +0.08%
        self.breakeven_pad = float(getattr(self.s, "breakeven_pad", 0.0001))    # sell if falls below entry+0.01%

        # time exits
        self.max_hold_sec = int(getattr(self.s, "max_hold_sec", 120))            # soft max
        self.hard_max_hold_sec = int(getattr(self.s, "hard_max_hold_sec", 240))  # hard max
        self.min_time_exit_pnl = float(getattr(self.s, "min_time_exit_pnl", -0.0003))

        # trading cadence
        self.poll_seconds = int(getattr(self.s, "poll_seconds", 1))
        self.cooldown_seconds = int(getattr(self.s, "cooldown_seconds", 0))

        # logging throttle
        self._last_log_ms = 0
        self.vol_regime = "MID"

    # ---------------- lifecycle ----------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        try:
            self.client.sync_time()
        except Exception:
            pass
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    # ---------------- utils ----------------
    @staticmethod
    def _split_symbol(symbol: str) -> Tuple[str, str]:
        quotes = ["USDT", "USDC", "BTC", "ETH"]
        for q in quotes:
            if symbol.endswith(q):
                return symbol[:-len(q)], q
        return symbol[:-4], symbol[-4:]

    def _load_position(self) -> Position:
        raw = dbm.get_state(self.con, "position_json")
        if not raw:
            return Position()
        try:
            return Position(**json.loads(raw))
        except Exception:
            return Position()

    def _save_position(self) -> None:
        dbm.set_state(self.con, "position_json", json.dumps(self.position.__dict__))

    def _save_sim(self) -> None:
        dbm.set_state(self.con, "sim_quote", str(self.sim_quote))
        dbm.set_state(self.con, "sim_base", str(self.sim_base))
        dbm.set_state(self.con, "sim_base_equity", str(self.base_equity))
        dbm.set_state(self.con, "sim_peak_equity", str(self.peak_equity))
        dbm.set_state(self.con, "loss_streak", str(self.loss_streak))
        dbm.set_state(self.con, "pause_until_ms", str(self.pause_until_ms))
        dbm.set_state(self.con, "pause_reason", self.pause_reason)
        dbm.set_state(self.con, "last_trade_ts", str(self.last_trade_ts))
        dbm.set_state(self.con, "last_any_trade_ms", str(self._last_any_trade_ms))

    def _cooldown_ok(self, now_ms: int) -> bool:
        return (now_ms - self.last_trade_ts) >= (self.cooldown_seconds * 1000)

    def _in_pause(self, now_ms: int) -> bool:
        return now_ms < int(self.pause_until_ms or 0)

    def _set_pause(self, now_ms: int, seconds: int, reason: str) -> None:
        self.pause_until_ms = now_ms + int(seconds * 1000)
        self.pause_reason = reason
        self._save_sim()

    def _update_equity(self, now_ms: int, price: float) -> float:
        equity = float(self.sim_quote) + float(self.sim_base) * float(price)
        unreal = 0.0
        if self.position.qty > 0:
            unreal = (float(price) - float(self.position.avg_price)) * float(self.position.qty)

        dbm.upsert_equity(
            self.con,
            ts=now_ms,
            equity_usdt=float(equity),
            realized_pnl_usdt=0.0,
            unrealized_pnl_usdt=float(unreal),
            price=float(price),
        )
        return float(equity)

    # ---------------- features ----------------
    def _compute_features(self, df1: pd.DataFrame) -> Dict[str, float]:
        """
        ret_1: 1-bar return (close[-1]/close[-2]-1)
        ret_5: 5-bar return
        trend: slope of linear fit over last N closes (per bar, normalized by price)
        vol_z: z-score of last volume vs window mean/std

        NEW:
        atr: ATR-like proxy from rolling std of returns
        """
        if df1 is None or len(df1) < max(6, self.momo_window + 2):
            return {}

        closes = df1["close"].astype(float).values
        vols = df1["volume"].astype(float).values

        c1 = closes[-1]
        c2 = closes[-2]
        c6 = closes[-6]

        ret1 = (c1 / c2) - 1.0 if c2 > 0 else 0.0
        ret5 = (c1 / c6) - 1.0 if c6 > 0 else 0.0

        w = self.momo_window
        y = closes[-w:]
        x = np.arange(w, dtype=float)
        slope = np.polyfit(x, y, 1)[0]  # price units per bar
        trend = (slope / c1) if c1 > 0 else 0.0  # normalize by price

        v = vols[-w:]
        v_mean = float(np.mean(v))
        v_std = float(np.std(v)) if float(np.std(v)) > 1e-9 else 1e-9
        vol_z = (float(v[-1]) - v_mean) / v_std

        # ATR-like: std of returns * factor
        rets = (y[1:] / y[:-1]) - 1.0
        rstd = float(np.std(rets)) if len(rets) >= 10 else float(np.std((closes[1:] / closes[:-1]) - 1.0)) if len(closes) >= 20 else 0.0
        atr = max(rstd * 2.2, 1e-9)

        return {
            "ret_1": float(ret1),
            "ret_5": float(ret5),
            "trend": float(trend),
            "vol_z": float(vol_z),
            "atr": float(atr),
        }

    # ---------------- trading actions (DRY) ----------------
    def _dry_trade(self, now_ms: int, side: str, qty: float, price: float, reason: str) -> None:
        fake_id = f"DRY-{now_ms}-{side}"
        dbm.insert_trade(
            self.con,
            ts=now_ms,
            symbol=self.s.symbol,
            side=side,
            qty=float(qty),
            price=float(price),
            fee=0.0,
            fee_asset=self.quote_asset,
            mexc_trade_id=fake_id,
            mexc_order_id=fake_id,
            raw_json=json.dumps({"reason": reason}),
        )

    def _pick_entry_fraction(self, equity: float, price: float) -> float:
        # random fraction bounded
        frac = random.uniform(self.min_trade_frac, self.max_trade_frac)

        # enforce max exposure cap
        current_pos_value = float(self.sim_base) * float(price)
        max_allowed_value = float(equity) * float(self.max_position_fraction)
        remaining = max(0.0, max_allowed_value - current_pos_value)
        desired = float(equity) * float(frac)

        spend = min(desired, remaining, float(self.sim_quote))
        if equity <= 0:
            return 0.0
        return max(0.0, spend / equity)

    def _dyn_tp_sl(self, feats: Dict[str, float]) -> Tuple[float, float]:
        """
        Dynamic TP/SL based on ATR-like proxy in feats.
        """
        atr = float(feats.get("atr", 0.0))
        tp = max(self.quick_tp, atr * self.tp_mult)
        sl = max(self.stop_loss, atr * self.sl_mult)
        return float(tp), float(sl)

    def _buy(self, now_ms: int, price: float, frac: float, reason: str, feats: Optional[Dict[str, float]] = None) -> None:
        equity = float(self.sim_quote) + float(self.sim_base) * float(price)
        spend = float(equity) * float(frac)
        spend = max(0.0, min(spend, float(self.sim_quote)))

        if spend <= 5.0:
            return

        qty = spend / float(price)

        self.sim_quote -= spend
        self.sim_base += qty

        tp_pct, sl_pct = self._dyn_tp_sl(feats or {})

        self.position = Position(
            qty=float(qty),
            cost_usdt=float(spend),
            entry_price=float(price),
            entry_ts=int(now_ms),
            last_mark_ts=int(now_ms),

            tp_pct=float(tp_pct),
            sl_pct=float(sl_pct),
            trail_active=False,
            trail_high=float(price),
            trail_stop=0.0,
        )
        self._save_position()

        self.last_trade_ts = int(now_ms)
        self._last_any_trade_ms = int(now_ms)
        self._save_sim()

        log.info("ENTRY RISK: tp=%.4f sl=%.4f (atr-based)", self.position.tp_pct, self.position.sl_pct)
        log.info("🔥 [DRY] BUY q=%.6f p=%.6f spend=%.2f frac=%.3f (%s)",
                 qty, price, spend, frac, reason)
        self._dry_trade(now_ms, "BUY", qty, price, reason)

    def _sell(self, now_ms: int, price: float, reason: str) -> None:
        if self.position.qty <= 0:
            return

        qty = float(self.position.qty)
        proceeds = qty * float(price)

        # realized PnL in USDT on this position
        realized = proceeds - float(self.position.cost_usdt)

        self.sim_base = max(0.0, float(self.sim_base) - qty)
        self.sim_quote += proceeds

        # loss streak / pause logic
        if realized < 0:
            self.loss_streak += 1
            self._set_pause(now_ms, 60, "loss_pause")
        else:
            self.loss_streak = 0
            self._set_pause(now_ms, 20, "win_pause")

        # clear position
        self.position = Position()
        self._save_position()

        self.last_trade_ts = int(now_ms)
        self._last_any_trade_ms = int(now_ms)
        self._save_sim()

        tag = "✅" if realized >= 0 else "❌"
        log.info("%s [DRY] SELL q=%.6f p=%.6f reason=%s realized=%.4f",
                 tag, qty, price, reason, realized)
        self._dry_trade(now_ms, "SELL", qty, price, reason)

    def _update_trailing(self, price: float) -> None:
        """
        Trailing stop that activates once pnl >= trail_activate.
        """
        if self.position.qty <= 0 or self.position.entry_price <= 0:
            return

        pnl = (float(price) / float(self.position.entry_price)) - 1.0

        # update high
        if float(price) > float(self.position.trail_high or 0.0):
            self.position.trail_high = float(price)

        # activate trailing
        if (not self.position.trail_active) and (pnl >= self.trail_activate):
            self.position.trail_active = True
            self.position.trail_high = float(price)

        if self.position.trail_active:
            self.position.trail_stop = float(self.position.trail_high) * (1.0 - float(self.trail_dist))

    # ---------------- main loop ----------------
    def run(self) -> None:
        log.info("Bot started. symbol=%s live_trading=%s (DRY)",
                 self.s.symbol, getattr(self.s, "live_trading", False))

        last_kline = 0
        df1: Optional[pd.DataFrame] = None

        while not self._stop.is_set():
            now_ms = int(time.time() * 1000)

            # price
            try:
                price = float(self.client.ticker_price(self.s.symbol))
            except Exception:
                time.sleep(1)
                continue

            # equity + peak
            equity = self._update_equity(now_ms, price)
            self.peak_equity = max(self.peak_equity, equity)

            # refresh klines
            if df1 is None or (now_ms - last_kline) > 30_000:
                try:
                    k1 = self.client.klines(self.s.symbol, "1m", 400)
                    df1 = build_df(k1)
                    last_kline = now_ms
                except Exception:
                    pass

            feats = self._compute_features(df1) if (self.momo_enabled and df1 is not None) else {}
            if feats:
                log.info("MOMO feats ret5=%.5f vol_z=%.2f trend=%.6f",
                         feats["ret_5"], feats["vol_z"], feats["trend"])

            # periodic status log
            if now_ms - self._last_log_ms > 30_000:
                self._last_log_ms = now_ms
                dd = 0.0
                if self.peak_equity > 0:
                    dd = max(0.0, (self.peak_equity - equity) / self.peak_equity)
                pause_left = max(0, int((self.pause_until_ms - now_ms) / 1000)) if self._in_pause(now_ms) else 0
                log.info("REGIME=%s eq=%.2f peak=%.2f dd=%.2f%% USDT=%.2f KAS=%.2f pause=%ss loss_streak=%d rand_frac=[%.2f..%.2f] max_pos=%.2f",
                         self.vol_regime, equity, self.peak_equity, dd * 100.0,
                         self.sim_quote, self.sim_base, pause_left, self.loss_streak,
                         self.min_trade_frac, self.max_trade_frac, self.max_position_fraction)

            # -----------------
            # Position management
            # -----------------
            if self.position.qty > 0:
                pnl_pct = (float(price) / float(self.position.entry_price) - 1.0) if self.position.entry_price > 0 else 0.0
                hold_ms = now_ms - int(self.position.entry_ts or now_ms)
                hold_sec = hold_ms / 1000.0

                # keep trailing updated
                self._update_trailing(price)
                if self.position.trail_active:
                    log.info("TRAIL active=%s high=%.6f stop=%.6f pnl=%.4f",
                             self.position.trail_active, self.position.trail_high, self.position.trail_stop, pnl_pct)

                # Dynamic TP/SL (stored at entry)
                tp = float(self.position.tp_pct or self.quick_tp)
                sl = float(self.position.sl_pct or self.stop_loss)

                # QUICK TP
                if pnl_pct >= tp:
                    self._sell(now_ms, price, "DYN_TP")
                    time.sleep(self.poll_seconds)
                    continue

                # STOP LOSS
                if pnl_pct <= -sl:
                    self._sell(now_ms, price, "DYN_SL")
                    time.sleep(self.poll_seconds)
                    continue

                # Trailing stop (once active)
                if self.position.trail_active and float(price) <= float(self.position.trail_stop):
                    self._sell(now_ms, price, "TRAIL_STOP")
                    time.sleep(self.poll_seconds)
                    continue

                # Breakeven protect after small profit (keep yours)
                if pnl_pct >= self.breakeven_arm:
                    be_level = self.position.entry_price * (1.0 + self.breakeven_pad)
                    if float(price) <= float(be_level):
                        self._sell(now_ms, price, "BREAKEVEN_PROTECT")
                        time.sleep(self.poll_seconds)
                        continue

                # Strategy-driven SELL (with grace period)
                if feats:
                    sig = self.momo.decide(feats, position_open=True, regime=self.vol_regime)
                    log.info("MOMO chk regime=%s pos=OPEN sig=%s ret1=%.5f ret5=%.5f vol_z=%.2f trend=%.6f",
                             self.vol_regime, sig, feats["ret_1"], feats["ret_5"], feats["vol_z"], feats["trend"])
                    log.info("MOMO dbg: ret1=%.5f ret5=%.5f trend=%.6f vol_z=%.2f | flip_ret1=%.5f trend_floor=%.6f min_ret5=%.5f vol_floor=%.2f",
                             feats["ret_1"], feats["ret_5"], feats["trend"], feats["vol_z"],
                             self.momo.flip_ret1, self.momo.trend_floor, self.momo.min_ret5_buy, self.momo.vol_floor)

                    if sig == "SELL":
                        if hold_sec >= float(self.momo_exit_grace_sec):
                            self._sell(now_ms, price, "MOMO_EXIT")
                            time.sleep(self.poll_seconds)
                            continue
                        else:
                            log.info("MOMO_EXIT blocked by grace: hold=%.1fs < %ss", hold_sec, self.momo_exit_grace_sec)

                # SMART TIME EXIT:
                if hold_ms >= (self.max_hold_sec * 1000):
                    if pnl_pct >= self.min_time_exit_pnl:
                        self._sell(now_ms, price, "TIME_EXIT")
                        time.sleep(self.poll_seconds)
                        continue

                # Hard max hold no matter what (risk control)
                if hold_ms >= (self.hard_max_hold_sec * 1000):
                    self._sell(now_ms, price, "HARD_TIME_EXIT")
                    time.sleep(self.poll_seconds)
                    continue

                time.sleep(self.poll_seconds)
                continue

            # -----------------
            # Entries
            # -----------------
            cooldown_ok = self._cooldown_ok(now_ms)
            in_pause = self._in_pause(now_ms)
            log.info("ENTRY GATES: cooldown_ok=%s in_pause=%s", cooldown_ok, in_pause)

            if self.position.qty <= 0 and cooldown_ok and (not in_pause):
                if feats:
                    sig = self.momo.decide(feats, position_open=False, regime=self.vol_regime)

                    log.info("MOMO chk regime=%s pos=FLAT sig=%s ret1=%.5f ret5=%.5f vol_z=%.2f trend=%.6f",
                             self.vol_regime, sig, feats["ret_1"], feats["ret_5"], feats["vol_z"], feats["trend"])
                    log.info("MOMO dbg: ret1=%.5f ret5=%.5f trend=%.6f vol_z=%.2f | flip_ret1=%.5f trend_floor=%.6f min_ret5=%.5f vol_floor=%.2f",
                             feats["ret_1"], feats["ret_5"], feats["trend"], feats["vol_z"],
                             self.momo.flip_ret1, self.momo.trend_floor, self.momo.min_ret5_buy, self.momo.vol_floor)

                    if sig == "BUY":
                        frac = self._pick_entry_fraction(equity, price)
                        if frac > 0:
                            log.info("ENTRY FINAL: reason=%s frac=%.4f regime=%s trend_ok=%s",
                                     "MOMO_VOL", frac, self.vol_regime, True)
                            self._buy(now_ms, price, frac, "MOMO_VOL", feats=feats)

            time.sleep(self.poll_seconds)

        log.info("Bot stopped.")
