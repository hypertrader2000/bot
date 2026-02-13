from __future__ import annotations
from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

momo_exit_grace_sec = 8
dip_volz_max = 1.40

tp_mult = 1.15
sl_mult = 1.35

trail_activate = 0.0015
trail_dist = 0.0010

def _get_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name, str(default)).strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _get_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, str(default))))
    except Exception:
        return default


@dataclass(frozen=True)
class Settings:
    api_key: str
    api_secret: str

    # ======================
    # CORE
    # ======================
    symbol: str = "KASUSDT"
    live_trading: bool = False
    poll_seconds: int = 2

    host: str = "127.0.0.1"
    port: int = 8000

    # ======================
    # SIM
    # ======================
    sim_start_usdt: float = 10000.0
    sim_start_quote_usdt: float = 10000.0  # alias (your bot uses this name sometimes)

    # ======================
    # POSITION SIZING
    # ======================
    min_trade_frac: float = 0.04
    max_trade_frac: float = 0.12
    max_position_fraction: float = 0.25
    min_notional_usdt: float = 5.0

    # ======================
    # ENTRY ENGINE
    # ======================
    enable_dip_entry: bool = True
    enable_micro_mom_entry: bool = True
    enable_fallback_entry: bool = True

    trend_filter: bool = True
    trend_sma_len: int = 40

    sma_len: int = 8
    dip_entry_pct: float = 0.0008
    mom_up_bars: int = 2

    # ======================
    # EXIT PROFILE
    # ======================
    scalp_stop: float = 0.0018
    scalp_tp1: float = 0.0035
    scalp_tp2: float = 0.0065
    scalp_trail: float = 0.0022

    tp1_sell_fraction: float = 0.60
    max_hold_sec: int = 120
    fallback_sec: int = 8

    # ======================
    # GUARD RAILS
    # ======================
    cooldown_seconds: int = 5

    max_trades_per_hour: int = 120
    pause_on_max_tph_sec: int = 900

    max_loss_streak: int = 6
    pause_on_loss_streak_sec: int = 1200

    # ======================
    # VOL FILTER
    # ======================
    vol_throttle: bool = True
    vol_window: int = 20
    vol_low: float = 0.0007
    vol_high: float = 0.0035

    # ======================
    # AUTO COMPOUNDING
    # ======================
    compound_enabled: bool = True
    compound_boost_per_10pct: float = 0.12
    drawdown_soft: float = 0.015
    drawdown_hard: float = 0.05
    compound_drawdown_cut: float = 0.40

    # ======================
    # OPTIONAL HARD STOPS
    # ======================
    goal_equity_usdt: float = 0.0
    stop_equity_usdt: float = 0.0

    # ======================
    # COST MODEL (recommended)
    # ======================
    taker_fee: float = 0.0004
    assumed_slippage: float = 0.0002
    min_edge: float = 0.0010

    # ---- (keep your existing fields too, harmless) ----
    fast_ema: int = 9
    slow_ema: int = 21
    rsi_len: int = 14
    rsi_buy_max: float = 68.0
    rsi_sell_min: float = 32.0
    atr_len: int = 14
    stop_atr_mult: float = 1.6
    trail_atr_mult: float = 1.4
    take_profit_atr_mult: float = 2.2

    max_daily_loss_usdt: float = 250.0

    # =====================
    # DRL STRATEGY (optional)
    # =====================
    # heuristic: use the existing handcrafted entries
    # drl:       use a PyTorch policy for entries (and optional exits)
    # hybrid:    drl must agree with heuristic to enter (extra safe)
    strategy_mode: str = "heuristic"  # heuristic | drl | hybrid
    drl_model_path: str = ""          # e.g. models/policy_kas.pt
    drl_min_conf: float = 0.55        # confidence threshold to act
    drl_obs_window: int = 64          # last N 1m bars used in obs


def get_settings() -> Settings:
    api_key = os.getenv("MEXC_API_KEY", "").strip()
    api_secret = os.getenv("MEXC_API_SECRET", "").strip()

    live_trading = _get_bool("LIVE_TRADING", False)
    if live_trading and (not api_key or not api_secret):
        raise RuntimeError("LIVE_TRADING=true but MEXC_API_KEY or MEXC_API_SECRET missing in .env")

    sim_start = _get_float("SIM_START_USDT", 10000.0)

    return Settings(
        api_key=api_key,
        api_secret=api_secret,

        symbol=os.getenv("SYMBOL", "KASUSDT").strip().upper(),
        live_trading=live_trading,
        poll_seconds=_get_int("POLL_SECONDS", 2),

        host=os.getenv("HOST", "127.0.0.1").strip(),
        port=_get_int("PORT", 8000),

        sim_start_usdt=sim_start,
        sim_start_quote_usdt=sim_start,  # keep both in sync

        min_trade_frac=_get_float("MIN_TRADE_FRAC", 0.04),
        max_trade_frac=_get_float("MAX_TRADE_FRAC", 0.12),
        max_position_fraction=_get_float("MAX_POSITION_FRACTION", 0.25),
        min_notional_usdt=_get_float("MIN_NOTIONAL_USDT", 5.0),

        enable_dip_entry=_get_bool("ENABLE_DIP_ENTRY", True),
        enable_micro_mom_entry=_get_bool("ENABLE_MICRO_MOM_ENTRY", True),
        enable_fallback_entry=_get_bool("ENABLE_FALLBACK_ENTRY", True),

        trend_filter=_get_bool("TREND_FILTER", True),
        trend_sma_len=_get_int("TREND_SMA_LEN", 40),

        sma_len=_get_int("SMA_LEN", 8),
        dip_entry_pct=_get_float("DIP_ENTRY_PCT", 0.0008),
        mom_up_bars=_get_int("MOM_UP_BARS", 2),

        scalp_stop=_get_float("SCALP_STOP", 0.0018),
        scalp_tp1=_get_float("SCALP_TP1", 0.0035),
        scalp_tp2=_get_float("SCALP_TP2", 0.0065),
        scalp_trail=_get_float("SCALP_TRAIL", 0.0022),

        tp1_sell_fraction=_get_float("TP1_SELL_FRACTION", 0.60),
        max_hold_sec=_get_int("MAX_HOLD_SEC", 120),
        fallback_sec=_get_int("FALLBACK_SEC", 8),

        cooldown_seconds=_get_int("COOLDOWN_SECONDS", 5),

        max_trades_per_hour=_get_int("MAX_TRADES_PER_HOUR", 120),
        pause_on_max_tph_sec=_get_int("PAUSE_ON_MAX_TPH_SEC", 900),

        max_loss_streak=_get_int("MAX_LOSS_STREAK", 6),
        pause_on_loss_streak_sec=_get_int("PAUSE_ON_LOSS_STREAK_SEC", 1200),

        vol_throttle=_get_bool("VOL_THROTTLE", True),
        vol_window=_get_int("VOL_WINDOW", 20),
        vol_low=_get_float("VOL_LOW", 0.0007),
        vol_high=_get_float("VOL_HIGH", 0.0035),

        compound_enabled=_get_bool("COMPOUND_ENABLED", True),
        compound_boost_per_10pct=_get_float("COMPOUND_BOOST_PER_10PCT", 0.12),
        drawdown_soft=_get_float("DRAWDOWN_SOFT", 0.015),
        drawdown_hard=_get_float("DRAWDOWN_HARD", 0.05),
        compound_drawdown_cut=_get_float("COMPOUND_DRAWDOWN_CUT", 0.40),

        goal_equity_usdt=_get_float("GOAL_EQUITY_USDT", 0.0),
        stop_equity_usdt=_get_float("STOP_EQUITY_USDT", 0.0),

        taker_fee=_get_float("TAKER_FEE", 0.0004),
        assumed_slippage=_get_float("ASSUMED_SLIPPAGE", 0.0002),
        min_edge=_get_float("MIN_EDGE", 0.0010),

        # keep your older indicators available
        fast_ema=_get_int("FAST_EMA", 9),
        slow_ema=_get_int("SLOW_EMA", 21),
        rsi_len=_get_int("RSI_LEN", 14),
        rsi_buy_max=_get_float("RSI_BUY_MAX", 68.0),
        rsi_sell_min=_get_float("RSI_SELL_MIN", 32.0),
        atr_len=_get_int("ATR_LEN", 14),
        stop_atr_mult=_get_float("STOP_ATR_MULT", 1.6),
        trail_atr_mult=_get_float("TRAIL_ATR_MULT", 1.4),
        take_profit_atr_mult=_get_float("TAKE_PROFIT_ATR_MULT", 2.2),

        max_daily_loss_usdt=_get_float("MAX_DAILY_LOSS_USDT", 250.0),

        # DRL strategy (optional)
        strategy_mode=os.getenv("STRATEGY_MODE", "heuristic").strip().lower(),
        drl_model_path=os.getenv("DRL_MODEL_PATH", "").strip(),
        drl_min_conf=_get_float("DRL_MIN_CONF", 0.55),
        drl_obs_window=_get_int("DRL_OBS_WINDOW", 64),
    )
