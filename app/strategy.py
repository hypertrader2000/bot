from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from .indicators import ema, rsi, atr

@dataclass
class Signal:
    action: str  # BUY / SELL / HOLD
    reason: str
    stop_price: float | None = None
    take_profit: float | None = None

def build_df(klines) -> pd.DataFrame:
    # klines list: [openTime, open, high, low, close, volume, closeTime, quoteAssetVolume, ...]
    rows = []
    for k in klines:
        rows.append({
            "open_time": int(k[0]),
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
            "close_time": int(k[6]),
        })
    df = pd.DataFrame(rows)
    return df

def compute_indicators(df: pd.DataFrame, fast: int, slow: int, rsi_len: int, atr_len: int) -> pd.DataFrame:
    df = df.copy()
    df["ema_fast"] = ema(df["close"], fast)
    df["ema_slow"] = ema(df["close"], slow)
    df["rsi"] = rsi(df["close"], rsi_len)
    df["atr"] = atr(df, atr_len)
    return df

def generate_signal(
    df_1m: pd.DataFrame,
    df_15m: pd.DataFrame,
    *,
    fast_ema: int,
    slow_ema: int,
    rsi_len: int,
    rsi_buy_max: float,
    rsi_sell_min: float,
    atr_len: int,
    stop_atr_mult: float,
    take_profit_atr_mult: float,
    has_position: bool,
    entry_price: float | None,
    current_stop: float | None,
) -> Signal:
    # indicator frames
    d1 = compute_indicators(df_1m, fast_ema, slow_ema, rsi_len, atr_len)
    d15 = compute_indicators(df_15m, fast_ema, slow_ema, rsi_len, atr_len)

    last = d1.iloc[-1]
    prev = d1.iloc[-2] if len(d1) >= 2 else last

    last15 = d15.iloc[-1]

    trend_bull = last15["ema_fast"] > last15["ema_slow"]
    trend_bear = last15["ema_fast"] < last15["ema_slow"]

    cross_up = prev["ema_fast"] <= prev["ema_slow"] and last["ema_fast"] > last["ema_slow"]
    cross_dn = prev["ema_fast"] >= prev["ema_slow"] and last["ema_fast"] < last["ema_slow"]

    r = float(last["rsi"]) if pd.notna(last["rsi"]) else 50.0
    a = float(last["atr"]) if pd.notna(last["atr"]) else 0.0
    price = float(last["close"])

    # If in position, prioritize risk exits (stop) elsewhere; here gives strategy exit suggestion.
    if has_position:
        # optional momentum exit: cross down OR RSI dropping below sell_min in bearish trend
        if cross_dn and trend_bear and r < 55:
            return Signal("SELL", f"Trend/momentum reversal (15m bear + 1m cross down, rsi={r:.1f})")
        # take profit suggestion
        if entry_price and a > 0:
            tp = entry_price + take_profit_atr_mult * a
            return Signal("HOLD", f"In position (rsi={r:.1f})", take_profit=tp)
        return Signal("HOLD", f"In position (rsi={r:.1f})")

    # No position -> look to buy
    if trend_bull and cross_up and r <= rsi_buy_max:
        stop = price - stop_atr_mult * a if a > 0 else None
        tp = price + take_profit_atr_mult * a if a > 0 else None
        return Signal("BUY", f"Momentum entry (15m bull + 1m cross up, rsi={r:.1f})", stop_price=stop, take_profit=tp)

    # No position -> optional short not supported in spot (skip)
    return Signal("HOLD", f"No setup (trend_bull={trend_bull}, cross_up={cross_up}, rsi={r:.1f})")
