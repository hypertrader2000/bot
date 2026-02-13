import numpy as np
import pandas as pd


class FeatureEngine:
    def __init__(self, window: int = 50):
        self.window = int(window)

    def compute(self, df: pd.DataFrame):
        """
        Expects df columns: open, high, low, close, volume
        Returns dict of momentum/volume/volatility features or None if insufficient data.
        """
        if df is None or len(df) < max(self.window, 60, 26, 14, 6):
            return None

        # Ensure numeric
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        volume = df["volume"].astype(float)

        # recent slice for stability/speed
        dlen = min(len(df), max(self.window, 80))
        close = close.tail(dlen)
        high = high.tail(dlen)
        low = low.tail(dlen)
        volume = volume.tail(dlen)

        # -------- MOMENTUM ----------
        c1 = float(close.iloc[-1])
        c2 = float(close.iloc[-2])
        c5 = float(close.iloc[-5])

        if c2 <= 0 or c5 <= 0:
            return None

        ret_1 = float(np.log(c1 / c2))
        ret_5 = float(np.log(c1 / c5))

        ema_fast = float(close.ewm(span=12, adjust=False).mean().iloc[-1])
        ema_slow = float(close.ewm(span=26, adjust=False).mean().iloc[-1])
        trend = float(ema_fast - ema_slow)

        # -------- VOLUME ----------
        vol_tail = volume.tail(30)
        vol_mean = float(vol_tail.mean())
        vol_std = float(vol_tail.std(ddof=0)) + 1e-9
        vol_z = float((float(volume.iloc[-1]) - vol_mean) / vol_std)

        # -------- VOLATILITY (ATR-ish) ----------
        atr = float((high - low).rolling(14).mean().iloc[-1])
        if not np.isfinite(atr):
            atr = float((high - low).tail(14).mean())

        return {
            "ret_1": ret_1,
            "ret_5": ret_5,
            "trend": trend,
            "vol_z": vol_z,
            "atr": atr,
        }
