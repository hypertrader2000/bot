class MomoVolumeStrategy:
    """
    BUY:
      - Breakout: ret5 > min_ret5 AND trend > trend_floor AND vol_z > vol_floor
      - Dip scalp: ret1 < -dip_ret1 AND trend > dip_trend_floor AND vol_z < dip_volz_max  (avoid falling knife)

    SELL:
      - trend breakdown
      - high-volume down impulse (optionally delayed by grace period in bot loop)
    """

    def __init__(
        self,
        # breakout
        min_ret5_buy: float = 0.00018,
        trend_floor: float = -0.00008,
        vol_floor: float = -0.80,
        # dip scalp
        dip_ret1: float = 0.00035,
        dip_trend_floor: float = -0.00010,
        dip_volz_max: float = 1.40,   # <-- NEW: block dip buys on extreme vol spikes
        # momentum flip
        flip_ret1: float = 0.00018,
        # exits
        sell_trend_ceiling: float = -0.00018,
        sell_vol_z: float = 1.60,
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

            # B) Dip scalp — ONLY if volume isn't a panic spike
            if (ret1 < -self.dip_ret1) and (trend > self.dip_trend_floor) and (vol_z <= self.dip_volz_max):
                return "BUY"

            # C) Momentum flip
            if ret1 > self.flip_ret1 and ret5 > 0.0 and trend > self.trend_floor:
                return "BUY"

            return "HOLD"

        # In position: exits
        if trend < self.sell_trend_ceiling:
            return "SELL"

        # High-volume down impulse exit
        if vol_z > self.sell_vol_z and ret1 < self.sell_ret1:
            return "SELL"

        return "HOLD"
