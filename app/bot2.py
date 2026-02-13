from __future__ import annotations

import time
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any

import pandas as pd

log = logging.getLogger("bot")


# =====================================================
# POSITION
# =====================================================

@dataclass
class Position:
    qty: float = 0.0
    entry_price: float = 0.0
    stop: float = 0.0
    tp: float = 0.0


# =====================================================
# BOT
# =====================================================

class KaspaBot:

    def __init__(self, settings, exchange, con):
        self.s = settings
        self.ex = exchange
        self.con = con

        # SIM balances
        self.usdt = float(getattr(settings, "sim_start_usdt", 10000))
        self.kas = 0.0

        self.position = Position()

        log.info("KaspaBot initialized")


# =====================================================
# DB HELPERS
# =====================================================

    def _insert_equity(self, ts: int, price: float):
        equity = self.usdt + self.kas * price

        self.con.execute(
            """
            INSERT OR REPLACE INTO equity
            (ts, equity_usdt, realized_pnl_usdt, unrealized_pnl_usdt, price)
            VALUES (?, ?, 0, 0, ?)
            """,
            (ts, equity, price),
        )
        self.con.commit()


    def _insert_trade(self, ts, side, qty, price):
        self.con.execute(
            """
            INSERT INTO trades
            (ts, symbol, side, qty, price, fee, raw_json)
            VALUES (?, ?, ?, ?, ?, 0, '')
            """,
            (ts, self.s.symbol, side, qty, price),
        )
        self.con.commit()


# =====================================================
# STRATEGY (simple but VERY stable)
# =====================================================

    def _signal(self, df: pd.DataFrame) -> Optional[str]:

        if len(df) < 30:
            return None

        closes = df["close"].astype(float)

        sma_fast = closes.tail(5).mean()
        sma_slow = closes.tail(20).mean()

        # CROSS UP → BUY
        if sma_fast > sma_slow and self.position.qty == 0:
            return "BUY"

        # CROSS DOWN → SELL
        if sma_fast < sma_slow and self.position.qty > 0:
            return "SELL"

        return None


# =====================================================
# EXECUTION
# =====================================================

    def step(self):

        now = int(time.time() * 1000)

        klines = self.ex.get_klines_1m(self.s.symbol, limit=100)
        df = pd.DataFrame(klines)

        price = float(df["close"].iloc[-1])

        signal = self._signal(df)

        # ---------------- BUY ----------------

        if signal == "BUY":

            spend = self.usdt * 0.10   # 10% per trade

            if spend > 5:

                qty = spend / price

                if self.s.live_trading:
                    self.ex.market_buy(self.s.symbol, spend)

                self.usdt -= spend
                self.kas += qty

                self.position = Position(
                    qty=qty,
                    entry_price=price,
                    stop=price * 0.992,
                    tp=price * 1.006,
                )

                self._insert_trade(now, "BUY", qty, price)

                log.info(f"BUY {qty:.2f} @ {price}")


        # ---------------- SELL ----------------

        if self.position.qty > 0:

            if (
                price <= self.position.stop
                or price >= self.position.tp
                or signal == "SELL"
            ):

                qty = self.position.qty

                if self.s.live_trading:
                    self.ex.market_sell(self.s.symbol, qty)

                proceeds = qty * price

                self.usdt += proceeds
                self.kas -= qty

                self._insert_trade(now, "SELL", qty, price)

                log.info(f"SELL {qty:.2f} @ {price}")

                self.position = Position()

        # always write equity
        self._insert_equity(now, price)


# =====================================================
# LOOP
# =====================================================

    def run_forever(self):

        log.info("BOT STARTED")

        while True:

            try:
                self.step()

            except Exception as e:
                log.exception(f"BOT ERROR: {e}")

            time.sleep(self.s.poll_seconds)
