# app/exchange.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from app.mexc_client import MexcClient  # <-- keep this if your file is app/mexc_client.py


@dataclass
class Exchange:
    client: MexcClient

    @classmethod
    def from_settings(cls, s) -> "Exchange":
        # Your Settings uses api_key / api_secret (from config.py)
        api_key = str(getattr(s, "api_key", "")).strip()
        api_secret = str(getattr(s, "api_secret", "")).strip()

        timeout = int(getattr(s, "mexc_timeout_sec", 15))

        if not api_key or not api_secret:
            raise RuntimeError("Missing MEXC credentials: set MEXC_API_KEY and MEXC_API_SECRET in .env")

        c = MexcClient(api_key=api_key, api_secret=api_secret, timeout=timeout)
        c.sync_time()
        return cls(client=c)

    # ---------- Market data ----------
    def get_klines_1m(self, symbol: str, limit: int = 160) -> List[Dict[str, Any]]:
        raw = self.client.klines(symbol=symbol, interval="1m", limit=limit)
        out: List[Dict[str, Any]] = []
        for row in raw:
            out.append(
                {
                    "ts": int(row[0]),
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5]),
                }
            )
        return out

    def get_price(self, symbol: str) -> float:
        return float(self.client.ticker_price(symbol))

    # ---------- Trading ----------
    def market_buy(self, symbol: str, spend_usdt: float) -> Any:
        return self.client.new_order(
            symbol=symbol,
            side="BUY",
            type_="MARKET",
            quote_order_qty=float(spend_usdt),
        )

    def market_sell(self, symbol: str, qty: float) -> Any:
        return self.client.new_order(
            symbol=symbol,
            side="SELL",
            type_="MARKET",
            quantity=float(qty),
        )

    # ---------- Account ----------
    def balances(self) -> Dict[str, Dict[str, float]]:
        return self.client.balances()
