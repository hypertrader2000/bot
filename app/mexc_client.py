from __future__ import annotations

import time
import hmac
import hashlib
from urllib.parse import urlencode
from typing import Any, Dict, Optional

import requests

BASE_URL = "https://api.mexc.com"


class MexcAPIError(RuntimeError):
    def __init__(self, status: int, message: str, payload: Any | None = None):
        super().__init__(f"MEXC API error {status}: {message}")
        self.status = status
        self.payload = payload


class MexcClient:
    def __init__(self, api_key: str, api_secret: str, timeout: int = 15):
        self.api_key = api_key
        self.api_secret = api_secret.encode("utf-8")
        self.timeout = timeout
        self._time_offset_ms = 0

        self.session = requests.Session()
        # IMPORTANT:
        # - Do NOT set X-MEXC-APIKEY globally (only send on signed endpoints)
        # - Avoid forcing Content-Type on public endpoints
        self.session.headers.update({
            "User-Agent": "mexc-kaspa-bot/1.0",
        })

    # ---- time sync ----
    def sync_time(self) -> None:
        r = self.session.get(f"{BASE_URL}/api/v3/time", timeout=self.timeout)
        r.raise_for_status()
        server_ms = int(r.json()["serverTime"])
        local_ms = int(time.time() * 1000)
        self._time_offset_ms = server_ms - local_ms

    def _ts(self) -> int:
        return int(time.time() * 1000) + int(self._time_offset_ms)

    # ---- signing (Spot V3) ----
    def _sign(self, params: Dict[str, Any]) -> str:
        # totalParams is the querystring sorted by key. Exclude None values.
        items = sorted((k, v) for k, v in params.items() if v is not None)
        qs = urlencode(items, doseq=True)
        return hmac.new(self.api_secret, qs.encode("utf-8"), hashlib.sha256).hexdigest()

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        signed: bool = False,
    ) -> Any:
        url = f"{BASE_URL}{path}"
        params = dict(params or {})
        method_u = method.upper()

        # SIGNED (private) endpoints
        if signed:
            params["timestamp"] = self._ts()
            params["signature"] = self._sign(params)

            headers: Dict[str, str] = {
                "X-MEXC-APIKEY": self.api_key,
            }

            # IMPORTANT FIX:
            # - For signed GET/DELETE: do NOT force Content-Type
            # - For signed POST: use x-www-form-urlencoded
            if method_u in ("GET", "DELETE"):
                r = self.session.request(
                    method_u,
                    url,
                    params=params,
                    headers=headers,
                    timeout=self.timeout,
                )
            else:
                headers["Content-Type"] = "application/x-www-form-urlencoded"
                r = self.session.request(
                    method_u,
                    url,
                    data=params,
                    headers=headers,
                    timeout=self.timeout,
                )

        # PUBLIC endpoints
        else:
            # Public endpoints should NOT include signature and should NOT send X-MEXC-APIKEY
            if method_u == "GET":
                r = self.session.request(
                    method_u,
                    url,
                    params=params,
                    timeout=self.timeout,
                )
            else:
                r = self.session.request(
                    method_u,
                    url,
                    json=params,
                    timeout=self.timeout,
                )

        if r.status_code >= 400:
            try:
                payload = r.json()
            except Exception:
                payload = r.text
            raise MexcAPIError(r.status_code, str(payload), payload)

        return r.json() if r.text else {}

    # ---- public market data ----
    def ticker_price(self, symbol: str) -> float:
        data = self._request("GET", "/api/v3/ticker/price", {"symbol": symbol}, signed=False)
        return float(data["price"])

    def klines(self, symbol: str, interval: str = "1m", limit: int = 200):
        # returns list of [openTime, open, high, low, close, volume, closeTime, quoteAssetVolume]
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        return self._request("GET", "/api/v3/klines", params, signed=False)

    # ---- account ----
    def account(self) -> Any:
        return self._request("GET", "/api/v3/account", {}, signed=True)

    def balances(self) -> Dict[str, Dict[str, float]]:
        acct = self.account()
        out: Dict[str, Dict[str, float]] = {}
        for b in acct.get("balances", []):
            asset = b.get("asset")
            if not asset:
                continue
            out[asset] = {
                "free": float(b.get("free", 0) or 0),
                "locked": float(b.get("locked", 0) or 0),
            }
        return out

    # ---- trading ----
    def new_order(
        self,
        *,
        symbol: str,
        side: str,
        type_: str,
        quantity: float | None = None,
        quote_order_qty: float | None = None,
        price: float | None = None,
    ) -> Any:
        params = {
            "symbol": symbol,
            "side": side,
            "type": type_,
            "quantity": None if quantity is None else self._fmt(quantity),
            "quoteOrderQty": None if quote_order_qty is None else self._fmt(quote_order_qty),
            "price": None if price is None else self._fmt(price),
        }
        return self._request("POST", "/api/v3/order", params, signed=True)

    def order(self, *, symbol: str, order_id: str) -> Any:
        return self._request("GET", "/api/v3/order", {"symbol": symbol, "orderId": order_id}, signed=True)

    def my_trades(self, *, symbol: str, limit: int = 100) -> Any:
        return self._request("GET", "/api/v3/myTrades", {"symbol": symbol, "limit": limit}, signed=True)

    @staticmethod
    def _fmt(x: float) -> str:
        # avoid scientific notation
        return f"{x:.10f}".rstrip("0").rstrip(".")
