from __future__ import annotations

import csv
import io
import json
import math
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, List, Optional, Tuple

import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request


# ------------------------------------------------------------
# App + paths
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DB_PATH = os.path.join(PROJECT_DIR, "data", "bot.sqlite3")

app = FastAPI(title="MEXC KASPA Bot")

static_dir = os.path.join(BASE_DIR, "static")
templates_dir = os.path.join(BASE_DIR, "templates")

if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

templates = Jinja2Templates(directory=templates_dir)


# ------------------------------------------------------------
# DB helpers
# ------------------------------------------------------------
def _db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def _iso_utc(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------
# JSON safety: kill inf/nan + numpy types
# ------------------------------------------------------------
def _clean_jsonable(x: Any) -> Any:
    try:
        if hasattr(x, "item"):
            x = x.item()
    except Exception:
        pass

    if isinstance(x, float):
        return x if math.isfinite(x) else 0.0
    if isinstance(x, (np.floating,)):
        xf = float(x)
        return xf if math.isfinite(xf) else 0.0
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, dict):
        return {k: _clean_jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clean_jsonable(v) for v in x]
    return x


# ------------------------------------------------------------
# STATE TABLE SCHEMA AUTODETECT
# ------------------------------------------------------------
_STATE_COLS_CACHE: Optional[Tuple[str, str]] = None


def _detect_state_columns(con: sqlite3.Connection) -> Tuple[str, str]:
    global _STATE_COLS_CACHE
    if _STATE_COLS_CACHE is not None:
        return _STATE_COLS_CACHE

    info = con.execute("PRAGMA table_info(state)").fetchall()
    cols = [r["name"] for r in info] if info else []

    key_cands = ["key", "k", "name", "setting", "param", "id"]
    val_cands = ["value", "val", "v", "data", "text", "json"]

    key_col = next((c for c in key_cands if c in cols), None)
    val_col = next((c for c in val_cands if c in cols), None)

    if key_col is None and cols:
        key_col = cols[0]
    if val_col is None and len(cols) >= 2:
        val_col = cols[1] if cols[1] != key_col else (cols[2] if len(cols) >= 3 else cols[1])

    if key_col is None or val_col is None:
        key_col, val_col = "key", "value"

    _STATE_COLS_CACHE = (key_col, val_col)
    return key_col, val_col


def _get_state(con: sqlite3.Connection, key: str) -> Optional[str]:
    key_col, val_col = _detect_state_columns(con)
    try:
        row = con.execute(
            f"SELECT {val_col} AS v FROM state WHERE {key_col}=?",
            (key,),
        ).fetchone()
        return row["v"] if row else None
    except Exception:
        return None


def _get_halt(con: sqlite3.Connection) -> tuple[bool, str]:
    halted_raw = _get_state(con, "halted")
    reason = _get_state(con, "halt_reason") or ""
    halted = str(halted_raw or "0").strip().lower() in ("1", "true", "yes", "y", "on")
    return halted, reason


# ------------------------------------------------------------
# Page
# ------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    settings = {"symbol": "KASUSDT", "live_trading": False}
    return templates.TemplateResponse("index.html", {"request": request, "settings": settings})


# ------------------------------------------------------------
# Debug helpers
# ------------------------------------------------------------
@app.get("/api/health")
def api_health():
    return {"status": "ok", "db_path": DB_PATH}


@app.get("/api/schema")
def api_schema():
    con = _db()
    try:
        tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()]
        state_cols = []
        if "state" in tables:
            state_cols = [r["name"] for r in con.execute("PRAGMA table_info(state)").fetchall()]
        return JSONResponse(_clean_jsonable({"tables": tables, "state_columns": state_cols}))
    finally:
        con.close()


# ------------------------------------------------------------
# API: runtime controls (mode + overrides)
# ------------------------------------------------------------
def _set_state(con: sqlite3.Connection, key: str, value: str) -> None:
    key_col, val_col = _detect_state_columns(con)
    try:
        con.execute(
            f"INSERT INTO state({key_col},{val_col}) VALUES(?,?) "
            f"ON CONFLICT({key_col}) DO UPDATE SET {val_col}=excluded.{val_col}",
            (key, value),
        )
        con.commit()
    except Exception:
        # best effort; do not crash API
        pass


@app.get("/api/config")
def api_config():
    con = _db()
    try:
        mode = _get_state(con, "mode") or "balanced"
        overrides_raw = _get_state(con, "runtime_overrides_json") or "{}"
        try:
            overrides = json.loads(overrides_raw)
            if not isinstance(overrides, dict):
                overrides = {}
        except Exception:
            overrides = {}
        return JSONResponse(_clean_jsonable({
            "mode": mode,
            "overrides": overrides,
            "presets": [
                "conservative",
                "balanced",
                "aggressive",
                "scalp_downtrend",
                "momentum",
                "maker_like",
            ],
        }))
    finally:
        con.close()


@app.post("/api/config/mode")
async def api_set_mode(request: Request):
    body = await request.json()
    mode = str(body.get("mode", "")).strip()
    if not mode:
        return JSONResponse({"ok": False, "error": "mode required"}, status_code=400)

    con = _db()
    try:
        _set_state(con, "mode", mode)
        return JSONResponse({"ok": True, "mode": mode})
    finally:
        con.close()


@app.post("/api/config/overrides")
async def api_set_overrides(request: Request):
    """
    Set/merge runtime overrides. Send JSON like:
      {"max_trade_frac": 0.12, "trail_pct": 0.0012}
    To clear overrides, send {"__clear__": true}
    """
    body = await request.json()
    con = _db()
    try:
        if body.get("__clear__"):
            _set_state(con, "runtime_overrides_json", "{}")
            return JSONResponse({"ok": True, "overrides": {}})

        raw = _get_state(con, "runtime_overrides_json") or "{}"
        try:
            cur = json.loads(raw)
            if not isinstance(cur, dict):
                cur = {}
        except Exception:
            cur = {}

        for k, v in body.items():
            if k.startswith("__"):
                continue
            if isinstance(v, (int, float, str, bool)) or v is None:
                cur[k] = v

        _set_state(con, "runtime_overrides_json", json.dumps(cur))
        return JSONResponse({"ok": True, "overrides": cur})
    finally:
        con.close()


# ------------------------------------------------------------
# API: equity series for charts
# ------------------------------------------------------------
@app.get("/api/equity")
def api_equity():
    con = _db()
    try:
        rows = con.execute(
            "SELECT ts, equity_usdt, realized_pnl_usdt, unrealized_pnl_usdt, price "
            "FROM equity ORDER BY ts ASC LIMIT 5000"
        ).fetchall()

        out = []
        for r in rows:
            ts = int(r["ts"])
            out.append(
                {
                    "ts": ts,
                    "t": _iso_utc(ts),
                    "equity": float(r["equity_usdt"] or 0.0),
                    "realized": float(r["realized_pnl_usdt"] or 0.0),
                    "unrealized": float(r["unrealized_pnl_usdt"] or 0.0),
                    "price": float(r["price"] or 0.0),
                }
            )
        return JSONResponse(_clean_jsonable(out))
    finally:
        con.close()


# ------------------------------------------------------------
# API: recent trades/orders + position + balances (from state)
# ------------------------------------------------------------
@app.get("/api/recent")
def api_recent():
    con = _db()
    try:
        trades = con.execute(
            "SELECT ts, side, qty, price FROM trades ORDER BY ts DESC LIMIT 50"
        ).fetchall()

        orders = con.execute(
            "SELECT ts, side, type, qty, quote_qty, mexc_order_id FROM orders ORDER BY ts DESC LIMIT 50"
        ).fetchall()

        # position_json is optional; if bot doesn't write it, show zeros
        pos_raw = _get_state(con, "position_json") or "{}"
        try:
            pos = json.loads(pos_raw)
        except Exception:
            pos = {}

        position = {
            "qty": float(pos.get("qty", 0.0) or 0.0),
            "cost_usdt": float(pos.get("cost_usdt", 0.0) or 0.0),
            "stop": float(pos.get("stop", 0.0) or 0.0),
            "trail": float(pos.get("trail", 0.0) or 0.0),
            "tp1": float(pos.get("tp1", 0.0) or 0.0),
            "tp2": float(pos.get("tp2", 0.0) or 0.0),
        }

        # balances are kept in state by the bot (paper or live)
        sim_usdt = float((_get_state(con, "sim_quote") or "0") or 0.0)
        sim_kas = float((_get_state(con, "sim_base") or "0") or 0.0)

        last_price_row = con.execute("SELECT price FROM equity ORDER BY ts DESC LIMIT 1").fetchone()
        last_price = float(last_price_row["price"] or 0.0) if last_price_row else 0.0
        total_equity_calc = sim_usdt + sim_kas * last_price

        out = {
            "trades": [
                {
                    "ts": int(r["ts"]),
                    "t": _iso_utc(int(r["ts"])),
                    "side": r["side"],
                    "qty": float(r["qty"] or 0.0),
                    "price": float(r["price"] or 0.0),
                }
                for r in trades
            ],
            "orders": [
                {
                    "ts": int(r["ts"]),
                    "t": _iso_utc(int(r["ts"])),
                    "side": r["side"],
                    "type": r["type"],
                    "qty": (None if r["qty"] is None else float(r["qty"])),
                    "quote_qty": (None if r["quote_qty"] is None else float(r["quote_qty"])),
                    "mexc_order_id": r["mexc_order_id"],
                }
                for r in orders
            ],
            "position": position,
            "sim_usdt": sim_usdt,
            "sim_kas": sim_kas,
            "sim_quote": sim_usdt,
            "sim_base": sim_kas,
            "last_price": last_price,
            "total_equity_calc": total_equity_calc,
            "equity_usdt": total_equity_calc,
        }

        return JSONResponse(_clean_jsonable(out))
    finally:
        con.close()


# ------------------------------------------------------------
# API: stats for tiles
# ------------------------------------------------------------
@app.get("/api/stats")
def api_stats():
    con = _db()
    try:
        sim_usdt = float((_get_state(con, "sim_quote") or "0") or 0.0)
        sim_kas = float((_get_state(con, "sim_base") or "0") or 0.0)

        halted, halt_reason = _get_halt(con)

        eq_rows = con.execute(
            "SELECT ts, equity_usdt, realized_pnl_usdt, unrealized_pnl_usdt, price "
            "FROM equity ORDER BY ts ASC"
        ).fetchall()

        last_price = 0.0
        if eq_rows:
            last_price = float(eq_rows[-1]["price"] or 0.0)
        total_equity_calc = sim_usdt + sim_kas * last_price

        if not eq_rows:
            stats = {
                "start_equity": sim_usdt,
                "end_equity": sim_usdt,
                "pnl_usdt": 0.0,
                "pnl_pct": 0.0,
                "max_drawdown_usdt": 0.0,
                "trades": 0,
                "round_trips": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "avg_hold_sec": 0.0,
                "realized_from_trades_usdt": 0.0,
                "sim_usdt": sim_usdt,
                "sim_kas": sim_kas,
                "last_price": last_price,
                "total_equity_calc": total_equity_calc,
                "equity_usdt": total_equity_calc,
                "realized_pnl_usdt": 0.0,
                "unrealized_pnl_usdt": 0.0,
                "halted": halted,
                "halt_reason": halt_reason,
            }
            return JSONResponse(_clean_jsonable(stats))

        eq = [float(r["equity_usdt"] or 0.0) for r in eq_rows]
        start_equity = float(eq[0])
        end_equity = float(eq[-1])

        pnl = end_equity - start_equity
        pnl_pct = (pnl / start_equity) if start_equity > 0 else 0.0

        peak = eq[0]
        max_dd = 0.0
        for v in eq:
            peak = max(peak, v)
            max_dd = max(max_dd, peak - v)

        trade_rows = con.execute(
            "SELECT side, qty, price, ts FROM trades ORDER BY ts ASC"
        ).fetchall()

        wins = 0
        losses = 0
        gross_profit = 0.0
        gross_loss = 0.0
        hold_times_ms: List[int] = []

        entry_price = None
        entry_ts = None

        for r in trade_rows:
            side = str(r["side"] or "").strip().upper()
            qty = float(r["qty"] or 0.0)
            price = float(r["price"] or 0.0)
            ts = int(r["ts"] or 0)

            if side.startswith("B"):
                side = "BUY"
            elif side.startswith("S"):
                side = "SELL"

            if side == "BUY":
                entry_price = price
                entry_ts = ts
            elif side == "SELL":
                if entry_price is None:
                    continue
                trade_pnl = (price - entry_price) * qty
                if trade_pnl > 0:
                    wins += 1
                    gross_profit += trade_pnl
                else:
                    losses += 1
                    gross_loss += abs(trade_pnl)
                if entry_ts is not None and ts >= entry_ts:
                    hold_times_ms.append(ts - entry_ts)
                entry_price = None
                entry_ts = None

        round_trips = wins + losses
        win_rate = (wins / round_trips * 100.0) if round_trips > 0 else 0.0
        avg_hold_sec = (sum(hold_times_ms) / len(hold_times_ms) / 1000.0) if hold_times_ms else 0.0

        profit_factor = 999.0 if (gross_loss <= 0.0 and gross_profit > 0.0) else (gross_profit / gross_loss if gross_loss > 0 else 0.0)

        last_realized = float(eq_rows[-1]["realized_pnl_usdt"] or 0.0)
        last_unreal = float(eq_rows[-1]["unrealized_pnl_usdt"] or 0.0)

        stats = {
            "start_equity": start_equity,
            "end_equity": end_equity,
            "pnl_usdt": pnl,
            "pnl_pct": pnl_pct,
            "max_drawdown_usdt": max_dd,
            "trades": len(trade_rows),
            "round_trips": round_trips,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_hold_sec": avg_hold_sec,
            "realized_from_trades_usdt": (gross_profit - gross_loss),
            "sim_usdt": sim_usdt,
            "sim_kas": sim_kas,
            "last_price": last_price,
            "total_equity_calc": total_equity_calc,
            "equity_usdt": total_equity_calc,
            "realized_pnl_usdt": last_realized,
            "unrealized_pnl_usdt": last_unreal,
            "halted": halted,
            "halt_reason": halt_reason,
        }

        return JSONResponse(_clean_jsonable(stats))
    finally:
        con.close()


# ------------------------------------------------------------
# CSV exports
# ------------------------------------------------------------
@app.get("/api/export/equity.csv")
def export_equity_csv():
    con = _db()
    try:
        rows = con.execute(
            "SELECT ts, equity_usdt, realized_pnl_usdt, unrealized_pnl_usdt, price "
            "FROM equity ORDER BY ts ASC"
        ).fetchall()

        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["ts_ms", "utc", "equity_usdt", "realized_pnl_usdt", "unrealized_pnl_usdt", "price"])
        for r in rows:
            ts = int(r["ts"])
            w.writerow(
                [
                    ts,
                    _iso_utc(ts),
                    float(r["equity_usdt"] or 0.0),
                    float(r["realized_pnl_usdt"] or 0.0),
                    float(r["unrealized_pnl_usdt"] or 0.0),
                    float(r["price"] or 0.0),
                ]
            )
        return Response(content=buf.getvalue(), media_type="text/csv")
    finally:
        con.close()


@app.get("/api/export/trades.csv")
def export_trades_csv():
    con = _db()
    try:
        rows = con.execute(
            "SELECT ts, side, qty, price, mexc_trade_id, mexc_order_id "
            "FROM trades ORDER BY ts ASC"
        ).fetchall()

        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["ts_ms", "utc", "side", "qty", "price", "mexc_trade_id", "mexc_order_id"])
        for r in rows:
            ts = int(r["ts"])
            w.writerow(
                [
                    ts,
                    _iso_utc(ts),
                    r["side"],
                    float(r["qty"] or 0.0),
                    float(r["price"] or 0.0),
                    r["mexc_trade_id"] or "",
                    r["mexc_order_id"] or "",
                ]
            )
        return Response(content=buf.getvalue(), media_type="text/csv")
    finally:
        con.close()
