from __future__ import annotations
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

DB_PATH = Path("data/bot.sqlite3")

SCHEMA = r'''
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS orders (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts INTEGER NOT NULL,
  symbol TEXT NOT NULL,
  side TEXT NOT NULL,
  type TEXT NOT NULL,
  qty REAL,
  quote_qty REAL,
  price REAL,
  mexc_order_id TEXT,
  status TEXT,
  raw_json TEXT
);

CREATE TABLE IF NOT EXISTS trades (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts INTEGER NOT NULL,
  symbol TEXT NOT NULL,
  side TEXT NOT NULL,
  qty REAL NOT NULL,
  price REAL NOT NULL,
  fee REAL DEFAULT 0,
  fee_asset TEXT,
  mexc_trade_id TEXT,
  mexc_order_id TEXT,
  raw_json TEXT
);

CREATE TABLE IF NOT EXISTS state (
  k TEXT PRIMARY KEY,
  v TEXT
);

CREATE TABLE IF NOT EXISTS equity (
  ts INTEGER PRIMARY KEY,
  equity_usdt REAL NOT NULL,
  realized_pnl_usdt REAL NOT NULL,
  unrealized_pnl_usdt REAL NOT NULL,
  price REAL NOT NULL
);
'''

def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH.as_posix(), check_same_thread=False)
    con.execute("PRAGMA foreign_keys=ON;")
    con.executescript(SCHEMA)
    return con

def set_state(con: sqlite3.Connection, key: str, value: str) -> None:
    con.execute(
        "INSERT INTO state(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v",
        (key, value),
    )
    con.commit()

def get_state(con: sqlite3.Connection, key: str) -> Optional[str]:
    cur = con.execute("SELECT v FROM state WHERE k=?", (key,))
    row = cur.fetchone()
    return None if row is None else row[0]

def insert_order(
    con: sqlite3.Connection,
    *,
    ts: int,
    symbol: str,
    side: str,
    type_: str,
    qty: float | None,
    quote_qty: float | None,
    price: float | None,
    mexc_order_id: str | None,
    status: str | None,
    raw_json: str,
) -> int:
    cur = con.execute(
        "INSERT INTO orders(ts,symbol,side,type,qty,quote_qty,price,mexc_order_id,status,raw_json) "
        "VALUES(?,?,?,?,?,?,?,?,?,?)",
        (ts, symbol, side, type_, qty, quote_qty, price, mexc_order_id, status, raw_json),
    )
    con.commit()
    return int(cur.lastrowid)

def insert_trade(
    con: sqlite3.Connection,
    *,
    ts: int,
    symbol: str,
    side: str,
    qty: float,
    price: float,
    fee: float,
    fee_asset: str | None,
    mexc_trade_id: str | None,
    mexc_order_id: str | None,
    raw_json: str,
) -> int:
    cur = con.execute(
        "INSERT INTO trades(ts,symbol,side,qty,price,fee,fee_asset,mexc_trade_id,mexc_order_id,raw_json) "
        "VALUES(?,?,?,?,?,?,?,?,?,?)",
        (ts, symbol, side, qty, price, fee, fee_asset, mexc_trade_id, mexc_order_id, raw_json),
    )
    con.commit()
    return int(cur.lastrowid)

def upsert_equity(
    con: sqlite3.Connection,
    *,
    ts: int,
    equity_usdt: float,
    realized_pnl_usdt: float,
    unrealized_pnl_usdt: float,
    price: float,
) -> None:
    con.execute(
        "INSERT INTO equity(ts,equity_usdt,realized_pnl_usdt,unrealized_pnl_usdt,price) VALUES(?,?,?,?,?) "
        "ON CONFLICT(ts) DO UPDATE SET equity_usdt=excluded.equity_usdt, "
        "realized_pnl_usdt=excluded.realized_pnl_usdt, "
        "unrealized_pnl_usdt=excluded.unrealized_pnl_usdt, "
        "price=excluded.price",
        (ts, equity_usdt, realized_pnl_usdt, unrealized_pnl_usdt, price),
    )
    con.commit()

def fetch_recent(con: sqlite3.Connection, table: str, limit: int = 50):
    cur = con.execute(f"SELECT * FROM {table} ORDER BY ts DESC LIMIT ?", (limit,))
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]

def fetch_equity(con: sqlite3.Connection, limit: int = 2000):
    cur = con.execute("SELECT * FROM equity ORDER BY ts ASC LIMIT ?", (limit,))
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]
