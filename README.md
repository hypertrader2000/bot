# MEXC KASPA Momentum Bot (Local Dashboard)

⚠️ **Risk warning**: This project can place **real** orders on your MEXC account. Crypto is volatile.
Use small size first. You are responsible for all losses.

## What it does
- Trades **Spot** on MEXC (default: `KASUSDT`) using a momentum + trend filter strategy
- Places market buys/sells, tracks orders/fills, computes realized/unrealized PnL
- Runs a local web dashboard showing:
  - balances, open position
  - recent trades & orders
  - equity curve & PnL

## Strategy (high level)
- Uses **1m candles** for signals.
- Entry engines (toggle in `.env`):
  - **DIP entry**: buy when price is below 1m SMA by `DIP_ENTRY_PCT`
  - **MICRO momentum**: buy after `MOM_UP_BARS` consecutive up closes
  - **Fallback timer**: optional “keep-alive” entry (recommended **OFF**)
  - **Trend filter**: optional; only take longs if price is above a slower SMA
- Exit profile:
  - Partial take-profit at TP1, full exit at TP2
  - Stop + trailing stop, plus a max-hold time exit
- Risk & guard rails:
  - Randomized position sizing within `MIN_TRADE_FRAC..MAX_TRADE_FRAC`, capped by `MAX_POSITION_FRACTION`
  - Trades/hour cap + pause, loss-streak pause
  - Optional hard goal/stop equity
  - Optional cost/edge gate (fees+slippage) to avoid low-edge scalps

## Quick start (Windows / PowerShell)
1) Install Python 3.10+  
2) Unzip this folder, open a terminal inside it
3) Create venv + install:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
4) Copy env file and add your keys:
```powershell
copy .env.example .env
notepad .env
```
5) Run:
```powershell
python run.py
```
Open dashboard: http://127.0.0.1:8000

## Config
Edit `.env`:
- `MEXC_API_KEY` / `MEXC_API_SECRET`
- `SYMBOL` (default `KASUSDT`)
- `MIN_TRADE_FRAC` / `MAX_TRADE_FRAC` (randomized entry size range)
- `ENABLE_FALLBACK_ENTRY` (recommended `false`)
- `SCALP_STOP`, `SCALP_TP1`, `SCALP_TP2`, `SCALP_TRAIL`
- `LIVE_TRADING`:
  - `false` = dry-run (signals logged, no orders)
  - `true`  = sends live orders

## Optional: DRL strategy integration (PyTorch)
You can plug a trained DRL policy into the strategy layer without changing execution/risk code.

Enable via `.env`:
- `STRATEGY_MODE=heuristic` (default)
  - `drl` = DRL decides entries
  - `hybrid` = DRL must agree with heuristic to enter (safer)
- `DRL_MODEL_PATH=...` (TorchScript `.ts` recommended)
- `DRL_MIN_CONF=0.55` (confidence threshold)
- `DRL_OBS_WINDOW=64`

The bot builds an observation vector in `app/drl.py::build_obs()` and feeds it to the model.
Train your model to expect that exact observation layout.

## Notes
- This bot uses MEXC Spot V3 REST endpoints:
  - base: `https://api.mexc.com`  (MEXC docs)
- It polls (no websockets) to keep things reliable on a home PC.

## Troubleshooting
- **401 / signature error**: check your system clock, API key permissions, and IP restrictions.
- **insufficient balance**: reduce `MIN_TRADE_FRAC/MAX_TRADE_FRAC` or deposit USDT/KAS.
- **symbol not found**: confirm the exact MEXC spot symbol in your account (e.g. `KASUSDT`).

