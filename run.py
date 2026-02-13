from __future__ import annotations

import logging
import threading
import uvicorn

from app.config import get_settings
from app.exchange import Exchange
import app.db as dbm
from app.bot import KaspaBot
from app import db



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)


def _bot_worker(bot: KaspaBot) -> None:
    """
    Runs the trading loop in a background thread.
    Keeps FastAPI responsive.
    """
    bot.run_forever()


def main() -> None:

    # ✅ Load settings
    s = get_settings()

    # ✅ Build exchange
    exchange = Exchange.from_settings(s)

    # ✅ Connect DB
    con = dbm.connect()

    # ✅ Create bot
    bot = KaspaBot(s, exchange, con, db)

    # ✅ Start bot thread
    t = threading.Thread(
        target=_bot_worker,
        args=(bot,),
        daemon=True
    )
    t.start()

    logging.info("Trading bot thread started.")

    # ✅ Start dashboard
    uvicorn.run(
        "webapp.main:app",
        host=s.host,
        port=s.port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
