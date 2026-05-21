#!/usr/bin/env python3
"""Telegram bot entry point.

Usage (from project root):
    python src/belgrade_recommender/bot/main.py
    python run_bot.py
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap — works whether run as a module or a script
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[3]
_SRC  = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ---------------------------------------------------------------------------
# Load .env
# ---------------------------------------------------------------------------
_env_file = _ROOT / ".env"
if _env_file.is_file():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bot.main")

# ---------------------------------------------------------------------------
# Bot
# ---------------------------------------------------------------------------
from telegram import BotCommand
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    MessageHandler,
    filters,
)

from belgrade_recommender.bot.handlers import on_button, on_find, on_go, on_message, on_start
from belgrade_recommender.bot.pipeline import load_corpus_sync

_BOT_COMMANDS = [
    BotCommand("start",      "Choose language / Odaberi jezik / Выбрать язык"),
    BotCommand("find",       "Start a new search session 🎯"),
    BotCommand("preporuci",  "Pokreni pretragu (srpski alias za /find)"),
    BotCommand("go",         "Skip timer — get results now ⚡"),
]


async def _post_init(application) -> None:
    """Register bot commands so they appear in the Telegram '/' menu."""
    await application.bot.set_my_commands(_BOT_COMMANDS)


def main() -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        log.error("TELEGRAM_BOT_TOKEN is not set. Add it to your .env file.")
        sys.exit(1)

    log.info("Loading event corpus — this may take a few minutes on first run…")
    load_corpus_sync()
    log.info("Corpus ready. Starting bot…")

    app = ApplicationBuilder().token(token).post_init(_post_init).build()

    app.add_handler(CommandHandler("start",      on_start))
    app.add_handler(CommandHandler("find",       on_find))
    app.add_handler(CommandHandler("preporuci",  on_find))
    app.add_handler(CommandHandler("go",         on_go))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))
    app.add_handler(CallbackQueryHandler(on_button))

    log.info("Bot is running. Press Ctrl+C to stop.")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
