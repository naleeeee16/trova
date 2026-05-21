"""Telegram update handlers: /start, /find, /go, message collection, inline buttons."""

from __future__ import annotations

import logging

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from belgrade_recommender.bot.formatter import (
    format_results,
    msg_acknowledged,
    msg_already_active,
    msg_choose_language,
    msg_collecting,
    msg_error,
    msg_lang_set,
    msg_new_search_prompt,
    msg_no_responses,
    msg_no_results,
    msg_no_session_go,
    msg_processing,
    msg_welcome,
    t,
)
from belgrade_recommender.bot.pipeline import recommend_for_group
from belgrade_recommender.bot.session import COLLECTION_WINDOW_SECONDS, SessionManager
from belgrade_recommender.bot.storage import save_request

log = logging.getLogger("bot.handlers")

sessions = SessionManager()

_KEY_RANKED = "ranked_{}"
_KEY_EXPL   = "explanation_{}"

_LANG_LABELS = {
    "srb": "🇷🇸 Srpski",
    "rus": "🇷🇺 Русский",
    "eng": "🇬🇧 English",
}


def _get_lang(context: ContextTypes.DEFAULT_TYPE) -> str:
    """Return the language set for this chat, defaulting to Serbian."""
    return (context.chat_data or {}).get("lang", "srb")


def _lang_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton(_LANG_LABELS["srb"], callback_data="lang:srb"),
        InlineKeyboardButton(_LANG_LABELS["rus"], callback_data="lang:rus"),
        InlineKeyboardButton(_LANG_LABELS["eng"], callback_data="lang:eng"),
    ]])


# ---------------------------------------------------------------------------
# /start
# ---------------------------------------------------------------------------

async def on_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        msg_choose_language(),
        reply_markup=_lang_keyboard(),
        parse_mode=ParseMode.MARKDOWN,
    )


# ---------------------------------------------------------------------------
# /find  /preporuci
# ---------------------------------------------------------------------------

async def on_find(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    lang = _get_lang(context)

    for job in context.job_queue.get_jobs_by_name(f"session_{chat_id}"):
        job.schedule_removal()

    if sessions.is_active(chat_id):
        n = sessions.get(chat_id).n_users
        await update.message.reply_text(
            msg_already_active(n, lang),
            parse_mode=ParseMode.MARKDOWN,
        )

    sessions.open(chat_id, triggered_by=user_id)

    await update.message.reply_text(
        msg_collecting(lang),
        parse_mode=ParseMode.MARKDOWN,
    )

    context.job_queue.run_once(
        callback=_on_timer_fired,
        when=COLLECTION_WINDOW_SECONDS,
        chat_id=chat_id,
        name=f"session_{chat_id}",
        data={"lang": lang},
    )


# ---------------------------------------------------------------------------
# /go  — skip waiting, run immediately with whoever responded so far
# ---------------------------------------------------------------------------

async def on_go(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    lang = _get_lang(context)

    if not sessions.is_active(chat_id):
        await update.message.reply_text(
            msg_no_session_go(lang),
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    for job in context.job_queue.get_jobs_by_name(f"session_{chat_id}"):
        job.schedule_removal()

    await _run_pipeline_for_chat(context.bot, context.application, chat_id, lang)


# ---------------------------------------------------------------------------
# Text messages (during active session)
# ---------------------------------------------------------------------------

async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    text = (update.message.text or "").strip()
    lang = _get_lang(context)

    if not text or not sessions.is_active(chat_id):
        return

    sessions.add_message(chat_id, user_id, text)

    first_name = update.effective_user.first_name or "User"
    n_users = sessions.get(chat_id).n_users

    await update.message.reply_text(
        msg_acknowledged(first_name, n_users, lang),
        parse_mode=ParseMode.MARKDOWN,
    )


# ---------------------------------------------------------------------------
# Timer callback
# ---------------------------------------------------------------------------

async def _on_timer_fired(context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = context.job.chat_id
    job_data = context.job.data or {}
    lang = job_data.get("lang") or (context.chat_data or {}).get("lang", "srb")
    await _run_pipeline_for_chat(context.bot, context.application, chat_id, lang)


# ---------------------------------------------------------------------------
# Shared pipeline runner
# ---------------------------------------------------------------------------

async def _run_pipeline_for_chat(bot, application, chat_id: int, lang: str = "srb") -> None:
    session = sessions.close(chat_id)

    if not session or not session.inputs:
        await bot.send_message(
            chat_id=chat_id,
            text=msg_no_responses(lang),
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    user_texts = session.combined_inputs()

    await bot.send_message(
        chat_id=chat_id,
        text=msg_processing(len(user_texts), lang),
        parse_mode=ParseMode.MARKDOWN,
    )

    try:
        ranked, explanation, mismatches, reasons, prefs, attrs_map, liked_vecs_list, event_vecs = await recommend_for_group(user_texts, lang=lang)
    except Exception:
        log.exception("Pipeline failed for chat %d", chat_id)
        await bot.send_message(chat_id=chat_id, text=msg_error(lang))
        return

    if not ranked:
        await bot.send_message(
            chat_id=chat_id,
            text=msg_no_results(lang),
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    save_request(
        chat_id=chat_id,
        lang=lang,
        user_inputs=user_texts,
        ranked=ranked,
        conflicts=explanation,
        reasons=reasons,
        prefs=prefs,
        attrs_map=attrs_map,
        liked_vecs_list=liked_vecs_list,
        event_vecs=event_vecs,
    )

    bd = application.bot_data
    bd[_KEY_RANKED.format(chat_id)]  = ranked
    bd[_KEY_EXPL.format(chat_id)]    = explanation
    bd[f"mismatches_{chat_id}"]      = mismatches
    bd[f"reasons_{chat_id}"]         = reasons
    bd[f"lang_{chat_id}"]            = lang

    text = format_results(ranked, mismatches, reasons, show_all=False, lang=lang)
    keyboard = _build_keyboard(chat_id, len(ranked), lang)

    await bot.send_message(
        chat_id=chat_id,
        text=text,
        parse_mode=ParseMode.HTML,
        disable_web_page_preview=True,
        reply_markup=InlineKeyboardMarkup(keyboard) if keyboard else None,
    )


# ---------------------------------------------------------------------------
# Inline keyboard callbacks
# ---------------------------------------------------------------------------

async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    data = query.data or ""

    # --- Language selection ---
    if data.startswith("lang:"):
        lang = data.split(":")[1]
        if lang not in _LANG_LABELS:
            return
        context.chat_data["lang"] = lang
        await query.edit_message_text(
            msg_lang_set(lang) + "\n\n" + msg_welcome(lang),
            parse_mode=ParseMode.MARKDOWN,
        )

    # --- Show all results ---
    elif data.startswith("show_all:"):
        chat_id = int(data.split(":")[1])
        bd = context.application.bot_data
        ranked     = bd.get(_KEY_RANKED.format(chat_id), [])
        mismatches = bd.get(f"mismatches_{chat_id}", {})
        reasons    = bd.get(f"reasons_{chat_id}", [])
        lang       = bd.get(f"lang_{chat_id}", _get_lang(context))

        text = format_results(ranked, mismatches, reasons, show_all=True, lang=lang)
        await query.edit_message_text(
            text=text,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton(
                    t("new_search_btn", lang),
                    callback_data=f"new_search:{chat_id}",
                ),
            ]]),
        )

    # --- New search prompt ---
    elif data.startswith("new_search:"):
        chat_id = int(data.split(":")[1])
        lang = context.application.bot_data.get(f"lang_{chat_id}", _get_lang(context))
        await query.edit_message_text(
            msg_new_search_prompt(lang),
            parse_mode=ParseMode.HTML,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_keyboard(chat_id: int, n_results: int, lang: str = "srb") -> list[list[InlineKeyboardButton]]:
    rows = []
    if n_results > 5:
        rows.append([InlineKeyboardButton(t("show_all_btn", lang), callback_data=f"show_all:{chat_id}")])
    rows.append([InlineKeyboardButton(t("new_search_btn", lang), callback_data=f"new_search:{chat_id}")])
    return rows
