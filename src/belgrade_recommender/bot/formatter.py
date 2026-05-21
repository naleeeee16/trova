"""Format recommendation results and bot messages for Telegram."""

from __future__ import annotations

from belgrade_recommender.bot.config import NO_CONSENSUS_THRESHOLD

# ---------------------------------------------------------------------------
# Translations  (srb = Serbian, rus = Russian, eng = English)
# ---------------------------------------------------------------------------

_T: dict[str, dict[str, str]] = {
    "welcome": {
        "srb": (
            "👋 Zdravo! Ja sam *Belgrade Event Bot*.\n\n"
            "Koristite */find* u grupnom čatu i svi napišite šta hoćete — "
            "naći ću mesta koja odgovaraju celoj ekipi. 🎯\n\n"
            "Kad su svi napisali, ukucajte */go* da odmah dobijete preporuke."
        ),
        "rus": (
            "👋 Привет! Я *Belgrade Event Bot*.\n\n"
            "Используйте */find* в групповом чате — каждый пишет чего хочет, "
            "а я найду места, которые подойдут всей компании. 🎯\n\n"
            "Когда все написали — введите */go* чтобы сразу получить результаты."
        ),
        "eng": (
            "👋 Hello! I'm *Belgrade Event Bot*.\n\n"
            "Use */find* in a group chat — everyone writes what they want "
            "and I'll find places that work for the whole crew. 🎯\n\n"
            "Once everyone has replied, type */go* to get recommendations right away."
        ),
    },
    "choose_lang": {
        "srb": "🌍 Odaberite jezik / Choose language / Выберите язык:",
        "rus": "🌍 Odaberite jezik / Choose language / Выберите язык:",
        "eng": "🌍 Odaberite jezik / Choose language / Выберите язык:",
    },
    "lang_set": {
        "srb": "🇷🇸 Jezik postavljen na *Srpski*.",
        "rus": "🇷🇺 Язык установлен: *Русский*.",
        "eng": "🇬🇧 Language set to *English*.",
    },
    "collecting": {
        "srb": (
            "🎯 *Ko ide večeras?*\n\n"
            "Napišite šta hoćete — skupljam odgovore *5 minuta* ⏱\n"
            "_(svako piše svoje, mogu i više poruka)_\n\n"
            "Kad su svi napisali, ukucajte */go* da odmah dobijete preporuke."
        ),
        "rus": (
            "🎯 *Кто идёт сегодня?*\n\n"
            "Напишите чего хотите — собираю ответы *5 минут* ⏱\n"
            "_(каждый пишет своё, можно несколько сообщений)_\n\n"
            "Когда все написали — введите */go* чтобы сразу получить результаты."
        ),
        "eng": (
            "🎯 *Who's going out tonight?*\n\n"
            "Write what you want — collecting replies for *5 minutes* ⏱\n"
            "_(everyone writes their own, multiple messages OK)_\n\n"
            "Once everyone has replied, type */go* to get recommendations right away."
        ),
    },
    "already_active": {
        "srb": "⏱ Već skupljam odgovore — {n} osoba je odgovorila.\nResetujem tajmer na 5 minuta.",
        "rus": "⏱ Уже собираю ответы — {n} человек ответили.\nСбрасываю таймер на 5 минут.",
        "eng": "⏱ Already collecting — {n} person(s) replied.\nResetting timer to 5 minutes.",
    },
    "acknowledged": {
        "srb": "✍️ _{name}_, zabeležio sam! ({n} osoba do sada)",
        "rus": "✍️ _{name}_, записал! ({n} чел. до сих пор)",
        "eng": "✍️ _{name}_, got it! ({n} person(s) so far)",
    },
    "processing": {
        "srb": "✅ Skupio sam odgovore od *{n}* člana, tražim najbolja mesta... 🔍",
        "rus": "✅ Собрал ответы от *{n}* участников, ищу лучшие места... 🔍",
        "eng": "✅ Collected replies from *{n}* member(s), searching for the best spots... 🔍",
    },
    "no_responses": {
        "srb": "⏰ Vreme je isteklo ali niko nije napisao šta hoće.\nKucajte */find* da pokušamo ponovo.",
        "rus": "⏰ Время вышло, но никто ничего не написал.\nВведите */find* чтобы попробовать снова.",
        "eng": "⏰ Time's up but nobody wrote anything.\nType */find* to try again.",
    },
    "no_results": {
        "srb": (
            "😕 Nisam pronašao mesta koja odgovaraju *svim* zahtevima.\n\n"
            "Pokušajte sa manje ograničenja ili kucajte */find* ponovo."
        ),
        "rus": (
            "😕 Не нашёл мест, которые подходят *всем* участникам.\n\n"
            "Попробуйте с меньшим числом ограничений или введите */find* снова."
        ),
        "eng": (
            "😕 Couldn't find places that work for *everyone*.\n\n"
            "Try with fewer restrictions or type */find* again."
        ),
    },
    "error": {
        "srb": "❌ Nešto je pošlo naopako. Pokušajte ponovo sa */find*.",
        "rus": "❌ Что-то пошло не так. Попробуйте снова с */find*.",
        "eng": "❌ Something went wrong. Try again with */find*.",
    },
    "no_session_go": {
        "srb": "Nema aktivne sesije. Koristite /find da pokrenete novu pretragu.",
        "rus": "Нет активной сессии. Используйте /find чтобы начать новый поиск.",
        "eng": "No active session. Use /find to start a new search.",
    },
    "new_search_prompt": {
        "srb": "Kucajte /find za novu pretragu.",
        "rus": "Введите /find для нового поиска.",
        "eng": "Type /find for a new search.",
    },
    # Results UI
    "results_header": {
        "srb": "🎯 <b>Top {n} mesta za vašu ekipu:</b>",
        "rus": "🎯 <b>Топ {n} мест для вашей компании:</b>",
        "eng": "🎯 <b>Top {n} places for your crew:</b>",
    },
    "score_label": {
        "srb": "Skor",
        "rus": "Оценка",
        "eng": "Score",
    },
    "not_ideal_label": {
        "srb": "Nije idealno",
        "rus": "Не идеально",
        "eng": "Not ideal",
    },
    "open_link": {
        "srb": "Otvori",
        "rus": "Открыть",
        "eng": "Open",
    },
    "show_all_btn": {
        "srb": "📋 Prikaži svih 10",
        "rus": "📋 Показать все 10",
        "eng": "📋 Show all 10",
    },
    "new_search_btn": {
        "srb": "🔄 Nova pretraga",
        "rus": "🔄 Новый поиск",
        "eng": "🔄 New search",
    },
    "no_consensus_warning": {
        "srb": "🔴 <i>Ni jedan rezultat nije idealan za celu grupu — evo najbliže što postoji:</i>\n\n",
        "rus": "🔴 <i>Ни один результат не идеален для всей группы — вот ближайшее что есть:</i>\n\n",
        "eng": "🔴 <i>No result is ideal for the whole group — here's the closest match:</i>\n\n",
    },
    "conflict_header": {
        "srb": "⚠️ <b>Suprotstavljene preference:</b>",
        "rus": "⚠️ <b>Противоречивые предпочтения:</b>",
        "eng": "⚠️ <b>Conflicting preferences:</b>",
    },
}


def t(key: str, lang: str, **kwargs: object) -> str:
    """Look up a translated string and optionally format it."""
    s = _T[key].get(lang) or _T[key].get("srb", "")
    return s.format(**kwargs) if kwargs else s


# ---------------------------------------------------------------------------
# Public message helpers
# ---------------------------------------------------------------------------

def msg_choose_language() -> str:
    return _T["choose_lang"]["srb"]


def msg_lang_set(lang: str) -> str:
    return t("lang_set", lang)


def msg_welcome(lang: str = "srb") -> str:
    return t("welcome", lang)


def msg_collecting(lang: str = "srb") -> str:
    return t("collecting", lang)


def msg_already_active(n_responded: int, lang: str = "srb") -> str:
    return t("already_active", lang, n=n_responded)


def msg_acknowledged(first_name: str, n_users: int, lang: str = "srb") -> str:
    return t("acknowledged", lang, name=first_name, n=n_users)


def msg_processing(n_users: int, lang: str = "srb") -> str:
    return t("processing", lang, n=n_users)


def msg_no_responses(lang: str = "srb") -> str:
    return t("no_responses", lang)


def msg_no_results(lang: str = "srb") -> str:
    return t("no_results", lang)


def msg_error(lang: str = "srb") -> str:
    return t("error", lang)


def msg_no_session_go(lang: str = "srb") -> str:
    return t("no_session_go", lang)


def msg_new_search_prompt(lang: str = "srb") -> str:
    return t("new_search_prompt", lang)


def format_results(
    ranked: list,
    mismatches: dict | None = None,
    reasons: list | None = None,
    *,
    show_all: bool = False,
    conflicts: list[str] | None = None,
    lang: str = "srb",
) -> str:
    """Build the full results message (Telegram HTML parse mode)."""
    limit = 10 if show_all else 5
    visible = ranked[:limit]
    mismatches = mismatches or {}
    reasons = list(reasons or [])

    conflict_block = ""
    if conflicts:
        bullets = "\n".join(f"  • <i>{_hesc(c)}</i>" for c in conflicts)
        conflict_block = t("conflict_header", lang) + f"\n{bullets}\n\n"

    best_min_soft = ranked[0][2] if ranked else 0.0
    no_consensus = best_min_soft < NO_CONSENSUS_THRESHOLD

    score_warning = t("no_consensus_warning", lang) if (no_consensus and not conflicts) else ""

    header = conflict_block + score_warning
    lines = [f"{header}" + t("results_header", lang, n=len(visible)) + "\n"]

    for i, (ev, sem, mn, mean) in enumerate(visible, 1):
        lines.extend(_format_event_block(i, ev, mn, mismatches, reasons, lang))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_event_block(
    rank: int,
    ev,
    min_soft: float,
    mismatches: dict,
    reasons: list[str],
    lang: str,
) -> list[str]:
    title = _safe(ev.event_description_resolved, 60)
    dot = _score_dot(min_soft)
    score_pct = f"{min_soft * 100:.0f}%"

    lines = [
        f"<b>{rank}. {_hesc(title)}</b>",
        f"   {dot} {t('score_label', lang)}: <b>{score_pct}</b>",
    ]

    reason = reasons[rank - 1] if rank - 1 < len(reasons) else ""
    if reason:
        lines.append(f"   💡 <i>{_hesc(reason)}</i>")

    gaps = mismatches.get(ev.event_id, [])
    if gaps:
        lines.append(f"   ⚠️ <i>{t('not_ideal_label', lang)}: {_hesc(', '.join(gaps))}</i>")

    lines.append(f'   <a href="{ev.link}">🔗 {t("open_link", lang)}</a>\n' if ev.link else "")
    return lines


def _safe(text: str | None, max_len: int) -> str:
    s = (text or "").replace("\n", " ").strip()
    return s[:max_len] + ("…" if len(s) > max_len else "")


def _score_dot(score: float) -> str:
    if score >= 0.70:
        return "🟢"
    if score >= 0.50:
        return "🟡"
    return "🔴"


def _hesc(text: str) -> str:
    """Escape HTML special chars for Telegram HTML parse mode."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
