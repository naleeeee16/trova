"""Per-group session state: tracks who responded and what they said."""

from __future__ import annotations

from dataclasses import dataclass, field

from belgrade_recommender.bot.config import COLLECTION_WINDOW_SECONDS


@dataclass
class GroupSession:
    triggered_by: int
    active: bool = True
    # user_id -> list of raw messages (concatenated later)
    inputs: dict[int, list[str]] = field(default_factory=dict)

    def add_message(self, user_id: int, text: str) -> None:
        self.inputs.setdefault(user_id, []).append(text.strip())

    def combined_inputs(self) -> dict[int, str]:
        """Return one merged string per user (all their messages joined)."""
        return {uid: " ".join(msgs) for uid, msgs in self.inputs.items()}

    @property
    def n_users(self) -> int:
        return len(self.inputs)


class SessionManager:
    """Thread-safe-enough for single-process asyncio bot."""

    def __init__(self) -> None:
        self._sessions: dict[int, GroupSession] = {}

    def open(self, chat_id: int, triggered_by: int) -> GroupSession:
        session = GroupSession(triggered_by=triggered_by)
        self._sessions[chat_id] = session
        return session

    def get(self, chat_id: int) -> GroupSession | None:
        return self._sessions.get(chat_id)

    def is_active(self, chat_id: int) -> bool:
        s = self._sessions.get(chat_id)
        return s is not None and s.active

    def add_message(self, chat_id: int, user_id: int, text: str) -> None:
        s = self._sessions.get(chat_id)
        if s and s.active:
            s.add_message(user_id, text)

    def close(self, chat_id: int) -> GroupSession | None:
        s = self._sessions.get(chat_id)
        if s:
            s.active = False
        return s
