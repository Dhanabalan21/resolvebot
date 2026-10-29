"""Chat session memory: owner, message history and the one waiting proposal.

Sessions live in process memory (assessment §2.5: "remember each chat's
messages and its waiting proposal by session_id"). Nothing is written to the
store database for a chat, so the only rows ever written there are confirmed
return requests and support tickets. Sessions are lost on restart; see README.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field

MAX_HISTORY_MESSAGES = 20


class SessionForbidden(Exception):
    """The session belongs to a different customer."""


@dataclass
class Session:
    session_id: str
    customer_id: str
    messages: list[dict] = field(default_factory=list)
    proposal: dict | None = None
    # Serialises requests within one session (e.g. two concurrent "yes" replies).
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def remember(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        del self.messages[:-MAX_HISTORY_MESSAGES]


class SessionStore:
    def __init__(self, max_sessions: int = 10_000):
        self._sessions: OrderedDict[str, Session] = OrderedDict()
        self._lock = threading.Lock()
        self._max = max_sessions

    def claim(self, session_id: str, customer_id: str) -> Session:
        """Return the customer's session, creating it on first use.
        Raises SessionForbidden if another customer started it."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = Session(session_id=session_id, customer_id=customer_id)
                self._sessions[session_id] = session
                while len(self._sessions) > self._max:
                    self._sessions.popitem(last=False)
            elif session.customer_id != customer_id:
                raise SessionForbidden(session_id)
            self._sessions.move_to_end(session_id)
            return session
