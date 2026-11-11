import hashlib
import hmac
import os
import secrets
import time
from dataclasses import dataclass, field


@dataclass
class Session:
    session_id: str
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    history: list[dict[str, str]] = field(default_factory=list)
    documents: list[dict] = field(default_factory=list)
    repositories: list[dict] = field(default_factory=list)


class SessionStore:
    """Short-lived in-memory sessions authenticated with signed random tokens."""

    def __init__(self) -> None:
        self._secret = os.getenv("SESSION_SECRET") or secrets.token_hex(32)
        self._sessions: dict[str, Session] = {}
        self.ttl_seconds = 24 * 60 * 60

    def create(self) -> Session:
        session_id = secrets.token_urlsafe(32)
        session = Session(session_id=session_id)
        self._sessions[session_id] = session
        return session

    def token_for(self, session: Session) -> str:
        signature = hmac.new(
            self._secret.encode(), session.session_id.encode(), hashlib.sha256
        ).hexdigest()
        return f"{session.session_id}.{signature}"

    def get_from_token(self, token: str) -> Session | None:
        try:
            session_id, signature = token.split(".", 1)
        except ValueError:
            return None
        expected = hmac.new(
            self._secret.encode(), session_id.encode(), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(signature, expected):
            return None
        session = self._sessions.get(session_id)
        if session is None or time.time() - session.last_seen > self.ttl_seconds:
            self._sessions.pop(session_id, None)
            return None
        session.last_seen = time.time()
        return session

    def remove(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


session_store = SessionStore()
