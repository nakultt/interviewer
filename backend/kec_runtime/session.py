from copy import deepcopy
from uuid import UUID

from .models import OutputIdentity, SessionAction, SessionSnapshot


class SessionController:
    """Authoritative in-process FSM for real-time interview state."""

    def __init__(self) -> None:
        self._sessions: dict[UUID, SessionSnapshot] = {}

    def create(self, session_id: UUID) -> SessionSnapshot:
        session = SessionSnapshot(session_id=session_id)
        self._sessions[session_id] = session
        return deepcopy(session)

    def get(self, session_id: UUID) -> SessionSnapshot:
        if session_id not in self._sessions:
            return self.create(session_id)
        return deepcopy(self._sessions[session_id])

    def dispatch(self, session_id: UUID, action: SessionAction) -> SessionSnapshot:
        session = self._sessions.setdefault(session_id, SessionSnapshot(session_id=session_id))
        kind = action.type
        if kind == "START" and session.state == "IDLE":
            session.state, session.turn_id = "PREPARING_QUESTION", 1
        elif kind == "GENERATION_STARTED" and action.generation_id:
            session.state = "AI_THINKING"
            session.active_output = OutputIdentity(
                generation_id=action.generation_id,
                playback_id=UUID(int=0),
            )
        elif kind == "PLAYBACK_STARTED" and action.output:
            session.state, session.active_output = "AI_SPEAKING", action.output
        elif kind == "SPEECH_STARTED" and session.state not in {"PAUSED", "COMPLETED"}:
            if session.active_output:
                generation_id = session.active_output.generation_id
                if generation_id not in session.cancelled_generation_ids:
                    session.cancelled_generation_ids.append(generation_id)
            session.state, session.active_output = "USER_SPEAKING", None
        elif kind == "SPEECH_ENDED" and session.state == "USER_SPEAKING":
            session.state = "ENDPOINT_PENDING"
        elif kind == "TRANSCRIPT_FINALIZED" and session.state == "ENDPOINT_PENDING":
            session.state = "EVALUATING"
        elif kind == "RESPONSE_FINISHED":
            session.state, session.active_output = "USER_SPEAKING", None
        elif kind == "PAUSE":
            session.state, session.active_output = "PAUSED", None
        elif kind == "RESUME" and session.state == "PAUSED":
            session.state = "USER_SPEAKING"
        elif kind == "COMPLETE":
            session.state, session.active_output = "COMPLETED", None
        elif kind == "FAIL":
            session.state, session.last_error = "ERROR_RECOVERY", action.message
        return deepcopy(session)
