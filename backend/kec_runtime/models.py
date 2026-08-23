from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field


class InterviewEvent(BaseModel):
    event_id: UUID
    session_id: UUID
    turn_id: int = Field(ge=0)
    timestamp_monotonic_ms: int = Field(ge=0)
    timestamp_wallclock: datetime
    source: Literal["ui", "vad", "barge_in_controller", "stt", "interviewer", "system"]
    type: str = Field(min_length=1, max_length=100)
    payload: dict[str, Any]


class TranscriptItem(BaseModel):
    role: Literal["interviewer", "candidate"]
    text: str = Field(min_length=1, max_length=12_000)


class InterviewRequest(BaseModel):
    sessionId: UUID
    turnId: int = Field(ge=0)
    answer: str = Field(min_length=1, max_length=12_000)
    recentTurns: list[TranscriptItem] = Field(max_length=8)


class InterviewResponse(BaseModel):
    text: str
    mode: Literal["local", "mock", "fallback"]
    warning: str | None = None


class VadFrame(BaseModel):
    session_id: UUID
    rms: float = Field(ge=0)
    duration_ms: int = Field(gt=0, le=1000)
    output_reference_rms: float = Field(default=0, ge=0)


class TtsRequest(BaseModel):
    text: str = Field(min_length=1, max_length=12_000)


class CreateSessionRequest(BaseModel):
    session_id: UUID


class OutputIdentity(BaseModel):
    generation_id: UUID
    playback_id: UUID
    spoken_text: str = ""
    generated_text: str = ""


class SessionAction(BaseModel):
    type: Literal[
        "START",
        "GENERATION_STARTED",
        "PLAYBACK_STARTED",
        "SPEECH_STARTED",
        "SPEECH_ENDED",
        "TRANSCRIPT_FINALIZED",
        "RESPONSE_FINISHED",
        "PAUSE",
        "RESUME",
        "COMPLETE",
        "FAIL",
    ]
    generation_id: UUID | None = None
    output: OutputIdentity | None = None
    message: str | None = None


class SessionSnapshot(BaseModel):
    session_id: UUID
    state: str = "IDLE"
    turn_id: int = 0
    active_output: OutputIdentity | None = None
    cancelled_generation_ids: list[UUID] = Field(default_factory=list)
    last_error: str | None = None


class VadResponse(BaseModel):
    speech_started: bool
    speech_ended: bool
    echo_likely: bool
    session: SessionSnapshot
