import asyncio
from collections import defaultdict
from uuid import UUID

import uvicorn
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .interview_api import agent, kokoro, repository, router, vision, voice_guard, whisper
from .interviewer import Interviewer
from .models import (
    CreateSessionRequest,
    InterviewEvent,
    InterviewRequest,
    InterviewResponse,
    SessionAction,
    SessionSnapshot,
    VadFrame,
    VadResponse,
)
from .session import SessionController
from .store import EventStore
from .vad import FastVadGate

app = FastAPI(title="KEC Local Runtime", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.ui_origin, "http://127.0.0.1:3000"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["content-type"],
)
app.include_router(router)

store = EventStore(settings.data_dir)
interviewer = Interviewer(settings)
vad_sessions: defaultdict[UUID, FastVadGate] = defaultdict(FastVadGate)
sessions = SessionController()


@app.get("/health")
async def health() -> dict[str, object]:
    mongo_ready, model_status = await asyncio.gather(repository.ping(), agent.model_status())
    model_ready, model_detail = model_status
    return {
        "ok": model_ready or settings.mock_mode,
        "mode": "mock" if settings.mock_mode else "local",
        "runtime": "python-3.11.9",
        "model_detail": model_detail,
        "services": {
            "interviewer": (
                "mock" if settings.mock_mode else "lm-studio" if model_ready else "unavailable"
            ),
            "event_store": "ready",
            "vad": "deterministic-gate",
            "mongodb": "ready" if mongo_ready else "unavailable",
            "whisper": "ready" if whisper.available else "unavailable",
            "mediapipe": "ready" if vision.available else "unavailable",
            "silero_vad": "ready",
            "speaker_verification": "ready" if voice_guard.available else "unavailable",
            "kokoro_tts": "ready" if kokoro.available else "unavailable",
        },
    }


@app.post("/v1/events", status_code=status.HTTP_202_ACCEPTED)
async def append_event(event: InterviewEvent) -> dict[str, bool]:
    await store.append(event)
    return {"accepted": True}


@app.post("/v1/sessions", response_model=SessionSnapshot)
async def create_session(request: CreateSessionRequest) -> SessionSnapshot:
    return sessions.create(request.session_id)


@app.post("/v1/sessions/{session_id}/actions", response_model=SessionSnapshot)
async def session_action(session_id: UUID, action: SessionAction) -> SessionSnapshot:
    return sessions.dispatch(session_id, action)


@app.post("/v1/interview/respond", response_model=InterviewResponse)
async def respond(request: InterviewRequest) -> InterviewResponse:
    return await interviewer.respond(request)


@app.post("/v1/audio/vad", response_model=VadResponse)
async def vad(frame: VadFrame) -> VadResponse:
    decision = vad_sessions[frame.session_id].process(
        frame.rms, frame.duration_ms, frame.output_reference_rms
    )
    if decision.speech_started:
        snapshot = sessions.dispatch(frame.session_id, SessionAction(type="SPEECH_STARTED"))
    elif decision.speech_ended:
        snapshot = sessions.dispatch(frame.session_id, SessionAction(type="SPEECH_ENDED"))
    else:
        snapshot = sessions.get(frame.session_id)
    return VadResponse(
        speech_started=decision.speech_started,
        speech_ended=decision.speech_ended,
        echo_likely=decision.echo_likely,
        session=snapshot,
    )


def run() -> None:
    uvicorn.run("kec_runtime.main:app", host=settings.host, port=settings.port, reload=False)


if __name__ == "__main__":
    run()
