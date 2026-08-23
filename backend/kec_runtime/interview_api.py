import asyncio
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, WebSocket, status
from fastapi.responses import Response

from .agent import (
    AutonomousInterviewAgent,
    InterviewModelError,
    add_turn,
    complete_record,
    track_question_competency,
)
from .config import settings
from .domain import InterviewRecord, InterviewReport
from .models import TtsRequest
from .realtime import RealtimeInterviewStream
from .repository import InterviewRepository
from .resume import extract_resume
from .roles import ROLES, get_role
from .stt import WhisperService
from .tts import KokoroTts
from .vision import VisionMonitor
from .voice import VoiceGuard

router = APIRouter(prefix="/v1")
repository = InterviewRepository(settings)
agent = AutonomousInterviewAgent(settings)
whisper = WhisperService(settings)
vision = VisionMonitor(settings.model_dir)
voice_guard = VoiceGuard(settings)
kokoro = KokoroTts(settings)


@router.get("/roles")
async def roles() -> list[dict[str, object]]:
    return [
        {
            "id": role.id,
            "title": role.title,
            "summary": role.summary,
            "competencies": role.competencies,
            "color": role.color,
        }
        for role in ROLES
    ]


@router.post("/tts")
async def synthesize_speech(request: TtsRequest) -> Response:
    try:
        audio = await asyncio.to_thread(kokoro.synthesize, request.text)
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return Response(content=audio, media_type="audio/wav")


@router.post("/interviews", response_model=InterviewRecord, status_code=status.HTTP_201_CREATED)
async def create_interview(
    resume: UploadFile = File(...),
    role_id: str = Form(...),
    candidate_name: str = Form("Candidate"),
    difficulty: str = Form("mid"),
    target_questions: int = Form(6),
) -> InterviewRecord:
    if difficulty not in {"junior", "mid", "senior"}:
        raise HTTPException(status_code=400, detail="Invalid difficulty")
    try:
        role = get_role(role_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Unknown role") from error
    content = await resume.read(settings.max_resume_bytes + 1)
    if len(content) > settings.max_resume_bytes:
        raise HTTPException(status_code=413, detail="Resume exceeds 8 MB")
    try:
        profile = extract_resume(resume.filename or "resume.txt", content)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    record = InterviewRecord(
        candidate_name=candidate_name.strip() or "Candidate",
        role_id=role.id,
        role_title=role.title,
        difficulty=difficulty,  # type: ignore[arg-type]
        target_questions=max(3, min(12, target_questions)),
        resume_filename=profile.filename,
        resume_text=profile.text,
        resume_skills=list(profile.skills),
        competencies=list(role.competencies),
        status="active",
    )
    try:
        opening = await agent.opening(record)
    except InterviewModelError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    add_turn(record, "interviewer", opening.question)
    track_question_competency(record, opening)
    record.conversation_phase = opening.phase
    await repository.save(record)
    return record


@router.get("/interviews/{interview_id}", response_model=InterviewRecord)
async def get_interview(interview_id: UUID) -> InterviewRecord:
    record = await repository.get(interview_id)
    if not record:
        raise HTTPException(status_code=404, detail="Interview not found")
    return record


@router.websocket("/interviews/{interview_id}/stream")
async def interview_stream(websocket: WebSocket, interview_id: UUID) -> None:
    record = await repository.get(interview_id)
    if not record or record.status != "active":
        await websocket.close(code=4404)
        return
    stream = RealtimeInterviewStream(
        websocket,
        record,
        settings,
        repository,
        agent,
        whisper,
        voice_guard,
        vision,
    )
    await stream.run()


@router.post("/interviews/{interview_id}/answer")
async def answer_interview(
    interview_id: UUID,
    text: str = Form(""),
    audio: UploadFile | None = File(None),
    audio_duration_ms: int | None = Form(None),
) -> dict[str, object]:
    record = await repository.get(interview_id)
    if not record:
        raise HTTPException(status_code=404, detail="Interview not found")
    if record.status != "active":
        raise HTTPException(status_code=409, detail="Interview is not active")
    voice_result: dict[str, object] = {"additional_voice_likely": False, "confidence": 0.0}
    transcript = text.strip()
    if audio:
        content = await audio.read(20_000_000)
        suffix = Path(audio.filename or "answer.webm").suffix or ".webm"
        voice_result = await voice_guard.analyze(interview_id, content, suffix)
        if bool(voice_result["additional_voice_likely"]):
            record.integrity.additional_voice_events += 1
            if "additional_voice_review" not in record.integrity.flags:
                record.integrity.flags.append("additional_voice_review")
        if not transcript:
            try:
                transcript = await whisper.transcribe(content, suffix)
            except RuntimeError as error:
                raise HTTPException(status_code=503, detail=str(error)) from error
    if not transcript:
        raise HTTPException(status_code=422, detail="No speech or typed answer was provided")
    add_turn(record, "candidate", transcript, audio_duration_ms)
    try:
        decision = await agent.next_question(record, transcript)
    except InterviewModelError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    record.conversation_phase = decision.phase
    report: InterviewReport | None = None
    if decision.should_finish:
        complete_record(record)
        try:
            report = await agent.evaluate(record)
        except InterviewModelError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        await repository.save_report(report)
    else:
        add_turn(record, "interviewer", decision.question)
        track_question_competency(record, decision)
    await repository.save(record)
    return {
        "transcript": transcript,
        "question": decision.question,
        "complete": decision.should_finish,
        "voice": voice_result,
        "record": record.model_dump(mode="json"),
        "report": report.model_dump(mode="json") if report else None,
    }


@router.post("/interviews/{interview_id}/vision")
async def analyze_frame(interview_id: UUID, frame: UploadFile = File(...)) -> dict[str, object]:
    record = await repository.get(interview_id)
    if not record:
        raise HTTPException(status_code=404, detail="Interview not found")
    content = await frame.read(2_000_000)
    record.integrity = vision.analyze(interview_id, content)
    await repository.save(record)
    return record.integrity.model_dump()


@router.post("/interviews/{interview_id}/finish", response_model=InterviewReport)
async def finish_interview(interview_id: UUID) -> InterviewReport:
    existing = await repository.get_report(interview_id)
    if existing:
        return existing
    record = await repository.get(interview_id)
    if not record:
        raise HTTPException(status_code=404, detail="Interview not found")
    complete_record(record)
    try:
        report = await agent.evaluate(record)
    except InterviewModelError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    await repository.save(record)
    await repository.save_report(report)
    return report


@router.get("/interviews/{interview_id}/report", response_model=InterviewReport)
async def get_report(interview_id: UUID) -> InterviewReport:
    report = await repository.get_report(interview_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return report
