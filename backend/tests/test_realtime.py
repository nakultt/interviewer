import asyncio

import numpy as np
from kec_runtime.domain import InterviewRecord, InterviewReport
from kec_runtime.realtime import RealtimeInterviewStream


class ResetSpy:
    def __init__(self) -> None:
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1


class UnexpectedVadCall(ResetSpy):
    def __call__(self, _frame: np.ndarray) -> float:
        raise AssertionError("playback audio must not reach voice activity detection")


def playback_stream() -> RealtimeInterviewStream:
    stream = object.__new__(RealtimeInterviewStream)
    stream.assistant_speaking = False
    stream.assistant_turn_pending = False
    stream.audio_generation = 0
    stream.accepting_audio = True
    stream.playback_cooldown_ms = 0
    stream.pending = np.ones(20, dtype=np.float32)
    stream.utterance = [np.ones(512, dtype=np.float32)]
    stream.preroll = []
    stream.endpoint = ResetSpy()
    stream.silero = UnexpectedVadCall()
    stream.turn_queue = asyncio.Queue()
    return stream


async def test_playback_boundary_clears_mic_audio_and_bypasses_vad() -> None:
    stream = playback_stream()

    stream._set_assistant_playback(True, 1)
    await stream._vad_frame(np.ones(512, dtype=np.float32), 1)

    assert stream.assistant_speaking
    assert stream.pending.size == 0
    assert stream.utterance == []
    assert stream.preroll == []
    assert stream.endpoint.reset_count == 1
    assert stream.silero.reset_count == 1


def test_playback_end_adds_echo_tail_cooldown() -> None:
    stream = playback_stream()
    stream._set_assistant_playback(True, 1)

    stream._set_assistant_playback(False, 1)

    assert not stream.assistant_speaking
    assert stream.playback_cooldown_ms == 320
    assert stream.endpoint.reset_count == 2
    assert stream.silero.reset_count == 2


async def test_audio_from_before_playback_is_discarded_after_playback() -> None:
    stream = playback_stream()
    stale_pcm = np.ones(512, dtype="<i2").tobytes()

    stream._set_assistant_playback(True, 1)
    stream._set_assistant_playback(False, 1)
    await stream._audio(stale_pcm, generation=0)

    assert stream.pending.size == 0


class FinishRepository:
    def __init__(self) -> None:
        self.report: InterviewReport | None = None

    async def save(self, _record: InterviewRecord) -> None:
        return None

    async def save_report(self, report: InterviewReport) -> None:
        self.report = report


class FinishAgent:
    async def evaluate(self, record: InterviewRecord) -> InterviewReport:
        return InterviewReport(
            interview_id=record.id,
            overall_score=0,
            summary="Ended by integrity policy.",
            strengths=[],
            improvements=[],
            action_plan=[],
            scores=[],
            integrity=record.integrity,
        )


async def test_three_people_strikes_finish_the_interview() -> None:
    stream = playback_stream()
    stream.auto_finish_started = False
    stream.record = InterviewRecord(
        candidate_name="Candidate",
        role_id="ai-engineer",
        role_title="AI Engineer",
        difficulty="mid",
        resume_filename="resume.txt",
        resume_text="Python",
        resume_skills=["Python"],
        competencies=["engineering"],
        status="active",
    )
    stream.repository = FinishRepository()
    stream.agent = FinishAgent()
    sent: list[dict[str, object]] = []

    async def capture(payload: dict[str, object]) -> None:
        sent.append(payload)

    stream._send = capture  # type: ignore[method-assign]

    await stream._finish_for_multiple_people()

    assert stream.auto_finish_started
    assert not stream.accepting_audio
    assert stream.record.status == "completed"
    assert stream.repository.report is not None
    assert sent[-1]["completion_reason"] == "multiple_people_limit"
