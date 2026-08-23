from unittest.mock import AsyncMock

import httpx
import pytest
from kec_runtime.agent import (
    AgentDecision,
    AnswerAssessment,
    AutonomousInterviewAgent,
    InterviewModelError,
    track_question_competency,
)
from kec_runtime.config import Settings
from kec_runtime.domain import InterviewRecord, Turn


def interview_record() -> InterviewRecord:
    return InterviewRecord(
        candidate_name="Nakul",
        role_id="ai-engineer",
        role_title="AI Engineer",
        difficulty="mid",
        target_questions=3,
        resume_filename="resume.txt",
        resume_text=(
            "Built a local retrieval system for support teams, cutting response time by 35%. "
            "Used Python, RAG evaluation, and production inference monitoring."
        ),
        resume_skills=["Python", "RAG"],
        competencies=["model development", "evaluation"],
        status="active",
    )


async def test_opening_is_always_a_self_introduction() -> None:
    agent = AutonomousInterviewAgent(Settings(mock_mode=False))
    generated = AgentDecision(
        question=(
            "Welcome, Nakul. Could you introduce yourself and share the experience most "
            "relevant to this role?"
        ),
        competency="professional narrative",
        phase="introduction",
    )
    structured = AsyncMock(return_value=generated)
    agent._structured = structured  # type: ignore[method-assign]

    opening = await agent.opening(interview_record())

    prompt = structured.await_args.args[0]
    assert "AI Engineer" in prompt
    assert "cutting response time by 35%" in prompt
    assert opening == generated
    assert opening.competency == "professional narrative"
    assert opening.phase == "introduction"
    assert not opening.should_finish


async def test_next_question_uses_the_answer_and_can_follow_its_natural_thread() -> None:
    agent = AutonomousInterviewAgent(Settings(mock_mode=False))
    record = interview_record()
    answer = "Offline relevance improved, but latency became our main production constraint."
    record.turns.extend(
        [
            Turn(role="interviewer", text="How did you evaluate the retrieval system?"),
            Turn(role="candidate", text=answer),
        ]
    )
    generated = AgentDecision(
        question=(
            "That latency trade-off is interesting. What did you change in production, and how "
            "did you verify that relevance did not regress?"
        ),
        competency="production inference",
    )
    structured = AsyncMock(
        side_effect=[
            AnswerAssessment(disposition="probe", reason="Credible approach needs evidence."),
            generated,
        ]
    )
    agent._structured = structured  # type: ignore[method-assign]

    decision = await agent.next_question(record, answer)

    assessment_prompt = structured.await_args_list[0].args[0]
    prompt = structured.await_args_list[1].args[0]
    assert answer in assessment_prompt
    assert answer in prompt
    assert record.resume_text in prompt
    assert "possibly mis-transcribed" in prompt
    assert "arbitrary or non-functional logic" in prompt
    assert "move to a different unexplored competency" in prompt
    assert "Do not repeat an earlier interviewer" in prompt
    assert "Required next move: probe" in prompt
    assert decision == generated


async def test_mock_fallback_starts_with_a_role_relevant_self_introduction() -> None:
    agent = AutonomousInterviewAgent(Settings(mock_mode=True))

    opening = await agent.opening(interview_record())

    assert opening.phase == "introduction"
    assert opening.competency == "professional narrative"
    assert "introduce yourself" in opening.question
    assert "AI Engineer" in opening.question
    assert "How are you doing today?" not in opening.question


async def test_interview_closes_after_the_configured_number_of_answers() -> None:
    agent = AutonomousInterviewAgent(Settings(mock_mode=True))
    record = interview_record()
    for index in range(record.target_questions):
        record.turns.extend(
            [
                Turn(role="interviewer", text=f"Question {index + 1}"),
                Turn(role="candidate", text=f"Answer {index + 1}"),
            ]
        )

    decision = await agent.next_question(record, record.turns[-1].text)

    assert decision.should_finish
    assert decision.phase == "closing"
    assert "?" not in decision.question


def test_dynamic_competencies_are_tracked_once() -> None:
    record = interview_record()
    decision = AgentDecision(question="How did you measure it?", competency="RAG reliability")

    track_question_competency(record, decision)
    track_question_competency(record, decision)

    assert record.covered_competencies == ["RAG reliability"]


async def test_local_model_failure_is_visible_instead_of_returning_fallback(
    monkeypatch,
) -> None:
    agent = AutonomousInterviewAgent(Settings(mock_mode=False))
    request = httpx.Request("POST", "http://127.0.0.1:1234/v1/chat/completions")
    monkeypatch.setattr(
        httpx.AsyncClient,
        "post",
        AsyncMock(side_effect=httpx.ConnectError("connection refused", request=request)),
    )

    with pytest.raises(InterviewModelError, match="Cannot reach LM Studio"):
        await agent.opening(interview_record())


def test_real_local_inference_is_the_default() -> None:
    assert Settings(_env_file=None).mock_mode is False
