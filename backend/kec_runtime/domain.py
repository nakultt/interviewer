from datetime import UTC, datetime
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class Turn(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    role: Literal["interviewer", "candidate"]
    text: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    audio_duration_ms: int | None = None


class IntegritySummary(BaseModel):
    frames_analyzed: int = 0
    face_visible_ratio: float = 1.0
    center_gaze_ratio: float = 1.0
    posture_stability: float = 1.0
    current_face_count: int = 0
    current_face_visible: bool = False
    current_gaze_center: bool = False
    current_posture_stability: float = 0.0
    multiple_people_events: int = 0
    additional_voice_events: int = 0
    flags: list[str] = Field(default_factory=list)


class InterviewRecord(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    candidate_name: str
    role_id: str
    role_title: str
    difficulty: Literal["junior", "mid", "senior"]
    target_questions: int = Field(default=6, ge=3, le=12)
    status: Literal["ready", "active", "completed", "error"] = "ready"
    conversation_phase: Literal["rapport", "introduction", "competency", "closing"] = "rapport"
    resume_filename: str
    resume_text: str
    resume_skills: list[str]
    competencies: list[str]
    turns: list[Turn] = Field(default_factory=list)
    covered_competencies: list[str] = Field(default_factory=list)
    integrity: IntegritySummary = Field(default_factory=IntegritySummary)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None


class ScoreItem(BaseModel):
    competency: str
    score: int = Field(ge=0, le=100)
    evidence: str
    improvement: str


class InterviewReport(BaseModel):
    interview_id: UUID
    overall_score: int = Field(ge=0, le=100)
    summary: str
    strengths: list[str]
    improvements: list[str]
    action_plan: list[str]
    scores: list[ScoreItem]
    integrity: IntegritySummary
