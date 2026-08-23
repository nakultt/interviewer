from typing import Any

import httpx

from .config import Settings
from .models import InterviewRequest, InterviewResponse

FALLBACK = "What was the hardest trade-off in that situation, and how did you measure the outcome?"
SYSTEM_PROMPT = """You are a concise structured mock interviewer.
Ask exactly one relevant follow-up. Never infer technical ability from gaze, posture,
accent, or appearance. If interrupted, clarify briefly and restate the complete question.
Do not claim the candidate heard generated text that was not spoken."""


class Interviewer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def respond(self, request: InterviewRequest) -> InterviewResponse:
        if self.settings.mock_mode:
            return InterviewResponse(text=FALLBACK, mode="mock")

        messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(
            {
                "role": "user" if turn.role == "candidate" else "assistant",
                "content": turn.text,
            }
            for turn in request.recentTurns
        )
        messages.append({"role": "user", "content": request.answer})
        body: dict[str, Any] = {
            "model": self.settings.lm_studio_model,
            "temperature": 0.35,
            "max_tokens": 140,
            "messages": messages,
        }
        try:
            async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
                response = await client.post(
                    f"{self.settings.lm_studio_base_url.rstrip('/')}/chat/completions", json=body
                )
                response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"].strip()
            return InterviewResponse(text=content or FALLBACK, mode="local")
        except (httpx.HTTPError, KeyError, IndexError, TypeError, AttributeError) as error:
            return InterviewResponse(text=FALLBACK, mode="fallback", warning=str(error))
