from unittest.mock import AsyncMock
from uuid import uuid4

from httpx import ASGITransport, AsyncClient
from kec_runtime.main import agent, app


async def test_health_and_python_owned_vad_session_transition(monkeypatch) -> None:
    monkeypatch.setattr(
        agent, "model_status", AsyncMock(return_value=(True, "qwen3.5-4b-mlx"))
    )
    session_id = str(uuid4())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        health = await client.get("/health")
        assert health.status_code == 200
        assert health.json()["runtime"] == "python-3.11.9"

        created = await client.post("/v1/sessions", json={"session_id": session_id})
        assert created.status_code == 200
        assert created.json()["state"] == "IDLE"

        decision = await client.post(
            "/v1/audio/vad",
            json={
                "session_id": session_id,
                "rms": 0.1,
                "duration_ms": 100,
                "output_reference_rms": 0,
            },
        )
    assert decision.status_code == 200
    assert decision.json()["speech_started"] is True
    assert decision.json()["session"]["state"] == "USER_SPEAKING"


async def test_role_catalog_is_available() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/v1/roles")
    assert response.status_code == 200
    role_ids = {role["id"] for role in response.json()}
    assert {"ai-engineer", "cloud-ops-engineer", "backend-engineer"} <= role_ids
