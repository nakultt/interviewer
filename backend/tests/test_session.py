from uuid import uuid4

from kec_runtime.models import OutputIdentity, SessionAction
from kec_runtime.session import SessionController


def test_barge_in_cancels_generation_in_python_controller() -> None:
    controller = SessionController()
    session_id, generation_id = uuid4(), uuid4()
    controller.create(session_id)
    controller.dispatch(session_id, SessionAction(type="START"))
    controller.dispatch(
        session_id,
        SessionAction(
            type="PLAYBACK_STARTED",
            output=OutputIdentity(
                generation_id=generation_id,
                playback_id=uuid4(),
                spoken_text="Tell me about",
                generated_text="Tell me about a difficult project.",
            ),
        ),
    )
    result = controller.dispatch(session_id, SessionAction(type="SPEECH_STARTED"))
    assert result.state == "USER_SPEAKING"
    assert result.active_output is None
    assert generation_id in result.cancelled_generation_ids
