from pathlib import Path
from uuid import uuid4

from kec_runtime.vision import FrameSignal, VisionMonitor


def test_summary_exposes_latest_frame_for_live_monitoring() -> None:
    monitor = VisionMonitor(Path("unused"))
    interview_id = uuid4()
    monitor._record_signal(interview_id, FrameSignal(100, 1, True, True, 0.9))
    monitor._record_signal(interview_id, FrameSignal(200, 2, True, False, 0.4))

    summary = monitor.summary(interview_id)

    assert summary.current_face_count == 2
    assert summary.current_face_visible
    assert not summary.current_gaze_center
    assert summary.current_posture_stability == 0.4
    assert summary.multiple_people_events == 1


def test_multiple_people_strikes_are_capped_at_three() -> None:
    monitor = VisionMonitor(Path("unused"))
    interview_id = uuid4()
    for timestamp in (100, 5_100, 10_100, 15_100):
        monitor._record_signal(interview_id, FrameSignal(timestamp, 2, True, True, 0.8))

    summary = monitor.summary(interview_id)

    assert summary.multiple_people_events == 3
    assert "multiple_people_limit_reached" in summary.flags


def test_multiple_people_strikes_have_five_second_buffer() -> None:
    monitor = VisionMonitor(Path("unused"))
    interview_id = uuid4()
    for timestamp in (100, 500, 1_500, 4_999):
        monitor._record_signal(interview_id, FrameSignal(timestamp, 2, True, True, 0.8))

    assert monitor.summary(interview_id).multiple_people_events == 1

    monitor._record_signal(interview_id, FrameSignal(5_100, 2, True, True, 0.8))
    assert monitor.summary(interview_id).multiple_people_events == 2
