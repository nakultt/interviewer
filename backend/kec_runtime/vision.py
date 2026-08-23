import importlib
import importlib.util
import io
import multiprocessing
import os
import time
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
from PIL import Image

from .domain import IntegritySummary

_WORKER_FACE: Any | None = None
_WORKER_POSE: Any | None = None
_WORKER_TIMESTAMP = 0
MULTIPLE_PEOPLE_STRIKE_BUFFER_MS = 5_000


@dataclass(slots=True)
class FrameSignal:
    timestamp_ms: int
    face_count: int
    face_visible: bool
    gaze_center: bool
    posture_stability: float


def _initialize_worker(model_dir: str) -> None:
    global _WORKER_FACE, _WORKER_POSE
    if _WORKER_FACE is not None:
        return
    cache_dir = Path(model_dir).parent / "cache" / "matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    tasks_python = importlib.import_module("mediapipe.tasks.python")
    vision = importlib.import_module("mediapipe.tasks.python.vision")
    cpu = tasks_python.BaseOptions.Delegate.CPU
    face_options = vision.FaceLandmarkerOptions(
        base_options=tasks_python.BaseOptions(
            model_asset_path=str(Path(model_dir) / "face_landmarker.task"), delegate=cpu
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=3,
        output_face_blendshapes=False,
        min_face_detection_confidence=0.55,
        min_face_presence_confidence=0.55,
        min_tracking_confidence=0.55,
    )
    pose_options = vision.PoseLandmarkerOptions(
        base_options=tasks_python.BaseOptions(
            model_asset_path=str(Path(model_dir) / "pose_landmarker_lite.task"), delegate=cpu
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.55,
        min_pose_presence_confidence=0.55,
        min_tracking_confidence=0.55,
    )
    _WORKER_FACE = vision.FaceLandmarker.create_from_options(face_options)
    _WORKER_POSE = vision.PoseLandmarker.create_from_options(pose_options)


def _analyze_worker(jpeg: bytes, model_dir: str, timestamp_ms: int) -> tuple[int, bool, float]:
    global _WORKER_TIMESTAMP
    _initialize_worker(model_dir)
    assert _WORKER_FACE is not None
    assert _WORKER_POSE is not None
    mediapipe = importlib.import_module("mediapipe")
    image = np.ascontiguousarray(Image.open(io.BytesIO(jpeg)).convert("RGB"))
    mp_image = mediapipe.Image(image_format=mediapipe.ImageFormat.SRGB, data=image)
    _WORKER_TIMESTAMP = max(_WORKER_TIMESTAMP + 1, timestamp_ms)
    face_result = _WORKER_FACE.detect_for_video(mp_image, _WORKER_TIMESTAMP)
    pose_result = _WORKER_POSE.detect_for_video(mp_image, _WORKER_TIMESTAMP)
    faces = list(face_result.face_landmarks or [])
    poses = list(pose_result.pose_landmarks or [])
    gaze_center = _gaze_is_centered(faces[0]) if faces else False
    posture = _posture_stability(poses[0] if poses else None)
    return len(faces), gaze_center, posture


def _gaze_is_centered(landmarks: list[Any]) -> bool:
    if len(landmarks) < 478:
        return True
    left_iris_x = sum(landmarks[index].x for index in (468, 469, 470, 471, 472)) / 5
    right_iris_x = sum(landmarks[index].x for index in (473, 474, 475, 476, 477)) / 5
    left_ratio = (left_iris_x - landmarks[33].x) / (landmarks[133].x - landmarks[33].x + 1e-6)
    right_ratio = (right_iris_x - landmarks[362].x) / (landmarks[263].x - landmarks[362].x + 1e-6)
    return bool(0.22 <= left_ratio <= 0.78 and 0.22 <= right_ratio <= 0.78)


def _posture_stability(pose_landmarks: list[Any] | None) -> float:
    if pose_landmarks is None:
        return 0.0
    shoulder_tilt = abs(pose_landmarks[11].y - pose_landmarks[12].y)
    return float(max(0.0, min(1.0, 1.0 - shoulder_tilt * 5)))


class VisionMonitor:
    """MediaPipe analysis isolated from the API process, with a safe face fallback."""

    def __init__(self, model_dir: Path) -> None:
        self._model_dir = model_dir
        self._executor: ProcessPoolExecutor | None = None
        self._history: defaultdict[UUID, deque[FrameSignal]] = defaultdict(
            lambda: deque(maxlen=240)
        )
        self._multiple_people_strikes: defaultdict[UUID, int] = defaultdict(int)
        self._last_multiple_people_strike_ms: dict[UUID, int] = {}
        self._error: str | None = None

    @property
    def available(self) -> bool:
        return (
            importlib.util.find_spec("mediapipe") is not None
            and (self._model_dir / "face_landmarker.task").exists()
            and (self._model_dir / "pose_landmarker_lite.task").exists()
        )

    @property
    def error(self) -> str | None:
        return self._error

    def analyze(
        self, interview_id: UUID, jpeg: bytes, timestamp_ms: int | None = None
    ) -> IntegritySummary:
        timestamp_ms = timestamp_ms or int(time.monotonic() * 1000)
        try:
            if self._executor is None:
                self._executor = ProcessPoolExecutor(
                    max_workers=1, mp_context=multiprocessing.get_context("spawn")
                )
            result = self._executor.submit(
                _analyze_worker, jpeg, str(self._model_dir), timestamp_ms
            ).result(timeout=20)
            face_count, gaze_center, posture = result
        except (Exception, TimeoutError) as error:
            self._error = str(error)
            if self._executor:
                self._executor.shutdown(wait=False, cancel_futures=True)
                self._executor = None
            face_count = self._fallback_face_count(jpeg)
            gaze_center, posture = bool(face_count), 0.5 if face_count else 0.0
        self._record_signal(
            interview_id,
            FrameSignal(timestamp_ms, face_count, bool(face_count), gaze_center, posture),
        )
        return self.summary(interview_id)

    def _record_signal(self, interview_id: UUID, signal: FrameSignal) -> None:
        self._history[interview_id].append(signal)
        if signal.face_count <= 1 or self._multiple_people_strikes[interview_id] >= 3:
            return
        previous = self._last_multiple_people_strike_ms.get(interview_id)
        if previous is not None and (
            signal.timestamp_ms - previous < MULTIPLE_PEOPLE_STRIKE_BUFFER_MS
        ):
            return
        self._multiple_people_strikes[interview_id] += 1
        self._last_multiple_people_strike_ms[interview_id] = signal.timestamp_ms

    def summary(self, interview_id: UUID) -> IntegritySummary:
        frames = list(self._history[interview_id])
        if not frames:
            return IntegritySummary(frames_analyzed=0)
        latest = frames[-1]
        # Strikes are persistent and spaced apart, so rapid camera samples cannot exhaust the
        # policy immediately and old history eviction cannot make the counter decrease.
        multiple = self._multiple_people_strikes[interview_id]
        flags: list[str] = []
        if multiple >= 3:
            flags.append("multiple_people_limit_reached")
        visible_ratio = sum(frame.face_visible for frame in frames) / len(frames)
        if len(frames) >= 10 and visible_ratio < 0.65:
            flags.append("candidate_frequently_out_of_frame")
        return IntegritySummary(
            frames_analyzed=len(frames),
            face_visible_ratio=round(visible_ratio, 3),
            center_gaze_ratio=round(sum(frame.gaze_center for frame in frames) / len(frames), 3),
            posture_stability=round(
                sum(frame.posture_stability for frame in frames) / len(frames), 3
            ),
            current_face_count=latest.face_count,
            current_face_visible=latest.face_visible,
            current_gaze_center=latest.gaze_center,
            current_posture_stability=round(latest.posture_stability, 3),
            multiple_people_events=multiple,
            flags=flags,
        )

    @staticmethod
    def _fallback_face_count(jpeg: bytes) -> int:
        try:
            cv2 = importlib.import_module("cv2")
            image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            cascade = cv2.CascadeClassifier(
                str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
            )
            return len(cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5))
        except Exception:
            return 0
