import asyncio
import importlib
import importlib.util
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
import soundfile as sf  # type: ignore[import-untyped]
import torch

from .config import Settings


class VoiceGuard:
    """ECAPA-TDNN speaker verification with conservative, review-only flags."""

    def __init__(self, settings: Settings) -> None:
        self.model_source = settings.speaker_model
        self.savedir = settings.model_dir / "speaker" / "ecapa-voxceleb"
        self.allow_model_downloads = settings.allow_model_downloads
        if not self.allow_model_downloads:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        self._classifier: Any | None = None
        self._enrollment: dict[UUID, np.ndarray] = {}
        self._enrollment_count: dict[UUID, int] = {}
        self._load_error: str | None = None

    @property
    def available(self) -> bool:
        return importlib.util.find_spec("speechbrain") is not None

    async def analyze(self, interview_id: UUID, content: bytes, suffix: str) -> dict[str, object]:
        samples = await asyncio.to_thread(self._decode, content, suffix)
        return await self.analyze_samples(interview_id, samples)

    async def analyze_samples(self, interview_id: UUID, samples: np.ndarray) -> dict[str, object]:
        return await asyncio.to_thread(self._analyze_samples_sync, interview_id, samples)

    def _load(self) -> Any | None:
        if self._classifier is not None or self._load_error is not None:
            return self._classifier
        try:
            module = importlib.import_module("speechbrain.inference.speaker")
            classifier_type = module.EncoderClassifier
            self.savedir.mkdir(parents=True, exist_ok=True)
            self._classifier = classifier_type.from_hparams(
                source=self._model_location(),
                savedir=str(self.savedir),
                run_opts={"device": "cpu"},
            )
        except Exception as error:
            self._load_error = str(error)
        return self._classifier

    def _model_location(self) -> str:
        if (self.savedir / "hyperparams.yaml").is_file() and (
            self.savedir / "embedding_model.ckpt"
        ).is_file():
            return str(self.savedir)
        if self.allow_model_downloads:
            return self.model_source
        raise RuntimeError(
            f"Speaker model is not installed at {self.savedir}. "
            "Install it once, or set KEC_ALLOW_MODEL_DOWNLOADS=true for setup."
        )

    def _analyze_samples_sync(self, interview_id: UUID, samples: np.ndarray) -> dict[str, object]:
        windows = self._speech_windows(np.asarray(samples, dtype=np.float32))
        if not windows:
            return self._result(False, 0.0, 0, "insufficient_speech")
        classifier = self._load()
        if classifier is None:
            return self._spectral_fallback(interview_id, windows)
        embeddings = [self._ecapa_embedding(classifier, window) for window in windows]
        reference = self._enrollment.get(interview_id)
        current = self._normalize(np.mean(embeddings, axis=0))
        if reference is None:
            self._enrollment[interview_id] = current
            self._enrollment_count[interview_id] = 1
            return self._result(False, 0.0, len(windows), "ecapa_enrolled")

        similarities = [float(np.dot(reference, embedding)) for embedding in embeddings]
        outlier_ratio = sum(similarity < 0.25 for similarity in similarities) / len(similarities)
        pairwise = [
            float(np.dot(embeddings[left], embeddings[right]))
            for left in range(len(embeddings))
            for right in range(left + 1, len(embeddings))
        ]
        internal_mismatch = bool(pairwise) and min(pairwise) < 0.12
        additional = (outlier_ratio >= 0.5 and len(windows) >= 2) or internal_mismatch
        confidence = min(0.99, max(outlier_ratio, 0.7 if internal_mismatch else 0.0))
        if not additional and float(np.mean(similarities)) >= 0.35:
            count = min(5, self._enrollment_count.get(interview_id, 1) + 1)
            updated = self._normalize(reference * (count - 1) + current)
            self._enrollment[interview_id] = updated
            self._enrollment_count[interview_id] = count
        return self._result(additional, confidence, len(windows), "ecapa_tdnn")

    @staticmethod
    def _ecapa_embedding(classifier: Any, samples: np.ndarray) -> np.ndarray:
        waveform = torch.from_numpy(samples).unsqueeze(0)
        with torch.inference_mode():
            embedding = classifier.encode_batch(waveform).squeeze().detach().cpu().numpy()
        return VoiceGuard._normalize(np.asarray(embedding, dtype=np.float32))

    @staticmethod
    def _speech_windows(samples: np.ndarray) -> list[np.ndarray]:
        window, hop = 25_600, 12_800  # 1.6 s windows, 0.8 s overlap at 16 kHz.
        if samples.size < 12_800:
            return []
        if samples.size < window:
            samples = np.pad(samples, (0, window - samples.size))
        windows: list[np.ndarray] = []
        for start in range(0, samples.size - window + 1, hop):
            chunk = samples[start : start + window]
            if float(np.sqrt(np.mean(chunk * chunk))) >= 0.008:
                windows.append(chunk)
        return windows

    def _spectral_fallback(
        self, interview_id: UUID, windows: list[np.ndarray]
    ) -> dict[str, object]:
        embeddings = [self._spectral_embedding(window) for window in windows]
        current = self._normalize(np.mean(embeddings, axis=0))
        reference = self._enrollment.get(interview_id)
        if reference is None:
            self._enrollment[interview_id] = current
            return self._result(False, 0.0, len(windows), "spectral_fallback")
        similarities = [float(np.dot(reference, item)) for item in embeddings]
        outlier_ratio = sum(value < 0.62 for value in similarities) / len(similarities)
        additional = outlier_ratio >= 0.6 and len(windows) >= 2
        return self._result(additional, outlier_ratio, len(windows), "spectral_fallback")

    @staticmethod
    def _spectral_embedding(samples: np.ndarray) -> np.ndarray:
        spectrum = np.abs(np.fft.rfft(samples * np.hanning(samples.size), n=4096))[:512]
        bands = np.log1p(spectrum).reshape(64, 8).mean(axis=1)
        bands -= bands.mean()
        return VoiceGuard._normalize(bands)

    @staticmethod
    def _normalize(values: np.ndarray) -> np.ndarray:
        return np.asarray(values / (np.linalg.norm(values) + 1e-9), dtype=np.float32)

    @staticmethod
    def _result(
        additional: bool, confidence: float, windows: int, method: str
    ) -> dict[str, object]:
        return {
            "additional_voice_likely": additional,
            "confidence": round(float(confidence), 3),
            "windows": windows,
            "method": method,
        }

    @staticmethod
    def _decode(content: bytes, suffix: str) -> np.ndarray:
        with tempfile.TemporaryDirectory(prefix="kec-voice-") as directory:
            source = Path(directory) / f"input{suffix}"
            wave = Path(directory) / "mono.wav"
            source.write_bytes(content)
            subprocess.run(
                [
                    "ffmpeg",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(source),
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    str(wave),
                ],
                check=True,
                timeout=30,
            )
            samples, _ = sf.read(wave, dtype="float32")
        return np.asarray(samples, dtype=np.float32)
