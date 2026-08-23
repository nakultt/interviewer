import asyncio
import importlib
import importlib.util
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .config import Settings


class WhisperService:
    """Lazy MLX-Whisper adapter so startup stays fast and inference remains local."""

    def __init__(self, settings: Settings) -> None:
        self.model = settings.whisper_model
        self.cache_dir = settings.model_dir / "huggingface"
        self.allow_model_downloads = settings.allow_model_downloads
        os.environ.setdefault("HF_HOME", str(self.cache_dir))
        if not self.allow_model_downloads:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        self._module: Any | None = None

    @property
    def available(self) -> bool:
        return importlib.util.find_spec("mlx_whisper") is not None

    async def transcribe(self, content: bytes, suffix: str = ".webm") -> str:
        if not self.available:
            raise RuntimeError("mlx-whisper is not installed")
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as stream:
            stream.write(content)
            path = Path(stream.name)
        try:
            return await asyncio.to_thread(self._transcribe_sync, path)
        finally:
            path.unlink(missing_ok=True)

    async def transcribe_samples(self, samples: np.ndarray, prompt: str = "") -> str:
        if not self.available:
            raise RuntimeError("mlx-whisper is not installed")
        normalized = np.asarray(samples, dtype=np.float32)
        return await asyncio.to_thread(self._transcribe_samples_sync, normalized, prompt)

    def _transcribe_sync(self, path: Path) -> str:
        return self._run_transcription(str(path), "")

    def _transcribe_samples_sync(self, samples: np.ndarray, prompt: str) -> str:
        return self._run_transcription(samples, prompt)

    def _run_transcription(self, audio: str | np.ndarray, prompt: str) -> str:
        if self._module is None:
            self._module = importlib.import_module("mlx_whisper")
        result: dict[str, Any] = self._module.transcribe(
            audio,
            path_or_hf_repo=self._model_location(),
            language="en",
            temperature=0.0,
            initial_prompt=prompt or None,
            condition_on_previous_text=False,
            no_speech_threshold=0.5,
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.4,
            hallucination_silence_threshold=1.0,
            word_timestamps=True,
        )
        text = str(result.get("text", "")).strip()
        return " ".join(text.split())

    def _model_location(self) -> str:
        """Use the resolved local Hub snapshot so mlx-whisper never revalidates remotely."""
        repository = self.cache_dir / "hub" / f"models--{self.model.replace('/', '--')}"
        ref = repository / "refs" / "main"
        revision = ref.read_text().strip() if ref.exists() else ""
        snapshot = repository / "snapshots" / revision
        if snapshot.is_dir() and (snapshot / "config.json").is_file() and any(
            (snapshot / filename).is_file() for filename in ("weights.safetensors", "weights.npz")
        ):
            return str(snapshot)
        if self.allow_model_downloads:
            return self.model
        raise RuntimeError(
            f"Whisper model is not installed at {snapshot}. "
            "Install it once, or set KEC_ALLOW_MODEL_DOWNLOADS=true for setup."
        )
