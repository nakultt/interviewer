import importlib
import importlib.util
import io
import re
import threading
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf  # type: ignore[import-untyped]

from .config import Settings

SAMPLE_RATE = 24_000
MODEL_REPOSITORY = "hexgrad/Kokoro-82M"


def normalize_tts_text(text: str) -> str:
    """Turn generated interview text into clean, sentence-delimited Kokoro input."""
    normalized = text.replace("\r", "\n")
    normalized = re.sub(r"(?m)^\s*(?:[-•]|\d+[.)])\s+", "", normalized)
    normalized = normalized.replace("`", "").replace("**", "").replace("__", "")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = re.sub(r"\s+([,.;:!?])", r"\1", normalized)
    normalized = re.sub(r"([,;:])(?=[A-Za-z])", r"\1 ", normalized)
    normalized = re.sub(r"([.!?])(?=[A-Z])", r"\1 ", normalized)
    if normalized and normalized[-1] not in ".!?":
        normalized += "."
    return normalized[:2_000]


class KokoroTts:
    """Local Kokoro 82M synthesis using the American English af_heart voice."""

    def __init__(self, settings: Settings) -> None:
        self.model_dir = settings.kokoro_model_dir
        self.voice = settings.kokoro_voice
        self.allow_model_downloads = settings.allow_model_downloads
        self._pipeline: Any | None = None
        self._load_lock = threading.Lock()

    @property
    def available(self) -> bool:
        return importlib.util.find_spec("kokoro") is not None and self._assets_ready()

    def synthesize(self, text: str) -> bytes:
        normalized = normalize_tts_text(text)
        if not normalized:
            raise ValueError("Text is required for speech synthesis")
        pipeline = self._load_pipeline()
        chunks = [
            np.asarray(audio, dtype=np.float32)
            for _, _, audio in pipeline(normalized, voice=str(self._voice_path()), speed=1.0)
        ]
        if not chunks:
            raise RuntimeError("Kokoro produced no audio")
        silence = np.zeros(int(SAMPLE_RATE * 0.12), dtype=np.float32)
        audio = np.concatenate([item for chunk in chunks for item in (chunk, silence)])
        output = io.BytesIO()
        sf.write(output, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        return output.getvalue()

    def _load_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        with self._load_lock:
            if self._pipeline is not None:
                return self._pipeline
            self._ensure_assets()
            kokoro = importlib.import_module("kokoro")
            model_module = importlib.import_module("kokoro.model")
            model = model_module.KModel(
                repo_id=MODEL_REPOSITORY,
                config=str(self.model_dir / "config.json"),
                model=str(self.model_dir / "kokoro-v1_0.pth"),
            )
            self._pipeline = kokoro.KPipeline(
                lang_code="a", repo_id=MODEL_REPOSITORY, model=model, device="cpu"
            )
        return self._pipeline

    def _ensure_assets(self) -> None:
        if self._assets_ready():
            return
        if not self.allow_model_downloads:
            raise RuntimeError(
                f"Kokoro model is not installed at {self.model_dir}. Run one-time setup with "
                "KEC_ALLOW_MODEL_DOWNLOADS=true."
            )
        hub = importlib.import_module("huggingface_hub")
        hub.snapshot_download(
            repo_id=MODEL_REPOSITORY,
            local_dir=str(self.model_dir),
            allow_patterns=["config.json", "kokoro-v1_0.pth", f"voices/{self.voice}.pt"],
        )
        if not self._assets_ready():
            raise RuntimeError("Kokoro model setup completed without the required af_heart assets")

    def _assets_ready(self) -> bool:
        return (
            (self.model_dir / "config.json").is_file()
            and (self.model_dir / "kokoro-v1_0.pth").is_file()
            and self._voice_path().is_file()
        )

    def _voice_path(self) -> Path:
        return self.model_dir / "voices" / f"{self.voice}.pt"
