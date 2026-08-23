from pathlib import Path

import pytest
from kec_runtime.config import Settings
from kec_runtime.stt import WhisperService
from kec_runtime.voice import VoiceGuard


def test_whisper_uses_cached_snapshot_without_network(tmp_path: Path) -> None:
    settings = Settings(model_dir=tmp_path)
    snapshot = (
        tmp_path
        / "huggingface"
        / "hub"
        / "models--mlx-community--whisper-large-v3-turbo"
        / "snapshots"
        / "revision"
    )
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    (snapshot / "weights.safetensors").write_bytes(b"weights")
    (snapshot.parent.parent / "refs").mkdir()
    (snapshot.parent.parent / "refs" / "main").write_text("revision\n")

    assert WhisperService(settings)._model_location() == str(snapshot)


def test_missing_models_fail_instead_of_downloading(tmp_path: Path) -> None:
    settings = Settings(model_dir=tmp_path)

    with pytest.raises(RuntimeError, match="not installed"):
        WhisperService(settings)._model_location()
    with pytest.raises(RuntimeError, match="not installed"):
        VoiceGuard(settings)._model_location()
