from pathlib import Path

from kec_runtime.config import Settings
from kec_runtime.tts import KokoroTts, normalize_tts_text


def test_tts_text_is_cleanly_punctuated() -> None:
    assert normalize_tts_text("  **Hello**  world  ") == "Hello world."
    assert normalize_tts_text("- First point\n- Second point") == "First point Second point."
    assert normalize_tts_text("Hello,world!Next question") == "Hello, world! Next question."


def test_kokoro_requires_local_assets_by_default(tmp_path: Path) -> None:
    service = KokoroTts(Settings(kokoro_model_dir=tmp_path))

    assert not service.available
    try:
        service._ensure_assets()
    except RuntimeError as error:
        assert "not installed" in str(error)
    else:
        raise AssertionError("missing local assets must not trigger a download")
