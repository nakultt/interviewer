from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="KEC_", extra="ignore")

    host: str = "127.0.0.1"
    port: int = 8765
    ui_origin: str = "http://localhost:3000"
    data_dir: Path = Path("data/sessions")
    lm_studio_base_url: str = "http://127.0.0.1:1234/v1"
    lm_studio_model: str = "qwen3.5-4b-mlx"
    # Real local inference is the normal product mode. Mock mode must be opted into explicitly.
    mock_mode: bool = False
    request_timeout_seconds: float = 25.0
    mongodb_uri: SecretStr | None = None
    mongodb_database: str = "kec_interviewer"
    whisper_model: str = "mlx-community/whisper-large-v3-turbo"
    model_dir: Path = Path("data/models")
    kokoro_model_dir: Path = Path("data/models/kokoro")
    kokoro_voice: str = "af_heart"
    # Model artifacts are installed explicitly; normal server starts must never fetch them.
    allow_model_downloads: bool = False
    uploads_dir: Path = Path("data/uploads")
    max_resume_bytes: int = 8_000_000
    vision_sample_interval_ms: int = 750
    vad_threshold: float = 0.52
    # 0 = noise-resistant, 1 = quickest pickup. Exposed as KEC_VAD_SENSITIVITY.
    vad_sensitivity: float = 0.42
    vad_onset_ms: int = 160
    vad_min_speech_ms: int = 240
    vad_silence_ms: int = 1100
    vad_max_utterance_ms: int = 45_000
    audio_preroll_ms: int = 320
    speaker_model: str = "speechbrain/spkrec-ecapa-voxceleb"


settings = Settings()
