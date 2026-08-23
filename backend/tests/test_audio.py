import numpy as np
from kec_runtime.audio import SpeechEnhancer


def test_speech_enhancer_reduces_stationary_background_and_keeps_speech() -> None:
    rng = np.random.default_rng(7)
    sample_rate = 16_000
    time = np.arange(sample_rate * 2) / sample_rate
    speech = 0.12 * np.sin(2 * np.pi * 220 * time)
    speech[:3_200] = 0
    speech[-3_200:] = 0
    noisy = speech + rng.normal(0, 0.018, time.size)

    cleaned = SpeechEnhancer().enhance(noisy.astype(np.float32))

    assert 0 < cleaned.size <= noisy.size
    assert float(np.sqrt(np.mean(cleaned * cleaned))) > 0.01
    assert float(np.sqrt(np.mean(cleaned[:3_200] ** 2))) < float(
        np.sqrt(np.mean(noisy[:3_200] ** 2))
    )


def test_speech_enhancer_handles_short_audio() -> None:
    samples = np.array([0.0, 0.1, -0.1], dtype=np.float32)
    assert np.array_equal(SpeechEnhancer().enhance(samples), samples)
