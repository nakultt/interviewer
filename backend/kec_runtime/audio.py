import numpy as np


class SpeechEnhancer:
    """Local speech cleanup for completed turns before any inference model sees them.

    The browser's acoustic echo cancellation handles loudspeaker feedback in real time. This
    second stage removes DC/very-low-frequency energy, estimates stationary background noise,
    applies a conservative spectral gate, and trims non-speech edges. It deliberately avoids
    aggressive voice separation that could distort a candidate's words.
    """

    def enhance(self, samples: np.ndarray, sample_rate: int = 16_000) -> np.ndarray:
        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        if audio.size < 512:
            return audio.copy()
        audio = self._high_pass(audio)
        cleaned = self._spectral_gate(audio, sample_rate)
        cleaned = self._trim_edges(cleaned, sample_rate)
        peak = float(np.max(np.abs(cleaned))) if cleaned.size else 0.0
        if peak > 0.98:
            cleaned = cleaned * (0.98 / peak)
        return np.asarray(cleaned, dtype=np.float32)

    @staticmethod
    def _high_pass(samples: np.ndarray, coefficient: float = 0.97) -> np.ndarray:
        output = np.empty_like(samples)
        output[0] = samples[0]
        for index in range(1, samples.size):
            output[index] = coefficient * (output[index - 1] + samples[index] - samples[index - 1])
        return output

    @staticmethod
    def _spectral_gate(samples: np.ndarray, sample_rate: int) -> np.ndarray:
        frame_size, hop = 512, 128
        window = np.hanning(frame_size).astype(np.float32)
        padded = np.pad(samples, (frame_size, frame_size))
        starts = range(0, padded.size - frame_size + 1, hop)
        spectra = np.stack(
            [np.fft.rfft(padded[start : start + frame_size] * window) for start in starts]
        )
        magnitudes = np.abs(spectra)
        noise = np.percentile(magnitudes, 20, axis=0)
        gain = np.clip(1.0 - (1.45 * noise[None, :]) / (magnitudes + 1e-7), 0.12, 1.0)
        frequencies = np.fft.rfftfreq(frame_size, 1 / sample_rate)
        gain[:, (frequencies < 80) | (frequencies > 7_600)] = 0.0
        output = np.zeros(padded.size, dtype=np.float64)
        weights = np.zeros(padded.size, dtype=np.float64)
        for index, start in enumerate(starts):
            frame = np.fft.irfft(spectra[index] * gain[index], n=frame_size).real
            output[start : start + frame_size] += frame * window
            weights[start : start + frame_size] += window * window
        output /= np.maximum(weights, 1e-7)
        return np.asarray(output[frame_size:-frame_size], dtype=np.float32)

    @staticmethod
    def _trim_edges(samples: np.ndarray, sample_rate: int) -> np.ndarray:
        frame = max(1, sample_rate // 100)
        count = samples.size // frame
        if count == 0:
            return samples
        energy = np.sqrt(np.mean(samples[: count * frame].reshape(count, frame) ** 2, axis=1))
        threshold = max(0.0015, float(np.percentile(energy, 25)) * 1.8)
        active = np.flatnonzero(energy >= threshold)
        if active.size == 0:
            return samples
        padding = sample_rate // 5
        start = max(0, int(active[0]) * frame - padding)
        end = min(samples.size, (int(active[-1]) + 1) * frame + padding)
        return samples[start:end]
