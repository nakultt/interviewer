from dataclasses import dataclass

import numpy as np
import torch
from silero_vad import load_silero_vad  # type: ignore[import-untyped]


@dataclass(slots=True)
class VadDecision:
    speech_started: bool = False
    speech_ended: bool = False
    echo_likely: bool = False


class FastVadGate:
    """Deterministic onset gate; Silero can confirm speech downstream.

    This layer intentionally has no model or network dependency, so it can own
    playback cancellation in the barge-in path.
    """

    def __init__(
        self,
        threshold: float = 0.085,
        onset_ms: int = 100,
        endpoint_ms: int = 650,
        echo_ratio: float = 0.7,
    ) -> None:
        self.threshold = threshold
        self.onset_ms = onset_ms
        self.endpoint_ms = endpoint_ms
        self.echo_ratio = echo_ratio
        self.voiced_ms = 0
        self.silent_ms = 0
        self.speaking = False

    def process(self, rms: float, duration_ms: int, output_reference_rms: float = 0) -> VadDecision:
        echo_likely = output_reference_rms > 0 and rms <= output_reference_rms * self.echo_ratio
        voiced = rms >= self.threshold and not echo_likely
        if voiced:
            self.voiced_ms += duration_ms
            self.silent_ms = 0
            if not self.speaking and self.voiced_ms >= self.onset_ms:
                self.speaking = True
                return VadDecision(speech_started=True, echo_likely=echo_likely)
        else:
            self.voiced_ms = 0
            if self.speaking:
                self.silent_ms += duration_ms
                if self.silent_ms >= self.endpoint_ms:
                    self.speaking = False
                    self.silent_ms = 0
                    return VadDecision(speech_ended=True, echo_likely=echo_likely)
        return VadDecision(echo_likely=echo_likely)


class SileroProbability:
    """Stateful ONNX Silero inference for fixed 512-sample, 16 kHz frames."""

    def __init__(self) -> None:
        self._model = load_silero_vad(onnx=True)

    def __call__(self, samples: np.ndarray) -> float:
        if samples.size != 512:
            raise ValueError("Silero requires exactly 512 samples at 16 kHz")
        tensor = torch.from_numpy(samples.astype(np.float32, copy=False))
        return float(self._model(tensor, 16_000).item())

    def reset(self) -> None:
        self._model.reset_states()


@dataclass(slots=True)
class EndpointEvent:
    speech_started: bool = False
    speech_ended: bool = False
    probability: float = 0.0


class HybridEndpointDetector:
    """Silero endpointing with an energy-assisted onset and trailing-silence debounce."""

    FRAME_MS = 32

    def __init__(
        self,
        threshold: float = 0.52,
        onset_ms: int = 160,
        min_speech_ms: int = 240,
        silence_ms: int = 720,
        max_utterance_ms: int = 45_000,
        sensitivity: float = 0.42,
    ) -> None:
        self.threshold = threshold
        self.onset_ms = max(self.FRAME_MS, onset_ms)
        self.min_speech_ms = min_speech_ms
        self.silence_ms = silence_ms
        self.max_utterance_ms = max_utterance_ms
        self.sensitivity = min(1.0, max(0.0, sensitivity))
        self.speaking = False
        self._positive_ms = 0
        self._speech_ms = 0
        self._silence_ms = 0

    def accept(
        self, probability: float, rms: float, *, immediate_onset: bool = False
    ) -> EndpointEvent:
        probability_threshold = max(0.30, self.threshold - 0.16 * self.sensitivity)
        strong_probability = max(0.22, probability_threshold - 0.18)
        strong_rms = max(0.012, 0.032 - 0.018 * self.sensitivity)
        voiced = probability >= probability_threshold
        strong_onset = probability >= strong_probability and rms >= strong_rms
        if not self.speaking:
            self._positive_ms = self._positive_ms + self.FRAME_MS if voiced or strong_onset else 0
            if immediate_onset or self._positive_ms >= self.onset_ms:
                self.speaking = True
                self._speech_ms = self._positive_ms
                self._silence_ms = 0
                return EndpointEvent(speech_started=True, probability=probability)
            return EndpointEvent(probability=probability)

        self._speech_ms += self.FRAME_MS
        if voiced or (probability >= 0.25 and rms >= 0.015):
            self._silence_ms = 0
        else:
            self._silence_ms += self.FRAME_MS
        finished = (
            self._speech_ms >= self.min_speech_ms and self._silence_ms >= self.silence_ms
        ) or self._speech_ms >= self.max_utterance_ms
        if finished:
            self.reset()
            return EndpointEvent(speech_ended=True, probability=probability)
        return EndpointEvent(probability=probability)

    def set_sensitivity(self, value: float) -> None:
        self.sensitivity = min(1.0, max(0.0, value))

    def reset(self) -> None:
        self.speaking = False
        self._positive_ms = 0
        self._speech_ms = 0
        self._silence_ms = 0
