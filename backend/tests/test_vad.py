from kec_runtime.vad import FastVadGate, HybridEndpointDetector


def test_speech_onset_is_deterministic() -> None:
    gate = FastVadGate(onset_ms=90)
    assert not gate.process(0.1, 30).speech_started
    assert not gate.process(0.1, 30).speech_started
    assert gate.process(0.1, 30).speech_started


def test_output_reference_rejects_likely_echo() -> None:
    gate = FastVadGate(onset_ms=30)
    decision = gate.process(rms=0.1, duration_ms=30, output_reference_rms=0.2)
    assert decision.echo_likely
    assert not decision.speech_started


def test_endpoint_after_trailing_silence() -> None:
    gate = FastVadGate(onset_ms=30, endpoint_ms=60)
    assert gate.process(0.1, 30).speech_started
    assert not gate.process(0.0, 30).speech_ended
    assert gate.process(0.0, 30).speech_ended


def test_hybrid_endpoint_starts_quickly_and_waits_for_stable_silence() -> None:
    detector = HybridEndpointDetector(onset_ms=64, min_speech_ms=192, silence_ms=96)
    assert not detector.accept(probability=0.8, rms=0.04).speech_started
    assert detector.accept(probability=0.8, rms=0.04).speech_started
    for _ in range(4):
        detector.accept(probability=0.8, rms=0.04)
    assert not detector.accept(probability=0.05, rms=0.002).speech_ended
    assert not detector.accept(probability=0.05, rms=0.002).speech_ended
    assert detector.accept(probability=0.05, rms=0.002).speech_ended


def test_hybrid_endpoint_rejects_a_single_noise_spike() -> None:
    detector = HybridEndpointDetector()
    event = detector.accept(probability=0.2, rms=0.08)
    assert not event.speech_started
    assert not detector.speaking


def test_hybrid_endpoint_allows_immediate_confirmed_barge_in() -> None:
    detector = HybridEndpointDetector()
    event = detector.accept(probability=0.9, rms=0.03, immediate_onset=True)
    assert event.speech_started
    assert detector.speaking


def test_sensitivity_changes_onset_threshold() -> None:
    quiet = HybridEndpointDetector(onset_ms=64, sensitivity=0.0)
    sensitive = HybridEndpointDetector(onset_ms=64, sensitivity=1.0)
    assert not quiet.accept(probability=0.40, rms=0.03).speech_started
    sensitive.accept(probability=0.40, rms=0.03)
    assert sensitive.accept(probability=0.40, rms=0.03).speech_started


def test_hybrid_endpoint_rejects_brief_voice_like_noise() -> None:
    detector = HybridEndpointDetector()
    for _ in range(4):
        assert not detector.accept(probability=0.9, rms=0.04).speech_started
    assert detector.accept(probability=0.9, rms=0.04).speech_started
