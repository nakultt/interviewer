import asyncio
import json
from collections import deque
from contextlib import suppress
from typing import Any

import numpy as np
from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

from .agent import (
    AutonomousInterviewAgent,
    add_turn,
    complete_record,
    track_question_competency,
)
from .audio import SpeechEnhancer
from .config import Settings
from .domain import InterviewRecord
from .repository import InterviewRepository
from .stt import WhisperService
from .vad import HybridEndpointDetector, SileroProbability
from .vision import VisionMonitor
from .voice import VoiceGuard

AUDIO_CHANNEL = 1
VIDEO_CHANNEL = 2


class RealtimeInterviewStream:
    """Full-duplex local interview sideband over one browser WebSocket."""

    def __init__(
        self,
        websocket: WebSocket,
        record: InterviewRecord,
        settings: Settings,
        repository: InterviewRepository,
        agent: AutonomousInterviewAgent,
        whisper: WhisperService,
        voice_guard: VoiceGuard,
        vision: VisionMonitor,
    ) -> None:
        self.websocket = websocket
        self.record = record
        self.repository = repository
        self.agent = agent
        self.whisper = whisper
        self.voice_guard = voice_guard
        self.vision = vision
        self.silero = SileroProbability()
        self.enhancer = SpeechEnhancer()
        self.endpoint = HybridEndpointDetector(
            threshold=settings.vad_threshold,
            onset_ms=settings.vad_onset_ms,
            min_speech_ms=settings.vad_min_speech_ms,
            silence_ms=settings.vad_silence_ms,
            max_utterance_ms=settings.vad_max_utterance_ms,
            sensitivity=settings.vad_sensitivity,
        )
        preroll_frames = max(2, settings.audio_preroll_ms // self.endpoint.FRAME_MS)
        self.preroll: deque[np.ndarray] = deque(maxlen=preroll_frames)
        self.pending = np.empty(0, dtype=np.float32)
        self.utterance: list[np.ndarray] = []
        self.turn_queue: asyncio.Queue[tuple[np.ndarray, int] | None] = asyncio.Queue(maxsize=2)
        self.send_lock = asyncio.Lock()
        self.assistant_speaking = False
        self.assistant_turn_pending = False
        self.audio_generation = 0
        self.auto_finish_started = False
        self.playback_cooldown_ms = 0
        self.vision_busy = False
        self.frames_since_save = 0
        self.closed = False
        self.accepting_audio = True
        self.ready_sent = False

    async def run(self) -> None:
        await self.websocket.accept()
        turn_worker = asyncio.create_task(self._turn_worker())
        await self._send({"type": "connected", "state": "connecting"})
        try:
            while True:
                message = await self.websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                payload = message.get("bytes")
                if payload:
                    await self._binary(payload)
                elif message.get("text"):
                    await self._command(message["text"])
        except WebSocketDisconnect:
            pass
        finally:
            self.closed = True
            self.accepting_audio = False
            turn_worker.cancel()
            with suppress(asyncio.CancelledError):
                await turn_worker

    async def _binary(self, payload: bytes) -> None:
        if not payload:
            return
        channel, content = payload[0], payload[1:]
        if channel == AUDIO_CHANNEL:
            if len(content) < 4:
                return
            generation = int.from_bytes(content[:4], byteorder="little", signed=False)
            await self._audio(content[4:], generation)
        elif channel == VIDEO_CHANNEL and not self.vision_busy:
            self.vision_busy = True
            asyncio.create_task(self._vision(content))

    async def _audio(self, pcm: bytes, generation: int) -> None:
        if (
            len(pcm) < 2
            or not self.accepting_audio
            or self.assistant_turn_pending
            or generation != self.audio_generation
        ):
            return
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        self.pending = np.concatenate((self.pending, samples))
        while self.pending.size >= 512:
            frame, self.pending = self.pending[:512], self.pending[512:]
            await self._vad_frame(frame, generation)

    async def _vad_frame(self, frame: np.ndarray, generation: int) -> None:
        # Browser speech synthesis and microphone capture share the same output device. Without
        # a clean output-reference signal, playback echo is indistinguishable from a real user
        # interruption. Ignore it completely and briefly discard the acoustic tail so the AI
        # cannot cancel itself or transcribe its own question.
        if self.assistant_speaking:
            return
        if self.playback_cooldown_ms > 0:
            self.playback_cooldown_ms = max(
                0, self.playback_cooldown_ms - self.endpoint.FRAME_MS
            )
            return
        probability = self.silero(frame)
        rms = float(np.sqrt(np.mean(frame * frame)))
        event = self.endpoint.accept(probability, rms)
        if event.speech_started:
            self.utterance = [*self.preroll, frame.copy()]
            self.preroll.clear()
            await self._send({"type": "speech_started", "state": "listening"})
        elif self.endpoint.speaking:
            self.utterance.append(frame.copy())
        else:
            self.preroll.append(frame.copy())
        if event.speech_ended and self.utterance:
            utterance = np.concatenate(self.utterance)
            self.utterance = []
            self.preroll.clear()
            self.silero.reset()
            if self.turn_queue.full():
                await self._send({"type": "busy", "state": "thinking"})
            else:
                await self.turn_queue.put((utterance, generation))
                await self._send({"type": "speech_ended", "state": "transcribing"})

    async def _turn_worker(self) -> None:
        while True:
            queued_turn = await self.turn_queue.get()
            if queued_turn is None:
                return
            samples, generation = queued_turn
            if not self._can_process_turn(generation):
                continue
            try:
                await self._process_turn(samples, generation)
            except Exception as error:
                await self._send({"type": "error", "state": "listening", "message": str(error)})

    async def _process_turn(self, samples: np.ndarray, generation: int) -> None:
        samples = await asyncio.to_thread(self.enhancer.enhance, samples)
        if not self._can_process_turn(generation):
            return
        prompt = (
            f"Technical interview for {self.record.role_title}. "
            f"Expected terms: {', '.join(self.record.resume_skills[:16])}."
        )
        voice_task = asyncio.create_task(self.voice_guard.analyze_samples(self.record.id, samples))
        transcript = await self.whisper.transcribe_samples(samples, prompt)
        if not self._can_process_turn(generation):
            voice_task.cancel()
            return
        if len(transcript) < 2:
            voice_task.cancel()
            await self._send(
                {
                    "type": "no_transcript",
                    "state": "listening",
                    "message": "I did not catch that. Please continue naturally.",
                }
            )
            return
        voice_result = await voice_task
        if bool(voice_result.get("additional_voice_likely")):
            self.record.integrity.additional_voice_events += 1
            if "additional_voice_review" not in self.record.integrity.flags:
                self.record.integrity.flags.append("additional_voice_review")
        add_turn(self.record, "candidate", transcript, int(samples.size / 16))
        await self.repository.save(self.record)
        await self._send(
            {
                "type": "user_transcript",
                "state": "thinking",
                "transcript": transcript,
                "voice": voice_result,
                "record": self.record.model_dump(mode="json"),
            }
        )
        decision = await self.agent.next_question(self.record, transcript)
        if not self._can_process_turn(generation):
            return
        self.record.conversation_phase = decision.phase
        if decision.should_finish:
            self.accepting_audio = False
            complete_record(self.record)
            report = await self.agent.evaluate(self.record)
            await self.repository.save(self.record)
            await self.repository.save_report(report)
            await self._send(
                {
                    "type": "interview_complete",
                    "state": "complete",
                    "record": self.record.model_dump(mode="json"),
                    "report": report.model_dump(mode="json"),
                    "closing_message": decision.question,
                }
            )
            return
        add_turn(self.record, "interviewer", decision.question)
        track_question_competency(self.record, decision)
        await self.repository.save(self.record)
        self.assistant_turn_pending = True
        await self._send(
            {
                "type": "assistant_question",
                "state": "assistant_speaking",
                "question": decision.question,
                "record": self.record.model_dump(mode="json"),
            }
        )

    async def _vision(self, jpeg: bytes) -> None:
        try:
            summary = await asyncio.to_thread(self.vision.analyze, self.record.id, jpeg)
            self.record.integrity = summary
            self.frames_since_save += 1
            if self.frames_since_save >= 12:
                self.frames_since_save = 0
                await self.repository.save(self.record)
            await self._send(
                {
                    "type": "integrity",
                    "integrity": summary.model_dump(mode="json"),
                }
            )
            if summary.multiple_people_events >= 3 and not self.auto_finish_started:
                await self._finish_for_multiple_people()
        finally:
            self.vision_busy = False

    async def _finish_for_multiple_people(self) -> None:
        self.auto_finish_started = True
        self.accepting_audio = False
        self.assistant_turn_pending = True
        self._discard_audio_boundary()
        complete_record(self.record)
        await self.repository.save(self.record)
        report = await self.agent.evaluate(self.record)
        await self.repository.save_report(report)
        await self._send(
            {
                "type": "interview_complete",
                "state": "complete",
                "completion_reason": "multiple_people_limit",
                "record": self.record.model_dump(mode="json"),
                "report": report.model_dump(mode="json"),
                "closing_message": (
                    "The interview has ended because multiple people were detected three times."
                ),
            }
        )

    async def _command(self, raw: str) -> None:
        try:
            command: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError:
            return
        if command.get("type") == "assistant_playback":
            raw_generation = command.get("audio_generation")
            if not isinstance(raw_generation, (int, float, str)):
                return
            with suppress(TypeError, ValueError):
                self._set_assistant_playback(
                    bool(command.get("speaking")), int(raw_generation)
                )
        elif command.get("type") == "vad_sensitivity":
            with suppress(TypeError, ValueError):
                self.endpoint.set_sensitivity(float(command.get("value", 0.42)))
        elif command.get("type") == "client_ready" and not self.ready_sent:
            self.ready_sent = True
            self.assistant_turn_pending = True
            await self._send(
                {
                    "type": "ready",
                    "state": "assistant_speaking",
                    "record": self.record.model_dump(mode="json"),
                }
            )
        elif command.get("type") == "ping":
            await self._send({"type": "pong"})

    def _set_assistant_playback(self, speaking: bool, generation: int) -> None:
        if generation < self.audio_generation:
            return
        if speaking:
            self.audio_generation = generation
            self.assistant_speaking = True
            self.assistant_turn_pending = True
            self._discard_audio_boundary()
            self.playback_cooldown_ms = 0
            return
        if generation != self.audio_generation:
            return
        self.assistant_speaking = speaking
        self.assistant_turn_pending = False
        self._discard_audio_boundary()
        self.playback_cooldown_ms = 320

    def _can_process_turn(self, generation: int) -> bool:
        return (
            self.accepting_audio
            and not self.assistant_turn_pending
            and not self.assistant_speaking
            and generation == self.audio_generation
        )

    def _discard_audio_boundary(self) -> None:
        self.pending = np.empty(0, dtype=np.float32)
        self.utterance.clear()
        self.preroll.clear()
        self.endpoint.reset()
        self.silero.reset()
        while not self.turn_queue.empty():
            self.turn_queue.get_nowait()

    async def _send(self, payload: dict[str, Any]) -> None:
        if self.closed:
            return
        async with self.send_lock:
            await self.websocket.send_json(payload)
