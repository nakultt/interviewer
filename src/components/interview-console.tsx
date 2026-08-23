"use client";

import { FormEvent, useEffect, useRef, useState } from "react";

const API = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://127.0.0.1:8765";

type Role = { id: string; title: string; summary: string; competencies: string[]; color: string };
type Turn = { id: string; role: "interviewer" | "candidate"; text: string };
type Integrity = {
  frames_analyzed: number; face_visible_ratio: number; center_gaze_ratio: number;
  posture_stability: number; current_face_count: number; current_face_visible: boolean;
  current_gaze_center: boolean; current_posture_stability: number;
  multiple_people_events: number; additional_voice_events: number; flags: string[];
};
type Interview = {
  id: string; candidate_name: string; role_title: string; difficulty: string; target_questions: number;
  resume_filename: string; resume_skills: string[]; turns: Turn[]; integrity: Integrity; status: string;
};
type Report = {
  overall_score: number; summary: string; strengths: string[]; improvements: string[]; action_plan: string[];
  scores: Array<{ competency: string; score: number; evidence: string; improvement: string }>;
  integrity: Integrity;
};
type Health = { mode: string; model_detail?: string; services: Record<string, string> };
type StreamState = "connecting" | "assistant_speaking" | "listening" | "transcribing" | "thinking";

export function InterviewConsole() {
  const [screen, setScreen] = useState<"setup" | "interview" | "report">("setup");
  const [roles, setRoles] = useState<Role[]>([]);
  const [health, setHealth] = useState<Health | null>(null);
  const [selectedRole, setSelectedRole] = useState("ai-engineer");
  const [difficulty, setDifficulty] = useState("mid");
  const [candidateName, setCandidateName] = useState("");
  const [resume, setResume] = useState<File | null>(null);
  const [interview, setInterview] = useState<Interview | null>(null);
  const [report, setReport] = useState<Report | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [streamState, setStreamState] = useState<StreamState>("connecting");
  const [voiceDetected, setVoiceDetected] = useState(false);
  const [vadSensitivity, setVadSensitivity] = useState(42);
  const [integrity, setIntegrity] = useState<Integrity | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const workletRef = useRef<AudioWorkletNode | null>(null);
  const ttsAudioRef = useRef<HTMLAudioElement | null>(null);
  const ttsAbortRef = useRef<AbortController | null>(null);
  const playbackActiveRef = useRef(false);
  const audioGenerationRef = useRef(0);
  const ttsRequestRef = useRef(0);
  const openingSpokenRef = useRef(false);
  const interviewerUnavailable = health?.services.interviewer === "unavailable";

  useEffect(() => {
    void Promise.all([
      fetch(`${API}/v1/roles`).then((response) => response.json()),
      fetch(`${API}/health`).then((response) => response.json()),
    ]).then(([roleData, healthData]) => {
      setRoles(roleData);
      setHealth(healthData);
    }).catch(() => setError("Python backend is not reachable on port 8765."));
  }, []);

  useEffect(() => {
    if (screen !== "interview" || !interview) return;
    let visionTimer: ReturnType<typeof setInterval> | undefined;
    let socket: WebSocket | null = null;
    let active = true;
    void navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 960 }, height: { ideal: 540 }, frameRate: { ideal: 20 } },
      audio: {
        echoCancellation: { ideal: true }, noiseSuppression: { ideal: true },
        autoGainControl: { ideal: false }, channelCount: { ideal: 1 }, sampleRate: { ideal: 16_000 },
      },
    }).then((stream) => {
      if (!active) return;
      mediaStreamRef.current = stream;
      if (videoRef.current) videoRef.current.srcObject = stream;
      const socketUrl = `${API.replace(/^http/, "ws")}/v1/interviews/${interview.id}/stream`;
      socket = new WebSocket(socketUrl);
      socket.binaryType = "arraybuffer";
      socketRef.current = socket;
      socket.onopen = () => {
        if (!socket) return;
        void startPcmStream(stream, socket).then(() => {
          socket?.send(JSON.stringify({ type: "client_ready" }));
          visionTimer = setInterval(() => void sendVisionFrame(), 250);
        }).catch(() => {
          setError("Live PCM capture could not start. Check browser audio permissions.");
        });
      };
      socket.onmessage = (message) => {
        const event = JSON.parse(String(message.data));
        if (event.record) setInterview(event.record);
        if (event.integrity) setIntegrity(event.integrity);
        if (event.type === "ready") {
          setVoiceDetected(false);
          setStreamState("assistant_speaking");
          if (!openingSpokenRef.current) {
            openingSpokenRef.current = true;
            const record = event.record as Interview;
            speak(record.turns.at(-1)?.text ?? "Let's begin.");
          }
        } else if (event.type === "speech_started") {
          setVoiceDetected(true);
          setStreamState("listening");
        } else if (event.type === "barge_in") {
          cancelAssistantSpeech();
          setStreamState("listening");
        } else if (event.type === "speech_ended") {
          setVoiceDetected(false);
          setStreamState("transcribing");
        } else if (event.type === "user_transcript") {
          setVoiceDetected(false);
          setStreamState("thinking");
        } else if (event.type === "assistant_question") {
          setVoiceDetected(false);
          setStreamState("assistant_speaking");
          speak(event.question);
        } else if (event.type === "interview_complete") {
          setVoiceDetected(false);
          setStreamState("assistant_speaking");
          speak(event.closing_message ?? "Thank you. I’ll prepare your feedback now.", () => {
            setReport(event.report); setScreen("report");
          });
        } else if (event.type === "no_transcript") {
          setVoiceDetected(false);
          setStreamState("listening");
        } else if (event.type === "error") {
          setVoiceDetected(false);
          setError(event.message); setStreamState("listening");
        }
      };
    }).catch(() => setError("Camera/microphone permission is required for a monitored interview."));
    return () => {
      active = false;
      if (visionTimer) clearInterval(visionTimer);
      workletRef.current?.disconnect();
      workletRef.current = null;
      void audioContextRef.current?.close();
      audioContextRef.current = null;
      socket?.close();
      socketRef.current = null;
      mediaStreamRef.current?.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
      cancelAssistantSpeech();
    };
  }, [screen, interview?.id]);

  async function createInterview(event: FormEvent) {
    event.preventDefault();
    if (!resume) return setError("Upload a resume before starting.");
    setBusy(true); setError("");
    const form = new FormData();
    form.append("resume", resume);
    form.append("role_id", selectedRole);
    form.append("candidate_name", candidateName || "Candidate");
    form.append("difficulty", difficulty);
    form.append("target_questions", "6");
    try {
      const response = await fetch(`${API}/v1/interviews`, { method: "POST", body: form });
      if (!response.ok) throw new Error(await apiError(response));
      const created = await response.json() as Interview;
      setInterview(created); setIntegrity(created.integrity); setScreen("interview");
    } catch (reason) { setError(reason instanceof Error ? reason.message : "Could not start interview"); }
    finally { setBusy(false); }
  }

  async function finishInterview() {
    if (!interview) return;
    setBusy(true); setError(""); cancelAssistantSpeech();
    socketRef.current?.close();
    socketRef.current = null;
    mediaStreamRef.current?.getTracks().forEach((track) => track.stop());
    mediaStreamRef.current = null;
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 35_000);
    try {
      const response = await fetch(`${API}/v1/interviews/${interview.id}/finish`, {
        method: "POST", signal: controller.signal,
      });
      if (!response.ok) throw new Error(await apiError(response));
      setReport(await response.json()); setScreen("report");
    } catch (reason) {
      setError(reason instanceof DOMException && reason.name === "AbortError"
        ? "Report generation timed out. Check that the Python backend is running."
        : reason instanceof Error ? reason.message : "Report generation failed");
    } finally { window.clearTimeout(timeout); setBusy(false); }
  }

  async function startPcmStream(stream: MediaStream, socket: WebSocket) {
    const context = new AudioContext({ sampleRate: 16_000, latencyHint: "interactive" });
    audioContextRef.current = context;
    await context.audioWorklet.addModule("/pcm-worklet.js");
    const source = context.createMediaStreamSource(new MediaStream(stream.getAudioTracks()));
    const worklet = new AudioWorkletNode(context, "pcm-capture");
    const silentOutput = context.createGain();
    silentOutput.gain.value = 0;
    source.connect(worklet); worklet.connect(silentOutput); silentOutput.connect(context.destination);
    worklet.port.onmessage = (event: MessageEvent<Float32Array>) => {
      if (playbackActiveRef.current || socket.readyState !== WebSocket.OPEN || socket.bufferedAmount > 1_000_000) return;
      const samples = event.data;
      const packet = new Uint8Array(5 + samples.length * 2);
      packet[0] = 1;
      new DataView(packet.buffer).setUint32(1, audioGenerationRef.current, true);
      const view = new DataView(packet.buffer);
      for (let index = 0; index < samples.length; index += 1) {
        const sample = Math.max(-1, Math.min(1, samples[index]));
        view.setInt16(5 + index * 2, sample < 0 ? sample * 32768 : sample * 32767, true);
      }
      socket.send(packet);
    };
    workletRef.current = worklet;
    await context.resume();
  }

  async function sendVisionFrame() {
    const video = videoRef.current, canvas = canvasRef.current;
    const socket = socketRef.current;
    if (!socket || socket.readyState !== WebSocket.OPEN || socket.bufferedAmount > 1_000_000) return;
    if (!video || !canvas || video.readyState < 2) return;
    canvas.width = 640; canvas.height = 360;
    canvas.getContext("2d")?.drawImage(video, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/jpeg", .72));
    if (!blob) return;
    const frame = new Uint8Array(await blob.arrayBuffer());
    const packet = new Uint8Array(frame.length + 1); packet[0] = 2; packet.set(frame, 1);
    socket.send(packet);
  }

  function speak(text: string, onComplete?: () => void) {
    cancelAssistantSpeech();
    const requestId = ++ttsRequestRef.current;
    audioGenerationRef.current += 1;
    playbackActiveRef.current = true;
    sendPlaybackState(true);
    const controller = new AbortController();
    ttsAbortRef.current = controller;
    let finished = false;
    const finish = () => {
      if (finished || requestId !== ttsRequestRef.current) return;
      finished = true;
      ttsAbortRef.current = null;
      ttsAudioRef.current = null;
      playbackActiveRef.current = false;
      sendPlaybackState(false);
      setStreamState("listening");
      onComplete?.();
    };
    void fetch(`${API}/v1/tts`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ text }),
      signal: controller.signal,
    }).then(async response => {
      if (!response.ok) throw new Error(await apiError(response));
      const source = URL.createObjectURL(await response.blob());
      if (requestId !== ttsRequestRef.current || controller.signal.aborted) {
        URL.revokeObjectURL(source);
        return;
      }
      const audio = new Audio(source);
      ttsAudioRef.current = audio;
      audio.onplay = () => setStreamState("assistant_speaking");
      audio.onended = () => { URL.revokeObjectURL(source); finish(); };
      audio.onerror = () => { URL.revokeObjectURL(source); finish(); };
      return audio.play();
    }).catch(reason => {
      if (controller.signal.aborted || requestId !== ttsRequestRef.current) return;
      setError(reason instanceof Error ? `Kokoro TTS failed: ${reason.message}` : "Kokoro TTS failed.");
      finish();
    });
  }

  function cancelAssistantSpeech() {
    ttsRequestRef.current += 1;
    ttsAbortRef.current?.abort();
    ttsAbortRef.current = null;
    ttsAudioRef.current?.pause();
    ttsAudioRef.current?.removeAttribute("src");
    ttsAudioRef.current = null;
    playbackActiveRef.current = false;
    sendPlaybackState(false);
  }

  function sendPlaybackState(speaking: boolean) {
    const socket = socketRef.current;
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({
        type: "assistant_playback", speaking, audio_generation: audioGenerationRef.current,
      }));
    }
  }

  function changeSensitivity(value: number) {
    setVadSensitivity(value);
    const socket = socketRef.current;
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: "vad_sensitivity", value: value / 100 }));
    }
  }

  if (screen === "report" && report) return <ReportScreen report={report} interview={interview} />;
  const peopleStrikes = Math.min(3, integrity?.multiple_people_events ?? 0);
  if (screen === "interview" && interview) return (
    <div className="interviewShell">
      <header className="sessionNav"><div className="brandMark">K</div><strong>{interview.role_title}</strong><span>{interview.difficulty} · local AI</span><button onClick={() => void finishInterview()} disabled={busy}>{busy ? "Finishing…" : "Finish interview"}</button></header>
      <div className="interviewGrid">
        <section className="cameraStage">
          <video ref={videoRef} autoPlay muted playsInline />
          <canvas ref={canvasRef} hidden />
          <div className="cameraOverlay"><span className="recDot" /> PRIVATE SESSION <b>{integrity?.frames_analyzed ?? 0} frames analyzed</b></div>
          <div className="monitorRail">
            <Monitor label="Face" value={integrity?.current_face_visible ? "Detected" : "Missing"} ok={integrity?.current_face_visible ?? false} />
            <Monitor label="Gaze zone" value={integrity?.current_gaze_center ? "Centered" : "Away"} ok={integrity?.current_gaze_center ?? false} />
            <Monitor label="People" value={peopleLabel(integrity?.current_face_count ?? 0, peopleStrikes)} ok={(integrity?.current_face_count ?? 0) <= 1 && peopleStrikes === 0} />
            <Monitor label="Voice" value={!voiceDetected ? "Not detected" : (integrity?.additional_voice_events ?? 0) ? "Review" : "Consistent"} ok={voiceDetected && !(integrity?.additional_voice_events)} idle={!voiceDetected} />
          </div>
          {peopleStrikes > 0 ? <div className="peopleWarning"><strong>Multiple people detected · {peopleStrikes}/3</strong><span>Strikes are spaced 5 seconds apart; the interview ends at 3/3.</span></div> : null}
          <div className={`recordOrb ${streamState === "listening" ? "active" : ""}`}><span>{streamStateLabel(streamState)}</span></div>
          <div className="handsFree"><i /> Hands-free · listen, then answer naturally</div>
          <p className="privacyNote">The microphone listens after each question, with echo cancellation and local noise cleanup active. Silence and raw audio are never stored.</p>
        </section>
        <aside className="conversationPanel">
          <div className="conversationHead"><div><small>ADAPTIVE INTERVIEW</small><h2>Conversation</h2></div><span>{Math.min(interview.turns.filter(t => t.role === "candidate").length + 1, interview.target_questions)}/{interview.target_questions}</span></div>
          <div className="conversationList">
            {interview.turns.map((turn) => <article key={turn.id} className={turn.role}><label>{turn.role === "candidate" ? "YOU" : "INTERVIEWER"}</label><p>{turn.text}</p></article>)}
            {streamState === "transcribing" ? <div className="thinking">Whisper is finalizing your answer…</div> : null}
            {streamState === "thinking" ? <div className="thinking">Qwen is considering your answer and choosing a natural follow-up…</div> : null}
          </div>
          <div className="liveTurnStatus"><i className={streamState} /><span>{streamStateLabel(streamState)}</span><b>No buttons needed</b></div>
          <div className="sensitivityControl"><label htmlFor="vad-sensitivity">Voice pickup</label><input id="vad-sensitivity" type="range" min="0" max="100" value={vadSensitivity} onChange={event => changeSensitivity(Number(event.target.value))} /><output>{vadSensitivity}%</output><small>Lower rejects background noise; higher picks up softer speech.</small></div>
          {error ? <p className="formError">{error}</p> : null}
        </aside>
      </div>
    </div>
  );

  return (
    <div className="setupShell">
      <header className="setupHeader"><div className="brand"><span>K</span> KEC Interview Lab</div><div className="localStatus"><i /> Python + Qwen · local inference</div></header>
      <main className="setupMain">
        <div className="setupIntro"><p className="eyebrow">PRIVATE AI INTERVIEW COACH</p><h1>Build the story behind<br /><em>your experience.</em></h1><p>Upload your resume, choose your target role, and complete a personalized technical interview entirely through your local AI stack.</p></div>
        <form className="setupCard" onSubmit={(event) => void createInterview(event)}>
          <div className="stepLabel"><span>01</span><div><strong>Your resume</strong><small>PDF, DOCX, TXT · maximum 8 MB</small></div></div>
          <label className={`resumeDrop ${resume ? "hasFile" : ""}`}><input type="file" accept=".pdf,.docx,.txt,.md" onChange={event => setResume(event.target.files?.[0] ?? null)} /><b>{resume ? resume.name : "Drop or choose your resume"}</b><span>{resume ? `${(resume.size / 1024).toFixed(0)} KB ready` : "The AI uses it to personalize questions"}</span></label>
          <div className="stepLabel"><span>02</span><div><strong>Target role</strong><small>Choose the interview track</small></div></div>
          <div className="roleGrid">{roles.map(role => <button type="button" key={role.id} className={`roleCard ${selectedRole === role.id ? "selected" : ""}`} onClick={() => setSelectedRole(role.id)}><i className={role.color}>{role.title.slice(0, 2).toUpperCase()}</i><b>{role.title}</b><small>{role.summary}</small></button>)}</div>
          <div className="setupRow"><label><span>Your name</span><input value={candidateName} onChange={event => setCandidateName(event.target.value)} placeholder="Candidate" /></label><label><span>Difficulty</span><select value={difficulty} onChange={event => setDifficulty(event.target.value)}><option value="junior">Junior</option><option value="mid">Mid-level</option><option value="senior">Senior</option></select></label></div>
          {interviewerUnavailable ? <p className="formError">LM Studio is unavailable: {health?.model_detail ?? "check the local server and loaded model"}.</p> : error ? <p className="formError">{error}</p> : null}
          <button className="launchButton" disabled={busy || !resume || interviewerUnavailable}>{busy ? "Building your interview…" : "Start personalized interview →"}</button>
          <div className="serviceLine">{health ? Object.entries(health.services).map(([name, value]) => <span key={name} className={value === "ready" || value === "lm-studio" ? "ready" : "warn"}>{name.replaceAll("_", " ")}: {value}</span>) : "Checking local services…"}</div>
        </form>
      </main>
    </div>
  );
}

function ReportScreen({ report, interview }: { report: Report; interview: Interview | null }) {
  return <div className="reportShell"><header className="reportHero"><p className="eyebrow">INTERVIEW COMPLETE</p><h1>Your next level is<br /><em>specific.</em></h1><div className="scoreRing"><strong>{report.overall_score}</strong><span>/ 100</span></div><p>{report.summary}</p></header><main className="reportMain"><section className="scoreCards">{report.scores.map(score => <article key={score.competency}><header><b>{score.competency}</b><strong>{score.score}</strong></header><div className="scoreBar"><i style={{ width: `${score.score}%` }} /></div><p>{score.evidence}</p><small>{score.improvement}</small></article>)}</section><div className="reportColumns"><section><h2>What worked</h2>{report.strengths.map(item => <p key={item}>✓ {item}</p>)}</section><section><h2>Improve next</h2>{report.improvements.map(item => <p key={item}>↗ {item}</p>)}</section></div><section className="actionPlan"><small>YOUR 7-DAY PLAN</small><h2>Turn feedback into evidence.</h2><ol>{report.action_plan.map(item => <li key={item}>{item}</li>)}</ol></section><footer>Report for {interview?.candidate_name} · Integrity signals are listed separately and did not affect technical scores.</footer></main></div>;
}

function Monitor({ label, value, ok, idle = false }: { label: string; value: string; ok: boolean; idle?: boolean }) { return <div><i className={idle ? "idle" : ok ? "good" : "review"} /><span>{label}</span><b>{value}</b></div>; }
function peopleLabel(count: number, strikes: number) { const people = count === 0 ? "None" : count === 1 ? "One" : `${count} people`; return strikes ? `${people} · ${strikes}/3` : people; }
function streamStateLabel(state: StreamState) { return ({ connecting: "Connecting…", assistant_speaking: "Interviewer speaking", listening: "Listening…", transcribing: "Understanding…", thinking: "Preparing follow-up…" })[state]; }
async function apiError(response: Response) { try { const data = await response.json(); return data.detail ?? `Request failed (${response.status})`; } catch { return `Request failed (${response.status})`; } }
