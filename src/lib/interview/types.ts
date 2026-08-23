export const SESSION_STATES = [
  "IDLE",
  "PREPARING_QUESTION",
  "AI_THINKING",
  "AI_SPEAKING",
  "USER_SPEAKING",
  "ENDPOINT_PENDING",
  "EVALUATING",
  "PAUSED",
  "COMPLETED",
  "ERROR_RECOVERY",
] as const;

export type SessionState = (typeof SESSION_STATES)[number];

export type EventSource =
  | "ui"
  | "vad"
  | "barge_in_controller"
  | "stt"
  | "interviewer"
  | "system";

export interface InterviewEvent<T = Record<string, unknown>> {
  event_id: string;
  session_id: string;
  turn_id: number;
  timestamp_monotonic_ms: number;
  timestamp_wallclock: string;
  source: EventSource;
  type: string;
  payload: T;
}

export interface ActiveOutput {
  generation_id: string;
  playback_id: string;
  spoken_text: string;
  generated_text: string;
}

export interface SessionSnapshot {
  state: SessionState;
  session_id: string;
  turn_id: number;
  active_output: ActiveOutput | null;
  cancelled_generation_ids: string[];
  last_error: string | null;
}
