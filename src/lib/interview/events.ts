import type { EventSource, InterviewEvent } from "./types";

export function makeEvent<T extends Record<string, unknown>>(
  sessionId: string,
  turnId: number,
  source: EventSource,
  type: string,
  payload: T,
): InterviewEvent<T> {
  return {
    event_id: crypto.randomUUID(),
    session_id: sessionId,
    turn_id: turnId,
    timestamp_monotonic_ms: Math.round(performance.now()),
    timestamp_wallclock: new Date().toISOString(),
    source,
    type,
    payload,
  };
}
