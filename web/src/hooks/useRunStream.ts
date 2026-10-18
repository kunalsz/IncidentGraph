import { useEffect, useRef } from "react";

import { streamUrl } from "../api/client";
import type { RunSnapshot } from "../types";

const MAX_ATTEMPTS = 4;
const BASE_DELAY_MS = 1000;
const MAX_DELAY_MS = 8000;

/**
 * Follow one live run over server-sent events.
 *
 * The stream is reopened with backoff if the connection drops, rather than
 * telling the user to reload: the server pushes the current snapshot to every
 * new subscriber, so a reconnect resynchronises by itself. Callbacks live in
 * refs so a render never reopens a healthy stream.
 */
export function useRunStream(
  runId: string | null,
  onSnapshot: (snapshot: RunSnapshot) => void,
  onError: (message: string) => void,
) {
  const snapshotHandler = useRef(onSnapshot);
  const errorHandler = useRef(onError);
  snapshotHandler.current = onSnapshot;
  errorHandler.current = onError;

  useEffect(() => {
    if (!runId) return;
    let source: EventSource | null = null;
    let timer: number | null = null;
    let attempts = 0;
    let finished = false;

    const open = () => {
      source = new EventSource(streamUrl(runId));
      source.onmessage = (event: MessageEvent<string>) => {
        attempts = 0;
        try {
          const snapshot = JSON.parse(event.data) as RunSnapshot;
          snapshotHandler.current(snapshot);
          if (snapshot.status !== "running") {
            finished = true;
            source?.close();
          }
        } catch {
          /* ignore a malformed frame and wait for the next one */
        }
      };
      source.onerror = () => {
        source?.close();
        if (finished) return;
        if (attempts >= MAX_ATTEMPTS) {
          errorHandler.current("progress stream interrupted; reload to resync");
          return;
        }
        const delay = Math.min(BASE_DELAY_MS * 2 ** attempts, MAX_DELAY_MS);
        attempts += 1;
        timer = window.setTimeout(open, delay);
      };
    };

    open();
    return () => {
      finished = true;
      if (timer !== null) window.clearTimeout(timer);
      source?.close();
    };
  }, [runId]);
}
