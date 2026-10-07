// Reads GET /events (server-sent events) with fetch, sending the session
// cookie, rather than EventSource, which can't tell a 401 (signed out:
// stop) from a dropped connection (retry forever). Does what EventSource
// does: parses the stream, reconnects when it drops
// (after the server's "retry:" delay), and sends the last event id it
// received (Last-Event-ID) so the server resumes after it.

import { API_URL } from "./api";
import { clearSession, getSession } from "./session";

export type StreamStatus = "connecting" | "live" | "reconnecting";

interface Options {
  path: string;
  onMessage: (event: string, data: string) => void;
  onStatus: (status: StreamStatus) => void;
  signal: AbortSignal;
}

const sleep = (ms: number, signal: AbortSignal) =>
  new Promise<void>((resolve) => {
    const timer = window.setTimeout(resolve, ms);
    signal.addEventListener("abort", () => {
      window.clearTimeout(timer);
      resolve();
    });
  });

export async function streamEvents({ path, onMessage, onStatus, signal }: Options) {
  let lastEventId: string | null = null;
  let retryMs = 2_000;

  onStatus("connecting");

  while (!signal.aborted) {
    if (!getSession()) return;

    try {
      const response = await fetch(`${API_URL}${path}`, {
        credentials: "include",
        headers: {
          Accept: "text/event-stream",
          ...(lastEventId !== null ? { "Last-Event-ID": lastEventId } : {}),
        },
        signal,
      });

      if (response.status === 401) {
        // Signed out or expired: stop, and let the app send the user to
        // sign in.
        clearSession();
        return;
      }

      if (!response.ok || !response.body) {
        throw new Error(`HTTP ${response.status}`);
      }

      onStatus("live");

      const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
      let buffer = "";

      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += value.replace(/\r\n?/g, "\n");

        // Messages end with a blank line.
        let end: number;
        while ((end = buffer.indexOf("\n\n")) !== -1) {
          const block = buffer.slice(0, end);
          buffer = buffer.slice(end + 2);

          let event = "message";
          const data: string[] = [];

          for (const line of block.split("\n")) {
            if (!line || line.startsWith(":")) continue;

            const colon = line.indexOf(":");
            const field = colon === -1 ? line : line.slice(0, colon);
            const text = colon === -1 ? "" : line.slice(colon + 1).replace(/^ /, "");

            if (field === "event") event = text;
            else if (field === "data") data.push(text);
            else if (field === "id") lastEventId = text;
            else if (field === "retry" && /^\d+$/.test(text)) retryMs = Number(text);
          }

          if (data.length) onMessage(event, data.join("\n"));
        }
      }
    } catch {
      if (signal.aborted) return;
    }

    // The stream ended (the server closes it every few minutes, or went
    // away): reconnect, resuming after lastEventId.
    onStatus("reconnecting");
    await sleep(retryMs, signal);
  }
}
