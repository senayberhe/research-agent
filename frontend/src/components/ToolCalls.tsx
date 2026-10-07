import type { ToolCall } from "../api";
import { formatCount, formatSeconds } from "../format";
import { Card } from "./Card";
import { StatusMark } from "./status";
import { toolName } from "./taskStatus";
import type { Tone } from "./tones";

function callState(call: ToolCall): { tone: Tone; label: string } {
  if (call.success === true) return { tone: "good", label: "Succeeded" };
  if (call.success === false || call.status === "failed") return { tone: "critical", label: "Failed" };
  return { tone: "neutral", label: "Running" };
}

function summarize(calls: ToolCall[]): string {
  const succeeded = calls.filter((c) => c.success === true).length;
  const failed = calls.filter((c) => c.success === false || c.status === "failed").length;
  const running = calls.length - succeeded - failed;

  return [
    `${calls.length} call${calls.length === 1 ? "" : "s"}`,
    succeeded && `${succeeded} succeeded`,
    failed && `${failed} failed`,
    running && `${running} running`,
  ]
    .filter(Boolean)
    .join(" · ");
}

// Every tool call, oldest first; each row expands to the full query, its
// timing, its error, and the start of what the tool returned.
export function ToolCalls({ calls }: { calls: ToolCall[] }) {
  return (
    <Card title="Tool calls" subtitle={calls.length ? summarize(calls) : "None yet"}>
      {calls.length === 0 ? (
        <p className="empty-note">The agent hasn’t searched anything yet.</p>
      ) : (
        <div className="tool-calls">
          {calls.map((call) => {
            const state = callState(call);

            return (
              // Failed calls start open, so their error shows without a
              // click (the reader can still close them).
              <details
                key={call.id}
                className="tool-call"
                open={state.tone === "critical" ? true : undefined}
              >
                <summary>
                  <span className="tool-call-state">
                    <StatusMark tone={state.tone} label={state.label} />
                  </span>
                  <span className="tool-call-tool">{toolName(call.tool)}</span>
                  <span className="tool-call-query">{call.query}</span>
                  <span className="tool-call-duration">
                    {call.duration_ms !== null ? formatSeconds(call.duration_ms / 1000) : "…"}
                  </span>
                </summary>

                <div className="tool-call-body">
                  <dl>
                    <dt>Query</dt>
                    <dd>{call.query}</dd>
                    <dt>Tool</dt>
                    <dd>{toolName(call.tool)}</dd>
                    <dt>Iteration</dt>
                    <dd>{call.iteration} (the LLM call that asked for it)</dd>
                    <dt>Started</dt>
                    <dd>{new Date(call.started_at).toLocaleTimeString()}</dd>
                    <dt>Duration</dt>
                    <dd>
                      {call.duration_ms !== null
                        ? `${formatSeconds(call.duration_ms / 1000)} (including retries)`
                        : "still running"}
                    </dd>
                  </dl>

                  {call.error && (
                    <div className="tool-call-error" role="note">
                      <strong>Error:</strong> {call.error}
                    </div>
                  )}

                  {call.result_preview !== null && (
                    <>
                      <p className="tool-call-result-label">
                        Result
                        {call.result_truncated &&
                          ` (first ${formatCount(call.result_preview.length)} of ${formatCount(call.result_length)} characters)`}
                      </p>
                      <pre className="tool-call-result">{call.result_preview}</pre>
                    </>
                  )}

                  {call.result_preview === null && !call.error && call.success === null && (
                    <p className="empty-note">Waiting for the result…</p>
                  )}
                </div>
              </details>
            );
          })}
        </div>
      )}
    </Card>
  );
}
