import { useEffect, useRef, useState, type FormEvent } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";

import { api } from "../api";
import { useAuth, useCan } from "../authModel";
import { Card } from "../components/Card";
import { StatusMark } from "../components/status";
import { TASK_STATUS } from "../components/taskStatus";
import { usePolling } from "../usePolling";

export function ResearchList() {
  const navigate = useNavigate();
  const location = useLocation();
  const { user } = useAuth();
  const canStart = useCan("research:create");
  const questionBox = useRef<HTMLTextAreaElement>(null);

  // "New research" (each click, even from this page): put the cursor in
  // the question box.
  useEffect(() => {
    const state = location.state as { focusQuestion?: boolean } | null;
    if (state?.focusQuestion) questionBox.current?.focus();
  }, [location.key, location.state]);
  const [question, setQuestion] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const list = usePolling((signal) => api.listResearch(signal), "research-list", 5_000);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!question.trim()) return;

    setSubmitting(true);
    setError(null);

    try {
      const task = await api.createResearch(question.trim());
      // Straight to the live view.
      navigate(`/research/${task.id}`);
    } catch (failure) {
      setError((failure as Error).message);
      setSubmitting(false);
    }
  };

  return (
    <>
      <div className="page-header">
        <h1>Research</h1>
      </div>

      {!canStart && (
        <div className="card read-only-note" role="note">
          Your role ({user?.role ?? "viewer"}) can view research but not start it. Ask an
          admin for the researcher role.
        </div>
      )}

      {canStart && (
      <form className="card ask" onSubmit={submit}>
        <label htmlFor="question" className="card-title">
          Ask a research question
        </label>
        <textarea
          id="question"
          ref={questionBox}
          rows={3}
          value={question}
          placeholder="e.g. What are the major approaches to retrieval-augmented generation?"
          onChange={(event) => setQuestion(event.target.value)}
          onKeyDown={(event) => {
            // Cmd/Ctrl+Enter submits.
            if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
              submit(event);
            }
          }}
        />
        {error && (
          <p className="form-error" role="alert">
            {error}
          </p>
        )}
        <div className="ask-actions">
          <span className="tile-note">A worker picks it up and you can watch it live.</span>
          <button type="submit" className="primary-button" disabled={submitting || !question.trim()}>
            {submitting ? "Starting…" : "Start research"}
          </button>
        </div>
      </form>
      )}

      <Card
        title="Recent research"
        subtitle={list.data ? `${list.data.total} in total` : undefined}
      >
        {list.error && !list.data && (
          <p className="empty-note">Couldn’t load research tasks ({list.error.message}).</p>
        )}
        {list.data && list.data.items.length === 0 && (
          <p className="empty-note">No research yet. Ask a question above.</p>
        )}
        {list.data && list.data.items.length > 0 && (
          <ul className="status-list task-list">
            {list.data.items.map((task) => {
              const status = TASK_STATUS[task.status];

              return (
                <li key={task.id}>
                  <Link to={`/research/${task.id}`} className="label task-link">
                    {task.question}
                  </Link>
                  {task.created_by && <span className="detail">by {task.created_by}</span>}
                  <span className="detail">
                    {new Date(task.created_at).toLocaleString(undefined, {
                      month: "short",
                      day: "numeric",
                      hour: "numeric",
                      minute: "2-digit",
                    })}
                  </span>
                  <span className="task-status">
                    <StatusMark tone={status.tone} label={status.label} />
                  </span>
                </li>
              );
            })}
          </ul>
        )}
      </Card>
    </>
  );
}
