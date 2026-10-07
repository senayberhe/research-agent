// Typed access to the Research Agent API. The base URL comes from
// VITE_API_URL (default: the API on localhost:8000, allowed by its
// CORS_ORIGINS).

import {
  clearSession,
  getSession,
  type Role,
  type Session,
  type SessionUser,
} from "./session";

export const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export type HealthStatus = "healthy" | "degraded" | "unhealthy";

export interface HealthComponent {
  status: HealthStatus;
  message: string;
}

export interface SystemHealth {
  status: HealthStatus;
  database: HealthComponent;
  workers: HealthComponent;
  jobs: HealthComponent;
}

export interface Worker {
  worker_id: string;
  // busy, idle or unresponsive.
  state: "busy" | "idle" | "unresponsive";
  current_job_id: number | null;
  last_attempt_started_at: string;
}

export interface SystemMetrics {
  since: string;
  generated_at: string;
  agents: {
    total_runs: number;
    total_tokens: number;
    estimated_cost_usd: number;
    unpriced_runs: number;
  };
  workers: Worker[];
  queue: {
    pending_jobs: number;
    running_jobs: number;
    oldest_pending_seconds: number | null;
    expired_leases: number;
  };
}

export interface TimeseriesBucket {
  start: string;
  completed: number;
  failed: number;
  p95_latency_seconds: number | null;
}

export interface JobTimeseries {
  range: RangeName;
  bucket_seconds: number;
  start: string;
  end: string;
  buckets: TimeseriesBucket[];
}

export type SLOStatus = "healthy" | "breached" | "no_data" | "unknown";

export interface SLOErrorBudget {
  slo_target: number;
  actual: number | null;
  budget_remaining: number | null;
  budget_remaining_percent: number | null;
  exhausted: boolean;
}

export interface SLOEntry {
  target: number;
  actual: number | null;
  target_seconds?: number | null;
  actual_ms?: number | null;
  percentiles?: { p50_ms: number | null; p95_ms: number | null; p99_ms: number | null } | null;
  by_tool?: Record<string, { p50_ms: number | null; p95_ms: number | null; p99_ms: number | null; status: SLOStatus }> | null;
  status: SLOStatus;
  error_budget: SLOErrorBudget | null;
}

export interface SLOReport {
  window_days: number;
  window_start: string;
  window_end: string;
  api_availability: SLOEntry;
  research_job_success: SLOEntry;
  tool_success: SLOEntry;
  job_latency: SLOEntry & { target_seconds: number };
  tool_latency: SLOEntry;
}

export type TaskStatus =
  | "pending"
  | "planning"
  | "researching"
  | "synthesizing"
  | "in_progress"
  | "completed"
  | "failed";

export interface ResearchTask {
  id: number;
  question: string;
  status: TaskStatus;
  summary: string | null;
  created_at: string;
  // Who started it (null for tasks from before accounts).
  created_by: string | null;
}

export interface ResearchList {
  items: ResearchTask[];
  total: number;
  limit: number;
  offset: number;
}

export type RunStatus =
  | "created"
  | "running"
  | "waiting_for_tool"
  | "processing_result"
  | "completed"
  | "failed";

export interface RunProgress {
  id: number;
  status: RunStatus;
  live: boolean;
  iterations: number;
  tool_calls: number;
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  estimated_cost_usd: number | null;
  error: string | null;
  started_at: string;
  completed_at: string | null;
  elapsed_seconds: number;
}

export interface TimelineEvent {
  timestamp: string;
  event_type: string;
  source: "job" | "step" | "agent";
  message: string | null;
  metadata: Record<string, unknown>;
}

export interface ToolCall {
  id: number;
  tool: string;
  query: string;
  iteration: number;
  status: "pending" | "running" | "completed" | "failed";
  duration_ms: number | null;
  started_at: string;
  success: boolean | null;
  error: string | null;
  result_preview: string | null;
  result_truncated: boolean;
  result_length: number;
}

export interface TaskProgress {
  task: ResearchTask;
  finished: boolean;
  can_resume: boolean;
  job: {
    id: number;
    status: string;
    attempts: number;
    worker_id: string | null;
    error: string | null;
  } | null;
  run: RunProgress | null;
  tool_calls: ToolCall[];
  events: TimelineEvent[];
}

export interface AdminUser {
  id: number;
  username: string;
  role: Role;
  is_active: boolean;
  created_at: string;
  last_login_at: string | null;
}

const pickUser = (user: SessionUser): SessionUser => ({
  id: user.id,
  username: user.username,
  role: user.role,
  permissions: user.permissions,
});

export type JobStatus = "pending" | "running" | "completed" | "failed";

export interface Job {
  id: number;
  task_id: number;
  status: JobStatus;
  attempts: number;
  worker_id: string | null;
  lease_expires_at: string | null;
  error: string | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
}

export interface JobListItem extends Job {
  question: string;
}

export interface JobList {
  items: JobListItem[];
  total: number;
  limit: number;
  offset: number;
}

export interface JobAttempt {
  attempt_number: number;
  worker_id: string | null;
  // running, completed, failed, lease_expired or released.
  outcome: string;
  error: string | null;
  started_at: string;
  ended_at: string | null;
  duration_seconds: number | null;
}

export interface JobEventDetail {
  id: number;
  event_type: string;
  message: string | null;
  summary: string | null;
  created_at: string;
}

export interface JobDetail extends Job {
  attempt_history: JobAttempt[];
  events: JobEventDetail[];
}

// A message from GET /events (server-sent).
export interface StreamedJobEvent {
  id: number;
  job_id: number;
  task_id: number;
  question: string;
  event_type: "completed" | "failed" | "tool_failed" | "requeued" | "lease_expired";
  message: string | null;
  metadata: Record<string, unknown>;
  created_at: string;
}

export type RangeName = "24h" | "7d" | "30d";

export const RANGE_HOURS: Record<RangeName, number> = {
  "24h": 24,
  "7d": 24 * 7,
  "30d": 24 * 30,
};

// A 401 on a request that was signed in: the session ended (expired,
// signed out elsewhere, account deactivated).
export class SessionExpired extends Error {}

async function request<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const signedIn = getSession() !== null;

  const response = await fetch(`${API_URL}${path}`, {
    ...init,
    // Sends the session cookie (the API is on another origin in
    // development).
    credentials: "include",
  });

  if (response.status === 401 && signedIn) {
    clearSession();
    throw new SessionExpired("Your session has ended. Sign in again.");
  }

  if (!response.ok) {
    // FastAPI puts the reason in "detail".
    let detail = `HTTP ${response.status}`;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      // Not JSON: keep the status.
    }
    throw new Error(detail);
  }

  if (response.status === 204) return undefined as T;

  return (await response.json()) as T;
}

const get = <T>(path: string, signal?: AbortSignal) =>
  request<T>(path, { signal });

const post = <T>(path: string, body?: unknown) =>
  request<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });

export const api = {
  health: (signal?: AbortSignal) => get<SystemHealth>("/health", signal),

  systemMetrics: (range: RangeName, signal?: AbortSignal) =>
    get<SystemMetrics>(
      `/research/metrics/system?hours=${RANGE_HOURS[range]}`,
      signal,
    ),

  jobTimeseries: (range: RangeName, signal?: AbortSignal) =>
    get<JobTimeseries>(`/research/metrics/timeseries?range=${range}`, signal),

  slo: (signal?: AbortSignal) => get<SLOReport>("/research/slo", signal),

  login: async (username: string, password: string): Promise<Session> => {
    // The session id comes back as a cookie, not in the body.
    const body = await post<{
      expires_at: string;
      user: SessionUser;
    }>("/auth/login", { username, password });

    return {
      expiresAt: body.expires_at,
      user: pickUser(body.user),
    };
  },

  // Ends the session on the server (and clears the cookie).
  logout: () => post<void>("/auth/logout"),

  me: async (signal?: AbortSignal) => pickUser(await get<SessionUser>("/auth/me", signal)),

  // Users (admins).
  listUsers: (signal?: AbortSignal) => get<AdminUser[]>("/users", signal),

  createUser: (username: string, password: string, role: Role) =>
    post<AdminUser>("/users", { username, password, role }),

  updateUser: (id: number, changes: Partial<{ role: Role; is_active: boolean; password: string }>) =>
    request<AdminUser>(`/users/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(changes),
    }),

  listResearch: (signal?: AbortSignal) =>
    get<ResearchList>("/research?limit=20", signal),

  createResearch: (question: string) =>
    post<ResearchTask>("/research", { question }),

  progress: (taskId: number, signal?: AbortSignal) =>
    get<TaskProgress>(`/research/${taskId}/progress`, signal),

  resume: (taskId: number) => post<ResearchTask>(`/research/${taskId}/resume`),

  listJobs: (status: JobStatus | null, offset: number, signal?: AbortSignal) =>
    get<JobList>(
      `/jobs?limit=20&offset=${offset}${status ? `&status=${status}` : ""}`,
      signal,
    ),

  job: (jobId: number, signal?: AbortSignal) => get<JobDetail>(`/jobs/${jobId}`, signal),

  task: (taskId: number, signal?: AbortSignal) =>
    get<ResearchTask>(`/research/${taskId}`, signal),
};
