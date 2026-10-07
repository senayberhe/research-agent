import { createContext, useContext, useEffect, useRef } from "react";

import type { StreamedJobEvent } from "./api";
import { toolName } from "./components/taskStatus";

// One event stream for the whole app (GET /events). Each job event becomes a
// notification: a toast that dismisses itself, and an entry in the bell's
// history. Pages can also subscribe to the raw events, e.g. to refresh at
// once instead of waiting for their next poll.

export type NotificationKind = "success" | "warning" | "error";

export interface Notification {
  // The job event's id: unique, and the same after a reconnect.
  id: number;
  kind: NotificationKind;
  title: string;
  body: string;
  link: string;
  createdAt: string;
  read: boolean;
  // Still showing as a toast.
  toast: boolean;
}

export type { StreamStatus } from "./eventStream";
import type { StreamStatus } from "./eventStream";

export interface NotificationsContext {
  notifications: Notification[];
  unread: number;
  status: StreamStatus;
  dismissToast: (id: number) => void;
  markAllRead: () => void;
  clear: () => void;
  subscribe: (listener: (event: StreamedJobEvent) => void) => () => void;
}

export const Context = createContext<NotificationsContext | null>(null);

// How long a toast stays, by kind (errors longest: they need reading).
export const TOAST_MS: Record<NotificationKind, number> = {
  success: 6_000,
  warning: 8_000,
  error: 10_000,
};

export const HISTORY_LIMIT = 50;

export function toNotification(event: StreamedJobEvent): Notification {
  const question = event.question;
  const m = event.metadata;
  const base = {
    id: event.id,
    link: `/research/${event.task_id}`,
    createdAt: event.created_at,
    read: false,
    toast: true,
  };

  switch (event.event_type) {
    case "completed":
      return { ...base, kind: "success", title: "Research completed", body: question };
    case "failed":
      return {
        ...base,
        kind: "error",
        title: "Research failed",
        body: event.message ? `${event.message} · ${question}` : question,
      };
    case "tool_failed":
      return {
        ...base,
        kind: "error",
        title: `Tool failed: ${toolName(m.tool)}`,
        body: m.error ? `${m.error} · ${question}` : question,
      };
    case "requeued":
      return {
        ...base,
        kind: "warning",
        title: "Job retrying",
        body: question,
        link: `/jobs/${event.job_id}`,
      };
    case "lease_expired":
      return {
        ...base,
        kind: "warning",
        title: "Worker stopped responding",
        body: `Job #${event.job_id} returned to the queue · ${question}`,
        link: `/jobs/${event.job_id}`,
      };
  }
}

export function useNotifications(): NotificationsContext {
  const context = useContext(Context);

  if (!context) {
    throw new Error("useNotifications needs a NotificationProvider");
  }

  return context;
}

// Calls `onEvent` for each streamed job event that matches `filter`.
export function useJobEvents(
  filter: (event: StreamedJobEvent) => boolean,
  onEvent: () => void,
) {
  const { subscribe } = useNotifications();
  const latest = useRef({ filter, onEvent });

  useEffect(() => {
    latest.current = { filter, onEvent };
  });

  useEffect(
    () =>
      subscribe((event) => {
        if (latest.current.filter(event)) latest.current.onEvent();
      }),
    [subscribe],
  );
}
