import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";

import type { StreamedJobEvent } from "./api";
import { streamEvents } from "./eventStream";
import {
  Context,
  HISTORY_LIMIT,
  toNotification,
  type Notification,
  type StreamStatus,
} from "./notificationModel";

// One stream for the whole app (GET /events, read by eventStream.ts); see
// notificationModel.ts.
export function NotificationProvider({ children }: { children: ReactNode }) {
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [status, setStatus] = useState<StreamStatus>("connecting");

  // Ids already shown: a second guard against duplicates (the server
  // already resumes after the last id on reconnect).
  const seen = useRef(new Set<number>());
  const listeners = useRef(new Set<(event: StreamedJobEvent) => void>());

  useEffect(() => {
    // fetch-based (EventSource can't send the Authorization header);
    // reconnects by itself, resuming after the last event id.
    const controller = new AbortController();

    streamEvents({
      path: "/events",
      signal: controller.signal,
      onStatus: setStatus,
      onMessage: (name, data) => {
        if (name !== "job_event") return;

        const event = JSON.parse(data) as StreamedJobEvent;

        listeners.current.forEach((listener) => listener(event));

        if (seen.current.has(event.id)) return;
        seen.current.add(event.id);

        setNotifications((current) =>
          [toNotification(event), ...current].slice(0, HISTORY_LIMIT),
        );
      },
    });

    return () => controller.abort();
  }, []);

  const dismissToast = useCallback(
    (id: number) =>
      setNotifications((current) =>
        current.map((n) => (n.id === id ? { ...n, toast: false } : n)),
      ),
    [],
  );

  const markAllRead = useCallback(
    () => setNotifications((current) => current.map((n) => ({ ...n, read: true }))),
    [],
  );

  const clear = useCallback(() => setNotifications([]), []);

  const subscribe = useCallback((listener: (event: StreamedJobEvent) => void) => {
    listeners.current.add(listener);
    return () => {
      listeners.current.delete(listener);
    };
  }, []);

  const value = useMemo(
    () => ({
      notifications,
      unread: notifications.filter((n) => !n.read).length,
      status,
      dismissToast,
      markAllRead,
      clear,
      subscribe,
    }),
    [notifications, status, dismissToast, markAllRead, clear, subscribe],
  );

  return <Context.Provider value={value}>{children}</Context.Provider>;
}

