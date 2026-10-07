import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";

import {
  TOAST_MS,
  useNotifications,
  type Notification,
  type NotificationKind,
} from "../notificationModel";
import { StatusMark } from "./status";
import type { Tone } from "./tones";

const TONE: Record<NotificationKind, Tone> = {
  success: "good",
  warning: "warning",
  error: "critical",
};

// How many toasts show at once (the newest); the rest are in the bell.
const MAX_TOASTS = 4;

function Toast({ notification }: { notification: Notification }) {
  const { dismissToast } = useNotifications();
  const [paused, setPaused] = useState(false);
  const remaining = useRef(TOAST_MS[notification.kind]);

  // Auto-dismiss, paused while hovered or focused (time left is kept).
  useEffect(() => {
    if (paused) return;

    const started = Date.now();
    const timer = window.setTimeout(
      () => dismissToast(notification.id),
      remaining.current,
    );

    return () => {
      window.clearTimeout(timer);
      remaining.current -= Date.now() - started;
    };
  }, [paused, notification.id, dismissToast]);

  return (
    <div
      className={`toast toast-${notification.kind}`}
      role={notification.kind === "error" ? "alert" : "status"}
      data-notification-id={notification.id}
      onMouseEnter={() => setPaused(true)}
      onMouseLeave={() => setPaused(false)}
      onFocus={() => setPaused(true)}
      onBlur={() => setPaused(false)}
    >
      <div className="toast-content">
        <div className="toast-title">
          <StatusMark tone={TONE[notification.kind]} label={notification.title} />
        </div>
        <Link to={notification.link} className="toast-body" onClick={() => dismissToast(notification.id)}>
          {notification.body}
        </Link>
      </div>
      <button
        type="button"
        className="toast-close"
        aria-label={`Dismiss: ${notification.title}`}
        onClick={() => dismissToast(notification.id)}
      >
        ×
      </button>
    </div>
  );
}

export function Toasts() {
  const { notifications } = useNotifications();

  const showing = notifications.filter((n) => n.toast).slice(0, MAX_TOASTS);

  return (
    <div className="toasts" aria-live="polite">
      {showing.map((notification) => (
        <Toast key={notification.id} notification={notification} />
      ))}
    </div>
  );
}

const STATUS_LABEL = {
  connecting: { tone: "neutral" as Tone, label: "Connecting…" },
  live: { tone: "good" as Tone, label: "Live" },
  reconnecting: { tone: "warning" as Tone, label: "Reconnecting…" },
};

export function NotificationBell() {
  const { notifications, unread, status, markAllRead, clear } = useNotifications();
  const [open, setOpen] = useState(false);
  const panel = useRef<HTMLDivElement>(null);

  // Opening the panel marks everything read.
  useEffect(() => {
    if (open) markAllRead();
  }, [open, notifications.length, markAllRead]);

  // Close on Escape or a click outside.
  useEffect(() => {
    if (!open) return;

    const onKey = (event: KeyboardEvent) => event.key === "Escape" && setOpen(false);
    const onClick = (event: MouseEvent) => {
      if (panel.current && !panel.current.contains(event.target as Node)) setOpen(false);
    };

    document.addEventListener("keydown", onKey);
    document.addEventListener("mousedown", onClick);

    return () => {
      document.removeEventListener("keydown", onKey);
      document.removeEventListener("mousedown", onClick);
    };
  }, [open]);

  const live = STATUS_LABEL[status];

  return (
    <div className="bell" ref={panel}>
      <button
        type="button"
        className="bell-button"
        aria-label={unread ? `Notifications, ${unread} unread` : "Notifications"}
        aria-expanded={open}
        onClick={() => setOpen((value) => !value)}
      >
        <svg width="16" height="16" viewBox="0 0 24 24" aria-hidden="true">
          <path
            d="M6 8a6 6 0 1 1 12 0c0 7 3 9 3 9H3s3-2 3-9M10.3 21a1.94 1.94 0 0 0 3.4 0"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
        {unread > 0 && <span className="bell-count">{unread > 99 ? "99+" : unread}</span>}
      </button>

      {open && (
        <div className="bell-panel" role="dialog" aria-label="Notifications">
          <div className="bell-header">
            <strong>Notifications</strong>
            <span className="bell-status">
              <StatusMark tone={live.tone} label={live.label} />
            </span>
          </div>

          {notifications.length === 0 ? (
            <p className="empty-note bell-empty">
              Nothing yet. You’ll be notified when research completes or fails, a
              tool fails, or a job is retried.
            </p>
          ) : (
            <>
              <ul className="bell-list">
                {notifications.map((notification) => (
                  <li key={notification.id}>
                    <Link to={notification.link} onClick={() => setOpen(false)}>
                      <span className="bell-item-title">
                        <StatusMark tone={TONE[notification.kind]} label={notification.title} />
                      </span>
                      <span className="bell-item-body">{notification.body}</span>
                      <time dateTime={notification.createdAt}>
                        {new Date(notification.createdAt).toLocaleTimeString(undefined, {
                          hour: "numeric",
                          minute: "2-digit",
                        })}
                      </time>
                    </Link>
                  </li>
                ))}
              </ul>
              <div className="bell-footer">
                <button type="button" className="link-button" onClick={clear}>
                  Clear all
                </button>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
