// Who is signed in, and until when. The sign-in itself is the session
// cookie the API sets (httpOnly: no script on the page can read it, and
// the browser sends it with every request). This copy, in localStorage, is
// only so a reload shows the right user straight away and other tabs hear
// about sign-in and sign-out; it holds nothing that grants access.

const KEY = "research-agent.session";

export type Role = "viewer" | "researcher" | "operator" | "admin";

// What a role may do (app/core/permissions.py on the server, which is what
// actually enforces it; the UI only hides what isn't allowed).
export type Permission =
  | "view"
  | "research:create"
  | "research:resume"
  | "analytics:view"
  | "users:manage";

export interface SessionUser {
  id: number;
  username: string;
  role: Role;
  permissions: Permission[];
}

export interface Session {
  expiresAt: string;
  user: SessionUser;
}

type Listener = (session: Session | null) => void;

const listeners = new Set<Listener>();

function read(): Session | null {
  try {
    const raw = window.localStorage.getItem(KEY);
    return raw ? (JSON.parse(raw) as Session) : null;
  } catch {
    return null;
  }
}

export function getSession(): Session | null {
  const session = read();

  if (session && new Date(session.expiresAt).getTime() <= Date.now()) {
    clearSession();
    return null;
  }

  return session;
}

export function setSession(session: Session) {
  try {
    window.localStorage.setItem(KEY, JSON.stringify(session));
  } catch {
    // Storage unavailable (private mode): signed in for this page only.
  }
  listeners.forEach((listener) => listener(session));
}

export function clearSession() {
  try {
    window.localStorage.removeItem(KEY);
  } catch {
    // Nothing stored.
  }
  listeners.forEach((listener) => listener(null));
}

export function onSessionChange(listener: Listener): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

// Another tab signed in or out.
if (typeof window !== "undefined") {
  window.addEventListener("storage", (event) => {
    if (event.key === KEY) listeners.forEach((listener) => listener(getSession()));
  });
}
