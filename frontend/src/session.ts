// The signed-in session: the access token and when it expires, kept in
// localStorage so a reload stays signed in. Read by api.ts for every
// request; an expired token is never sent.
//
// (localStorage is readable by any script on the page, so the app relies
// on not running untrusted script: React escapes rendered text, and the
// research summary's Markdown is rendered without raw HTML.)

const KEY = "research-agent.session";

export interface SessionUser {
  id: number;
  username: string;
}

export interface Session {
  token: string;
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

export function getToken(): string | null {
  return getSession()?.token ?? null;
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
