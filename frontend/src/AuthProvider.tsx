import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";

import { api } from "./api";
import { Context } from "./authModel";
import {
  clearSession,
  getSession,
  onSessionChange,
  setSession,
  type SessionUser,
} from "./session";

export function AuthProvider({ children }: { children: ReactNode }) {
  const stored = getSession();
  // A stored session is checked with the API before it's trusted.
  const [user, setUser] = useState<SessionUser | null | undefined>(
    stored ? undefined : null,
  );

  useEffect(() => {
    if (!stored) return;

    const controller = new AbortController();

    api
      .me(controller.signal)
      .then((me) => setUser({ id: me.id, username: me.username }))
      .catch((error: Error) => {
        if (error.name === "AbortError") return;
        // 401 already cleared it; anything else (API down): keep the
        // session, the pages show their own errors.
        setUser(getSession()?.user ?? null);
      });

    return () => controller.abort();
    // Only on load.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Signed out anywhere (a 401, another tab, Sign out).
  useEffect(
    () => onSessionChange((session) => setUser(session ? session.user : null)),
    [],
  );

  const signIn = useCallback(async (username: string, password: string) => {
    const session = await api.login(username, password);
    setSession(session);
  }, []);

  const signOut = useCallback(() => clearSession(), []);

  const value = useMemo(() => ({ user, signIn, signOut }), [user, signIn, signOut]);

  return <Context.Provider value={value}>{children}</Context.Provider>;
}
