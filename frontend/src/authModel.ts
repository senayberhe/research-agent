import { createContext, useContext } from "react";

import type { Permission, SessionUser } from "./session";

export interface AuthContext {
  // Undefined while an existing session is being checked on load.
  user: SessionUser | null | undefined;
  signIn: (username: string, password: string) => Promise<void>;
  signOut: () => void;
}

export const Context = createContext<AuthContext | null>(null);

// Whether the signed-in user's role allows this (for showing or hiding
// controls; the server enforces it regardless).
export function useCan(permission: Permission): boolean {
  const { user } = useAuth();
  // A session saved before roles existed has no list until /auth/me
  // refreshes it: allow nothing extra meanwhile.
  return Boolean(user?.permissions?.includes(permission));
}

export function useAuth(): AuthContext {
  const context = useContext(Context);

  if (!context) {
    throw new Error("useAuth needs an AuthProvider");
  }

  return context;
}
