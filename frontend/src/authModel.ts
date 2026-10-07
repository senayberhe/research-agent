import { createContext, useContext } from "react";

import type { SessionUser } from "./session";

export interface AuthContext {
  // Undefined while an existing session is being checked on load.
  user: SessionUser | null | undefined;
  signIn: (username: string, password: string) => Promise<void>;
  signOut: () => void;
}

export const Context = createContext<AuthContext | null>(null);

export function useAuth(): AuthContext {
  const context = useContext(Context);

  if (!context) {
    throw new Error("useAuth needs an AuthProvider");
  }

  return context;
}
