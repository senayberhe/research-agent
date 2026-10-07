import type { ReactNode } from "react";

import { useAuth, useCan } from "./authModel";
import type { Permission } from "./session";

const WHO: Record<Permission, string> = {
  view: "Everyone signed in",
  "research:create": "Researchers, operators and admins",
  "research:resume": "Researchers, operators and admins",
  "analytics:view": "Operators and admins",
  "users:manage": "Admins",
};

// A page only some roles may open. Typed-in URLs land here too: the page
// isn't rendered (so it makes no API calls), an explanation is. The API
// refuses the data anyway (403); this just says so politely.
export function RequirePermission({
  permission,
  title,
  children,
}: {
  permission: Permission;
  title: string;
  children: ReactNode;
}) {
  const allowed = useCan(permission);
  const { user } = useAuth();

  if (allowed) return <>{children}</>;

  return (
    <>
      <div className="page-header">
        <h1>{title}</h1>
      </div>
      <div className="card access-denied" role="alert">
        <strong>You don’t have access to {title}.</strong> Your role is{" "}
        {user?.role ?? "unknown"}; {WHO[permission].toLowerCase()} can open this page.
      </div>
    </>
  );
}
