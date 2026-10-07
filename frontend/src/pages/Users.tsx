import { useState, type FormEvent } from "react";

import { api, type AdminUser } from "../api";
import { useAuth, useCan } from "../authModel";
import { Card } from "../components/Card";
import { StatusMark } from "../components/status";
import type { Role } from "../session";
import { usePolling } from "../usePolling";

const ROLES: { value: Role; label: string; can: string }[] = [
  { value: "viewer", label: "Viewer", can: "view research and jobs" },
  { value: "researcher", label: "Researcher", can: "+ start and resume research" },
  { value: "operator", label: "Operator", can: "+ analytics and SLOs" },
  { value: "admin", label: "Admin", can: "+ manage users" },
];

const when = (iso: string | null) =>
  iso
    ? new Date(iso).toLocaleString(undefined, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" })
    : "never";

function AddUser({ onAdded }: { onAdded: () => void }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState<Role>("viewer");
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setSaving(true);
    setError(null);
    try {
      await api.createUser(username, password, role);
      setUsername("");
      setPassword("");
      setRole("viewer");
      onAdded();
    } catch (failure) {
      setError((failure as Error).message);
    } finally {
      setSaving(false);
    }
  };

  return (
    <form className="add-user" onSubmit={submit}>
      <label>
        Username
        <input value={username} onChange={(e) => setUsername(e.target.value)} required autoComplete="off" />
      </label>
      <label>
        Password
        <input
          type="password"
          value={password}
          placeholder="12+ characters"
          onChange={(e) => setPassword(e.target.value)}
          required
          autoComplete="new-password"
        />
      </label>
      <label>
        Role
        <select value={role} onChange={(e) => setRole(e.target.value as Role)}>
          {ROLES.map((r) => (
            <option key={r.value} value={r.value}>
              {r.label}
            </option>
          ))}
        </select>
      </label>
      <button type="submit" className="primary-button" disabled={saving || !username || !password}>
        {saving ? "Adding…" : "Add user"}
      </button>
      {error && (
        <p className="form-error add-user-error" role="alert">
          {error}
        </p>
      )}
    </form>
  );
}

function UserRow({ user, self, onChanged }: { user: AdminUser; self: boolean; onChanged: (error?: string) => void }) {
  const [busy, setBusy] = useState(false);

  const change = async (changes: Parameters<typeof api.updateUser>[1]) => {
    setBusy(true);
    try {
      await api.updateUser(user.id, changes);
      onChanged();
    } catch (failure) {
      onChanged((failure as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <tr className={user.is_active ? undefined : "row-inactive"}>
      <td>
        <strong>{user.username}</strong>
        {self && <span className="tag tag-neutral">you</span>}
      </td>
      <td>
        <select
          aria-label={`Role of ${user.username}`}
          value={user.role}
          disabled={busy || self}
          title={self ? "You can't change your own role" : undefined}
          onChange={(e) => change({ role: e.target.value as Role })}
        >
          {ROLES.map((r) => (
            <option key={r.value} value={r.value}>
              {r.label}
            </option>
          ))}
        </select>
      </td>
      <td>
        <span className="task-status">
          <StatusMark tone={user.is_active ? "good" : "neutral"} label={user.is_active ? "Active" : "Deactivated"} />
        </span>
      </td>
      <td className="cell-muted">{when(user.last_login_at)}</td>
      <td className="cell-muted">{when(user.created_at)}</td>
      <td>
        <button
          type="button"
          className="link-button"
          disabled={busy || self}
          title={self ? "You can't deactivate yourself" : undefined}
          onClick={() => change({ is_active: !user.is_active })}
        >
          {user.is_active ? "Deactivate" : "Activate"}
        </button>
      </td>
    </tr>
  );
}

export function Users() {
  const canManage = useCan("users:manage");
  const { user: me } = useAuth();
  const [message, setMessage] = useState<string | null>(null);

  const users = usePolling(
    (signal) => (canManage ? api.listUsers(signal) : Promise.resolve([])),
    `users-${canManage}`,
    0,
  );

  if (!canManage) {
    return (
      <>
        <div className="page-header">
          <h1>Users</h1>
        </div>
        <div className="card read-only-note" role="note">
          Only admins can manage users.
        </div>
      </>
    );
  }

  const changed = (error?: string) => {
    setMessage(error ?? null);
    users.reload();
  };

  return (
    <>
      <div className="page-header">
        <h1>Users</h1>
      </div>

      {message && (
        <div className="error-banner" role="alert">
          {message}
        </div>
      )}

      <Card title="Roles">
        <ul className="role-list">
          {ROLES.map((r) => (
            <li key={r.value}>
              <span className={`role-badge role-${r.value}`}>{r.value}</span> {r.can}
            </li>
          ))}
        </ul>
      </Card>

      <div className="users-section">
        <Card title="Add a user" subtitle="New users get the role you pick (viewer by default)">
          <AddUser onAdded={() => changed()} />
        </Card>
      </div>

      <div className="users-section">
        <Card title="Accounts" subtitle={users.data ? `${users.data.length} user${users.data.length === 1 ? "" : "s"}` : undefined}>
          {users.error && !users.data && <p className="empty-note">Couldn’t load users ({users.error.message}).</p>}
          {users.data && (
            <div className="table-wrap">
              <table className="jobs-table users-table">
                <thead>
                  <tr>
                    <th scope="col">User</th>
                    <th scope="col">Role</th>
                    <th scope="col">Status</th>
                    <th scope="col">Last sign-in</th>
                    <th scope="col">Created</th>
                    <th scope="col">
                      <span className="visually-hidden">Actions</span>
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {users.data.map((user) => (
                    <UserRow key={user.id} user={user} self={user.id === me?.id} onChanged={changed} />
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>
      </div>
    </>
  );
}
