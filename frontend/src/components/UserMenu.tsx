import { useAuth } from "../authModel";

export function UserMenu() {
  const { user, signOut } = useAuth();

  if (!user) return null;

  return (
    <div className="user-menu">
      <span className="user-name" title="Signed in">
        {user.username}
      </span>
      <button type="button" className="link-button" onClick={signOut}>
        Sign out
      </button>
    </div>
  );
}
