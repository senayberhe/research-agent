import { Link, NavLink, Navigate, Route, Routes } from "react-router-dom";

import { AuthProvider } from "./AuthProvider";
import { useAuth, useCan } from "./authModel";
import { HealthBadge } from "./components/HealthBadge";
import { NotificationBell, Toasts } from "./components/Notifications";
import { UserMenu } from "./components/UserMenu";
import { NotificationProvider } from "./notifications";
import { ComingSoon } from "./pages/ComingSoon";
import { Dashboard } from "./pages/Dashboard";
import { JobDetail } from "./pages/JobDetail";
import { JobsList } from "./pages/JobsList";
import { Login } from "./pages/Login";
import { ResearchList } from "./pages/ResearchList";
import { SLO } from "./pages/SLO";
import { TaskDetail } from "./pages/TaskDetail";
import { Users } from "./pages/Users";
import { RequireAuth } from "./RequireAuth";
import { RequirePermission } from "./RequirePermission";
import type { Permission } from "./session";

// The sidebar: each page and the permission it needs (pages a role can't
// open aren't listed; RequirePermission blocks them if typed in).
const PAGES: { path: string; label: string; ready: boolean; needs: Permission }[] = [
  { path: "/analytics", label: "Analytics", ready: true, needs: "analytics:view" },
  { path: "/research", label: "Research", ready: true, needs: "view" },
  { path: "/jobs", label: "Jobs", ready: true, needs: "view" },
  { path: "/slo", label: "SLO", ready: true, needs: "analytics:view" },
  { path: "/monitoring", label: "Monitoring", ready: false, needs: "analytics:view" },
  { path: "/workers", label: "Workers", ready: false, needs: "analytics:view" },
  { path: "/settings", label: "Settings", ready: false, needs: "view" },
  { path: "/users", label: "Users", ready: true, needs: "users:manage" },
];

// "/": analytics for those who may see it, research for everyone else.
function Home() {
  return <Navigate to={useCan("analytics:view") ? "/analytics" : "/research"} replace />;
}

// The app behind sign-in: shell, notifications (their stream needs the
// token), pages.
function Shell() {
  const { user } = useAuth();
  const canStart = useCan("research:create");
  const allowed = new Set(user?.permissions ?? []);

  return (
    <NotificationProvider>
      <div className="app">
        <header className="topbar">
          <span className="brand">Research Agent</span>
          <div className="topbar-actions">
            {/* On every page: straight to the question box, ready to type
                (for roles that may start research). */}
            {canStart && (
              <Link to="/research" state={{ focusQuestion: true }} className="primary-button new-research">
                + New research
              </Link>
            )}
            <NotificationBell />
            <HealthBadge />
            <UserMenu />
          </div>
        </header>

        <nav className="sidebar" aria-label="Main">
          {PAGES.filter((page) => allowed.has(page.needs)).map((page) => (
            <NavLink
              key={page.path}
              to={page.path}
              className={({ isActive }) => (isActive ? "nav-link active" : "nav-link")}
            >
              {page.label}
              {!page.ready && <span className="nav-soon">soon</span>}
            </NavLink>
          ))}
        </nav>

        <main className="content">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route
              path="/analytics"
              element={
                <RequirePermission permission="analytics:view" title="Analytics">
                  <Dashboard />
                </RequirePermission>
              }
            />
            <Route
              path="/slo"
              element={
                <RequirePermission permission="analytics:view" title="SLOs">
                  <SLO />
                </RequirePermission>
              }
            />
            <Route path="/research" element={<ResearchList />} />
            <Route path="/research/:taskId" element={<TaskDetail />} />
            <Route path="/jobs" element={<JobsList />} />
            <Route path="/jobs/:jobId" element={<JobDetail />} />
            <Route
              path="/users"
              element={
                <RequirePermission permission="users:manage" title="Users">
                  <Users />
                </RequirePermission>
              }
            />
            {PAGES.filter((page) => !page.ready).map((page) => (
              <Route
                key={page.path}
                path={page.path}
                element={
                  <RequirePermission permission={page.needs} title={page.label}>
                    <ComingSoon title={page.label} />
                  </RequirePermission>
                }
              />
            ))}
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>

        <Toasts />
      </div>
    </NotificationProvider>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route
          path="*"
          element={
            <RequireAuth>
              <Shell />
            </RequireAuth>
          }
        />
      </Routes>
    </AuthProvider>
  );
}
