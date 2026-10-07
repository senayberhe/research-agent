import { Link, NavLink, Navigate, Route, Routes } from "react-router-dom";

import { AuthProvider } from "./AuthProvider";
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
import { TaskDetail } from "./pages/TaskDetail";
import { RequireAuth } from "./RequireAuth";

const PAGES = [
  { path: "/", label: "Dashboard", ready: true },
  { path: "/research", label: "Research", ready: true },
  { path: "/jobs", label: "Jobs", ready: true },
  { path: "/slo", label: "SLO", ready: false },
  { path: "/monitoring", label: "Monitoring", ready: false },
  { path: "/workers", label: "Workers", ready: false },
  { path: "/settings", label: "Settings", ready: false },
];

// The app behind sign-in: shell, notifications (their stream needs the
// token), pages.
function Shell() {
  return (
    <NotificationProvider>
      <div className="app">
        <header className="topbar">
          <span className="brand">Research Agent</span>
          <div className="topbar-actions">
            {/* On every page: straight to the question box, ready to type. */}
            <Link to="/research" state={{ focusQuestion: true }} className="primary-button new-research">
              + New research
            </Link>
            <NotificationBell />
            <HealthBadge />
            <UserMenu />
          </div>
        </header>

        <nav className="sidebar" aria-label="Main">
          {PAGES.map((page) => (
            <NavLink
              key={page.path}
              to={page.path}
              end={page.path === "/"}
              className={({ isActive }) => (isActive ? "nav-link active" : "nav-link")}
            >
              {page.label}
              {!page.ready && <span className="nav-soon">soon</span>}
            </NavLink>
          ))}
        </nav>

        <main className="content">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/research" element={<ResearchList />} />
            <Route path="/research/:taskId" element={<TaskDetail />} />
            <Route path="/jobs" element={<JobsList />} />
            <Route path="/jobs/:jobId" element={<JobDetail />} />
            {PAGES.filter((page) => !page.ready).map((page) => (
              <Route key={page.path} path={page.path} element={<ComingSoon title={page.label} />} />
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
