import { api } from "../api";
import { usePolling } from "../usePolling";
import { StatusMark } from "./status";
import { HEALTH_LABEL, HEALTH_TONE } from "./tones";

// Top-bar badge from GET /health; hover or focus shows each component.
export function HealthBadge() {
  const { data, error } = usePolling((signal) => api.health(signal), "health");

  if (error && !data) {
    return (
      <div className="health-badge" tabIndex={0}>
        <StatusMark tone="critical" label="API unreachable" />
        <div className="health-details" role="tooltip">
          <div className="health-row">
            <span>{error.message}</span>
          </div>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="health-badge">
        <StatusMark tone="neutral" label="Checking…" />
      </div>
    );
  }

  return (
    <div className="health-badge" tabIndex={0} aria-label={HEALTH_LABEL[data.status]}>
      <StatusMark tone={HEALTH_TONE[data.status]} label={HEALTH_LABEL[data.status]} />
      <div className="health-details" role="tooltip">
        {(["database", "workers", "jobs"] as const).map((name) => (
          <div className="health-row" key={name}>
            <strong>
              <StatusMark
                tone={HEALTH_TONE[data[name].status]}
                label={name[0].toUpperCase() + name.slice(1)}
              />
            </strong>
            <span>{data[name].message}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
