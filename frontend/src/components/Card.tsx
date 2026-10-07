import { useState, type ReactNode } from "react";

export function Card({
  title,
  subtitle,
  action,
  children,
}: {
  title: string;
  subtitle?: string;
  action?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className="card">
      <div className="card-header">
        <div>
          <h2 className="card-title">{title}</h2>
          {subtitle && <p className="card-subtitle">{subtitle}</p>}
        </div>
        {action}
      </div>
      {children}
    </section>
  );
}

// A chart card with a table-view twin, so no value is hover-only.
export function ChartCard({
  title,
  subtitle,
  chart,
  table,
}: {
  title: string;
  subtitle?: string;
  chart: ReactNode;
  table: ReactNode;
}) {
  const [showTable, setShowTable] = useState(false);

  return (
    <Card
      title={title}
      subtitle={subtitle}
      action={
        <button
          type="button"
          className="link-button"
          aria-pressed={showTable}
          onClick={() => setShowTable((value) => !value)}
        >
          {showTable ? "Chart" : "Table"}
        </button>
      }
    >
      {showTable ? table : chart}
    </Card>
  );
}

export function StatTile({
  label,
  value,
  note,
}: {
  label: string;
  value: string | null;
  note?: string;
}) {
  return (
    <section className="card">
      <p className="tile-label">{label}</p>
      <p className={value === null ? "tile-value no-data" : "tile-value"}>
        {value ?? "No data"}
      </p>
      {note && <p className="tile-note">{note}</p>}
    </section>
  );
}
