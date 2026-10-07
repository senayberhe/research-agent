import { useEffect, useRef, useState } from "react";

import type { TimeseriesBucket } from "../api";
import {
  formatAxisTick,
  formatBucket,
  formatCount,
  formatSeconds,
} from "../format";

// Plot + axis bands. The container's height includes the x-axis band, so
// tick labels never get clipped.
const PLOT_HEIGHT = 196;
const AXIS_BAND = 24;
const LEFT = 44;
const RIGHT = 12;
// The latency chart's right margin holds the SLO label, clear of the data.
const RIGHT_WITH_LABEL = 52;
const TOP = 8;

function useWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [width, setWidth] = useState(0);

  useEffect(() => {
    const element = ref.current;
    if (!element) return;

    const observer = new ResizeObserver(([entry]) =>
      setWidth(entry.contentRect.width),
    );
    observer.observe(element);

    return () => observer.disconnect();
  }, []);

  return [ref, width] as const;
}

// Round axis maximum and ~4 evenly spaced ticks.
function niceTicks(max: number, count = 4): number[] {
  if (max <= 0) return [0, 1];

  const rough = max / count;
  const magnitude = 10 ** Math.floor(Math.log10(rough));
  const step =
    [1, 2, 2.5, 5, 10].map((m) => m * magnitude).find((s) => s >= rough) ??
    rough;

  const ticks = [];
  for (let value = 0; value <= max + step * 0.001; value += step) {
    ticks.push(Number(value.toFixed(10)));
  }
  if (ticks[ticks.length - 1] < max) ticks.push(ticks[ticks.length - 1] + step);

  return ticks;
}

// A bar rising from the baseline with only its data end (the top) rounded.
function topRoundedBar(x: number, y: number, w: number, h: number, r = 4) {
  const radius = Math.min(r, w / 2, h);

  return [
    `M${x},${y + h}`,
    `V${y + radius}`,
    `Q${x},${y} ${x + radius},${y}`,
    `H${x + w - radius}`,
    `Q${x + w},${y} ${x + w},${y + radius}`,
    `V${y + h}`,
    "Z",
  ].join(" ");
}

function XAxis({
  buckets,
  bucketSeconds,
  x,
  width,
}: {
  buckets: TimeseriesBucket[];
  bucketSeconds: number;
  x: (index: number) => number;
  width: number;
}) {
  // ~6 labels, whatever the width.
  const every = Math.max(1, Math.ceil(buckets.length / Math.max(2, width / 110)));

  return (
    <g>
      <line
        x1={LEFT}
        x2={LEFT + width}
        y1={TOP + PLOT_HEIGHT}
        y2={TOP + PLOT_HEIGHT}
        stroke="var(--axis)"
      />
      {buckets.map((bucket, index) =>
        index % every === 0 ? (
          <text
            key={bucket.start}
            x={x(index)}
            y={TOP + PLOT_HEIGHT + 16}
            textAnchor="middle"
          >
            {formatAxisTick(bucket.start, bucketSeconds)}
          </text>
        ) : null,
      )}
    </g>
  );
}

function YGrid({
  ticks,
  y,
  width,
  format,
}: {
  ticks: number[];
  y: (value: number) => number;
  width: number;
  format: (value: number) => string;
}) {
  return (
    <g>
      {ticks.map((tick) => (
        <g key={tick}>
          {tick > 0 && (
            <line
              x1={LEFT}
              x2={LEFT + width}
              y1={y(tick)}
              y2={y(tick)}
              stroke="var(--grid)"
            />
          )}
          <text x={LEFT - 8} y={y(tick) + 4} textAnchor="end">
            {format(tick)}
          </text>
        </g>
      ))}
    </g>
  );
}

// ---------------------------------------------------------------------
// Job throughput: stacked bars per bucket
// ---------------------------------------------------------------------

export function ThroughputChart({
  buckets,
  bucketSeconds,
}: {
  buckets: TimeseriesBucket[];
  bucketSeconds: number;
}) {
  const [ref, outerWidth] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);

  const width = Math.max(0, outerWidth - LEFT - RIGHT);
  const band = buckets.length ? width / buckets.length : 0;
  // A 2px surface gap between neighbouring bars.
  const barWidth = Math.max(1, band - 2);

  const max = Math.max(0, ...buckets.map((b) => b.completed + b.failed));
  const ticks = niceTicks(max);
  const top = ticks[ticks.length - 1];

  const y = (value: number) => TOP + PLOT_HEIGHT - (value / top) * PLOT_HEIGHT;
  const xCenter = (index: number) => LEFT + band * index + band / 2;

  const hovered = hover !== null ? buckets[hover] : null;

  return (
    <div>
      <div className="legend" aria-hidden="true">
        <span className="legend-item">
          <span className="legend-swatch" style={{ background: "var(--series-1)" }} />
          Completed
        </span>
        <span className="legend-item">
          <span className="legend-swatch" style={{ background: "var(--critical)" }} />
          Failed
        </span>
      </div>

      <div className="chart" ref={ref} onPointerLeave={() => setHover(null)}>
        {width > 0 && (
          <svg
            width={outerWidth}
            height={TOP + PLOT_HEIGHT + AXIS_BAND}
            role="img"
            aria-label="Jobs finished per period: completed and failed"
          >
            <YGrid ticks={ticks} y={y} width={width} format={(v) => formatCount(v)} />

            {buckets.map((bucket, index) => {
              const x = LEFT + band * index + (band - barWidth) / 2;
              const completedTop = y(bucket.completed);
              const failedTop = y(bucket.completed + bucket.failed);
              const dim = hover !== null && hover !== index;

              return (
                <g key={bucket.start} opacity={dim ? 0.55 : 1}>
                  {bucket.completed > 0 && (
                    <path
                      d={
                        bucket.failed > 0
                          ? // Not the data end: square top.
                            topRoundedBar(x, completedTop, barWidth, TOP + PLOT_HEIGHT - completedTop, 0)
                          : topRoundedBar(x, completedTop, barWidth, TOP + PLOT_HEIGHT - completedTop)
                      }
                      fill="var(--series-1)"
                    />
                  )}
                  {bucket.failed > 0 && (
                    <path
                      // 2px surface gap above the completed segment.
                      d={topRoundedBar(
                        x,
                        failedTop,
                        barWidth,
                        Math.max(1, completedTop - failedTop - (bucket.completed > 0 ? 2 : 0)),
                      )}
                      fill="var(--critical)"
                    />
                  )}
                </g>
              );
            })}

            <XAxis buckets={buckets} bucketSeconds={bucketSeconds} x={xCenter} width={width} />

            {/* Hit targets: the whole column, bigger than the bar. */}
            {buckets.map((bucket, index) => (
              <rect
                key={bucket.start}
                x={LEFT + band * index}
                y={TOP}
                width={band}
                height={PLOT_HEIGHT}
                fill="transparent"
                tabIndex={0}
                aria-label={`${formatBucket(bucket.start, bucketSeconds)}: ${bucket.completed} completed, ${bucket.failed} failed`}
                onPointerEnter={() => setHover(index)}
                onFocus={() => setHover(index)}
                onBlur={() => setHover(null)}
              />
            ))}
          </svg>
        )}

        {hovered && hover !== null && (
          <div
            className="tooltip"
            style={{
              left: Math.min(Math.max(xCenter(hover), 80), outerWidth - 80),
              top: y(hovered.completed + hovered.failed),
            }}
          >
            <div className="tooltip-title">{formatBucket(hovered.start, bucketSeconds)}</div>
            <div className="tooltip-row">
              <span className="legend-line" style={{ background: "var(--series-1)" }} />
              <strong>{formatCount(hovered.completed)}</strong>
              <span>Completed</span>
            </div>
            <div className="tooltip-row">
              <span className="legend-line" style={{ background: "var(--critical)" }} />
              <strong>{formatCount(hovered.failed)}</strong>
              <span>Failed</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------
// Job latency: p95 line with the SLO target
// ---------------------------------------------------------------------

export function LatencyChart({
  buckets,
  bucketSeconds,
  targetSeconds,
}: {
  buckets: TimeseriesBucket[];
  bucketSeconds: number;
  // The SLO line; not drawn until known.
  targetSeconds?: number;
}) {
  const [ref, outerWidth] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);

  const width = Math.max(0, outerWidth - LEFT - RIGHT_WITH_LABEL);
  const step = buckets.length > 1 ? width / (buckets.length - 1) : 0;
  const x = (index: number) => LEFT + step * index;

  const values = buckets
    .map((b) => b.p95_latency_seconds)
    .filter((v): v is number => v !== null);

  // Always show the SLO line, with headroom above it.
  const ticks = niceTicks(Math.max((targetSeconds ?? 0) * 1.15, ...values, 1));
  const top = ticks[ticks.length - 1];
  const y = (value: number) => TOP + PLOT_HEIGHT - (value / top) * PLOT_HEIGHT;

  // Break the line where a bucket has no data (no jobs finished), rather
  // than drawing it through zero.
  const segments: string[] = [];
  let current: string[] = [];
  const isolated: number[] = [];

  buckets.forEach((bucket, index) => {
    if (bucket.p95_latency_seconds === null) {
      if (current.length === 1) isolated.push(index - 1);
      if (current.length > 1) segments.push(current.join(" "));
      current = [];
      return;
    }
    current.push(`${current.length ? "L" : "M"}${x(index)},${y(bucket.p95_latency_seconds)}`);
  });
  if (current.length === 1) isolated.push(buckets.length - 1);
  if (current.length > 1) segments.push(current.join(" "));

  const onPointerMove = (event: React.PointerEvent<SVGRectElement>) => {
    const box = event.currentTarget.getBoundingClientRect();
    const index = Math.round((event.clientX - box.left) / (step || 1));
    setHover(Math.max(0, Math.min(buckets.length - 1, index)));
  };

  const hovered = hover !== null ? buckets[hover] : null;

  return (
    <div className="chart" ref={ref} onPointerLeave={() => setHover(null)}>
      {width > 0 && (
        <svg
          width={outerWidth}
          height={TOP + PLOT_HEIGHT + AXIS_BAND}
          role="img"
          aria-label={
            targetSeconds !== undefined
              ? `p95 job latency per period, against the ${targetSeconds} second SLO`
              : "p95 job latency per period"
          }
        >
          <YGrid ticks={ticks} y={y} width={width} format={(v) => `${v}s`} />

          {/* The SLO threshold: a threshold, so dashed and labelled. */}
          {targetSeconds !== undefined && (
            <>
              <line
                x1={LEFT}
                x2={LEFT + width}
                y1={y(targetSeconds)}
                y2={y(targetSeconds)}
                stroke="var(--serious)"
                strokeDasharray="4 4"
              />
              <text
                x={LEFT + width + 8}
                y={y(targetSeconds) + 4}
                style={{ fill: "var(--text-secondary)" }}
              >
                SLO {targetSeconds}s
              </text>
            </>
          )}

          {segments.map((d) => (
            <path
              key={d}
              d={d}
              fill="none"
              stroke="var(--series-1)"
              strokeWidth={2}
              strokeLinejoin="round"
              strokeLinecap="round"
            />
          ))}
          {isolated.map((index) => (
            <circle
              key={index}
              cx={x(index)}
              cy={y(buckets[index].p95_latency_seconds!)}
              r={3}
              fill="var(--series-1)"
            />
          ))}

          <XAxis buckets={buckets} bucketSeconds={bucketSeconds} x={x} width={width} />

          {hovered && hover !== null && (
            <g>
              <line
                x1={x(hover)}
                x2={x(hover)}
                y1={TOP}
                y2={TOP + PLOT_HEIGHT}
                stroke="var(--axis)"
              />
              {hovered.p95_latency_seconds !== null && (
                <circle
                  cx={x(hover)}
                  cy={y(hovered.p95_latency_seconds)}
                  r={4.5}
                  fill="var(--series-1)"
                  stroke="var(--surface)"
                  strokeWidth={2}
                />
              )}
            </g>
          )}

          {/* The crosshair finds the X: the whole plot is the target. */}
          <rect
            x={LEFT - step / 2}
            y={TOP}
            width={width + step}
            height={PLOT_HEIGHT}
            fill="transparent"
            onPointerMove={onPointerMove}
          />
        </svg>
      )}

      {hovered && hover !== null && (
        <div
          className="tooltip"
          style={{
            left: Math.min(Math.max(x(hover), 80), outerWidth - 80),
            top: hovered.p95_latency_seconds !== null ? y(hovered.p95_latency_seconds) : TOP + PLOT_HEIGHT / 2,
          }}
        >
          <div className="tooltip-title">{formatBucket(hovered.start, bucketSeconds)}</div>
          <div className="tooltip-row">
            <span className="legend-line" style={{ background: "var(--series-1)" }} />
            <strong>
              {hovered.p95_latency_seconds !== null
                ? formatSeconds(hovered.p95_latency_seconds)
                : "No jobs"}
            </strong>
            <span>p95 latency</span>
          </div>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------
// The table view: every value, without hovering
// ---------------------------------------------------------------------

export function TimeseriesTable({
  buckets,
  bucketSeconds,
}: {
  buckets: TimeseriesBucket[];
  bucketSeconds: number;
}) {
  return (
    <div className="table-scroll">
      <table className="data-table">
        <thead>
          <tr>
            <th scope="col">Period</th>
            <th scope="col">Completed</th>
            <th scope="col">Failed</th>
            <th scope="col">p95 latency</th>
          </tr>
        </thead>
        <tbody>
          {[...buckets].reverse().map((bucket) => (
            <tr key={bucket.start}>
              <td>{formatBucket(bucket.start, bucketSeconds)}</td>
              <td>{formatCount(bucket.completed)}</td>
              <td>{formatCount(bucket.failed)}</td>
              <td>
                {bucket.p95_latency_seconds !== null
                  ? formatSeconds(bucket.p95_latency_seconds)
                  : "–"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
