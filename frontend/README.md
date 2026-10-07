# Research Agent frontend

React + TypeScript (Vite) dashboard for the Research Agent API.

```bash
npm install
cp .env.example .env.local   # VITE_API_URL, default http://localhost:8000
npm run dev                  # http://localhost:5173
```

The API must allow this origin: its `CORS_ORIGINS` defaults to
`["http://localhost:5173"]`. If you serve the app elsewhere, add that origin.

`npm run build` type-checks and builds to `dist/`; `npm run lint` runs oxlint.

## Dashboard

| Part | API |
|---|---|
| System health badge (hover for details) | `GET /health` |
| Jobs, Success, Cost tiles | `GET /research/metrics/timeseries`, `GET /research/metrics/system` |
| Job Throughput, Job Latency (p95) charts | `GET /research/metrics/timeseries?range=` |
| Active Jobs, Worker Status | `GET /research/metrics/system?hours=` |

Everything refreshes every 15 seconds; the time range (24 hours, 7 or 30
days) is kept in the URL (`?range=7d`) and scopes every number on the page.

Charts follow a few rules: no dual axes (throughput and latency are separate
charts), every chart has a table view, statuses use an icon and a label
(never color alone), colors are defined once as CSS variables in
`src/index.css` with separate light and dark values, and a period with no
finished jobs shows "No data" rather than 0% or 100%.
