"""Prometheus metrics, served by the API at GET /metrics.

Everything is computed from the database when Prometheus scrapes (see
app/services/prometheus_service.py), so the API is the only target: the
totals survive worker restarts and count every worker's work. All names
start with research_; labels are small fixed sets (status, outcome, tool,
token type).

Counters (only go up; use rate() / increase()):
  research_jobs_created_total                  jobs queued
  research_job_attempts_total{outcome}         finished job attempts
  research_tool_calls_total{tool,outcome}      finished tool calls
  research_agent_iterations_total              LLM calls
  research_llm_tokens_total{type}              input / output tokens
  research_llm_cost_usd_total                  priced runs only

Histograms:
  research_job_attempt_duration_seconds{outcome}
  research_tool_latency_seconds{tool}

Gauges (current state):
  research_jobs{status}
  research_queue_oldest_pending_seconds
  research_jobs_expired_leases
  research_active_workers
  research_agent_runs{status}
  research_agent_runs_unpriced
  research_slo_sli{slo}                        share of good events (0-1)
  research_slo_error_budget_remaining{slo}     0-1 of the 30-day budget
  research_slo_burn_rate{slo}                  last hour; 1 = on budget pace
"""

from prometheus_client import CollectorRegistry, generate_latest
from prometheus_client.core import (
    CounterMetricFamily,
    GaugeMetricFamily,
    HistogramMetricFamily,
)

from app.services.prometheus_service import (
    JOB_DURATION_BUCKETS,
    TOOL_LATENCY_BUCKETS,
)


def _histogram(name, documentation, label, data, buckets):

    family = HistogramMetricFamily(name, documentation, labels=[label])

    for value, histogram in sorted(data.items()):
        family.add_metric(
            [value],
            buckets=[
                *(
                    (str(float(bound)), count)
                    for bound, count in zip(buckets, histogram["buckets"])
                ),
                ("+Inf", histogram["count"]),
            ],
            sum_value=histogram["sum"],
        )

    return family


class _SnapshotCollector:
    """Turns a collect_prometheus_snapshot() dict into metric families."""

    def __init__(self, snapshot: dict):
        self.snapshot = snapshot

    def collect(self):

        snapshot = self.snapshot
        queue = snapshot["queue"]

        # -- Current state ------------------------------------------------

        jobs = GaugeMetricFamily(
            "research_jobs",
            "Jobs in the database, by status.",
            labels=["status"],
        )
        for status, count in queue["jobs"].items():
            jobs.add_metric([status], count)
        yield jobs

        yield GaugeMetricFamily(
            "research_queue_oldest_pending_seconds",
            "How long the oldest pending job has waited (0 if none).",
            value=queue["oldest_pending_seconds"] or 0,
        )

        yield GaugeMetricFamily(
            "research_jobs_expired_leases",
            "Running jobs whose worker's lease has expired (the worker "
            "probably died; recovered on the next poll).",
            value=queue["expired_leases"],
        )

        yield GaugeMetricFamily(
            "research_active_workers",
            "Workers holding a live lease on a running job.",
            value=len(queue["active_workers"]),
        )

        runs = GaugeMetricFamily(
            "research_agent_runs",
            "Agent runs, by current status (a failed run can be resumed, "
            "so this isn't a counter).",
            labels=["status"],
        )
        for status, count in sorted(snapshot["agent_runs"].items()):
            runs.add_metric([status], count)
        yield runs

        yield GaugeMetricFamily(
            "research_agent_runs_unpriced",
            "Agent runs whose model pricing is unknown (left out of "
            "research_llm_cost_usd_total).",
            value=snapshot["unpriced_agent_runs"],
        )

        # -- SLOs and error budgets (app/services/error_budget_service.py)

        budgets = snapshot["error_budgets"]

        for name, documentation, field in (
            (
                "research_slo_sli",
                "Share of good events for each SLO over its 30-day window "
                "(0-1).",
                "actual",
            ),
            (
                "research_slo_error_budget_remaining",
                "Share of each SLO's 30-day error budget left (0-1; 0 when "
                "exhausted).",
                "budget_remaining",
            ),
            (
                "research_slo_burn_rate",
                "How fast each SLO's error budget is being used, from the "
                "last hour (1 = it lasts exactly the window).",
                "burn_rate_1h",
            ),
        ):
            family = GaugeMetricFamily(name, documentation, labels=["slo"])

            for slo, budget in sorted(budgets.items()):
                # No data: no sample, rather than a value that looks
                # healthy.
                if budget[field] is not None:
                    family.add_metric([slo], budget[field])

            yield family

        # -- Activity -----------------------------------------------------

        yield CounterMetricFamily(
            "research_jobs_created",
            "Research jobs created (queued).",
            value=snapshot["jobs_created"],
        )

        attempts = CounterMetricFamily(
            "research_job_attempts",
            "Finished job attempts, by outcome: completed, failed, "
            "lease_expired (the worker died) or released (handed back on "
            "shutdown).",
            labels=["outcome"],
        )
        for outcome, count in sorted(snapshot["job_attempts"].items()):
            attempts.add_metric([outcome], count)
        yield attempts

        yield _histogram(
            "research_job_attempt_duration_seconds",
            "How long finished job attempts ran, by outcome.",
            "outcome",
            snapshot["job_attempt_durations"],
            JOB_DURATION_BUCKETS,
        )

        tool_calls = CounterMetricFamily(
            "research_tool_calls",
            "Finished tool calls (searches), by tool and outcome (success "
            "or failure).",
            labels=["tool", "outcome"],
        )
        for (tool, outcome), count in sorted(snapshot["tool_calls"].items()):
            tool_calls.add_metric([tool, outcome], count)
        yield tool_calls

        yield _histogram(
            "research_tool_latency_seconds",
            "How long tool calls took, including retries, by tool.",
            "tool",
            snapshot["tool_latency"],
            TOOL_LATENCY_BUCKETS,
        )

        yield CounterMetricFamily(
            "research_agent_iterations",
            "LLM calls (agent iterations) made.",
            value=snapshot["agent_iterations"],
        )

        tokens = CounterMetricFamily(
            "research_llm_tokens",
            "LLM tokens used, by type (input or output).",
            labels=["type"],
        )
        for token_type, count in snapshot["llm_tokens"].items():
            tokens.add_metric([token_type], count)
        yield tokens

        yield CounterMetricFamily(
            "research_llm_cost_usd",
            "Estimated LLM cost in USD, for runs whose pricing is known.",
            value=snapshot["llm_cost_usd"],
        )


def render_prometheus_metrics(snapshot: dict) -> bytes:
    """The snapshot in Prometheus text format. A fresh registry each time:
    the values come from the database, nothing is kept between scrapes."""

    registry = CollectorRegistry()
    registry.register(_SnapshotCollector(snapshot))

    return generate_latest(registry)
