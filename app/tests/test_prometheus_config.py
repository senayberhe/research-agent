from pathlib import Path


# From this file (app/tests/) up to the project root, so the tests pass
# whichever directory pytest is run from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_prometheus_config_exists():
    config_path = (
        PROJECT_ROOT / "monitoring/prometheus.yml"
    )

    assert config_path.exists()


def test_prometheus_config_contains_api_target():
    config_path = (
        PROJECT_ROOT / "monitoring/prometheus.yml"
    )

    content = config_path.read_text()

    assert "research-api" in content
    assert "api:8000" in content
    assert "/metrics" in content


# -------------------------
# Grafana
# -------------------------


import json
import re

import yaml

from app.core.prometheus import render_prometheus_metrics


GRAFANA = PROJECT_ROOT / "monitoring/grafana"
DASHBOARD = GRAFANA / "dashboards/research-agent.json"


def test_grafana_files_exist():

    for path in (
        "provisioning/datasources/datasource.yml",
        "provisioning/dashboards/dashboard.yml",
        "dashboards/research-agent.json",
    ):
        assert (GRAFANA / path).exists(), path


def test_grafana_datasource_points_at_prometheus():

    config = yaml.safe_load(
        (GRAFANA / "provisioning/datasources/datasource.yml").read_text()
    )

    [datasource] = config["datasources"]

    assert datasource["type"] == "prometheus"
    assert datasource["url"] == "http://prometheus:9090"
    assert datasource["uid"] == "prometheus"


def dashboard_queries() -> list[str]:
    dashboard = json.loads(DASHBOARD.read_text())

    return [
        target["expr"]
        for panel in dashboard["panels"]
        for target in panel.get("targets", [])
    ]


def test_dashboard_uses_the_provisioned_datasource():

    dashboard = json.loads(DASHBOARD.read_text())

    for panel in dashboard["panels"]:
        if panel["type"] == "row":
            continue

        assert panel["datasource"]["uid"] == "prometheus", panel["title"]


def exported_metric_names() -> set[str]:
    """Every metric GET /metrics serves (declared even with no data)."""

    empty = {
        "queue": {
            "jobs": {},
            "oldest_pending_seconds": None,
            "expired_leases": 0,
            "active_workers": [],
        },
        "jobs_created": 0,
        "job_attempts": {},
        "job_attempt_durations": {},
        "tool_calls": {},
        "tool_latency": {},
        "agent_runs": {},
        "agent_iterations": 0,
        "llm_tokens": {},
        "llm_cost_usd": 0.0,
        "unpriced_agent_runs": 0,
        "error_budgets": {},
    }

    text = render_prometheus_metrics(empty).decode()

    return set(re.findall(r"^# TYPE (\S+) ", text, re.MULTILINE))


def test_dashboard_only_queries_exported_metrics():
    """A renamed metric would leave a panel silently empty."""

    exported = exported_metric_names()

    for expr in dashboard_queries():
        for name in re.findall(r"\bresearch_\w+", expr):
            base = re.sub(r"_(bucket|sum|count)$", "", name)

            assert base in exported, f"{name} in: {expr}"


# -------------------------
# Alert rules
# -------------------------


ALERTS = PROJECT_ROOT / "monitoring/alerts/research-agent-alerts.yml"


def alert_rules() -> list[dict]:
    config = yaml.safe_load(ALERTS.read_text())

    return [
        rule
        for group in config["groups"]
        for rule in group["rules"]
    ]


def test_prometheus_loads_the_alert_rules():

    config = yaml.safe_load(
        (PROJECT_ROOT / "monitoring/prometheus.yml").read_text()
    )

    assert "/etc/prometheus/alerts/*.yml" in config["rule_files"]


def test_alert_rules_are_complete():

    rules = alert_rules()

    assert len(rules) == 12

    for rule in rules:
        assert rule["labels"]["severity"] in ("warning", "critical")
        assert rule["labels"]["service"] == "research-agent"
        assert rule["annotations"]["summary"]
        assert rule["annotations"]["description"]


def test_alert_rules_only_query_exported_metrics():
    """A rule over a metric that doesn't exist never fires."""

    exported = exported_metric_names()

    for rule in alert_rules():
        for name in re.findall(r"\bresearch_\w+", rule["expr"]):
            base = re.sub(r"_(bucket|sum|count)$", "", name)

            assert base in exported, f"{rule['alert']}: {name}"


# -------------------------
# Alertmanager
# -------------------------


ALERTMANAGER = PROJECT_ROOT / "monitoring/alertmanager"


def test_prometheus_sends_alerts_to_alertmanager():

    config = yaml.safe_load(
        (PROJECT_ROOT / "monitoring/prometheus.yml").read_text()
    )

    [alertmanagers] = config["alerting"]["alertmanagers"]

    assert alertmanagers["static_configs"][0]["targets"] == [
        "alertmanager:9093"
    ]


def test_alertmanager_config_keeps_credentials_out():
    """Addresses and logins are placeholders filled in from
    .env.alertmanager; the password is read from a file."""

    text = (ALERTMANAGER / "alertmanager.yml").read_text()

    for placeholder in (
        "${SMTP_SMARTHOST}",
        "${SMTP_FROM}",
        "${SMTP_USERNAME}",
        "${SMTP_REQUIRE_TLS}",
        "${ALERT_EMAIL_TO}",
    ):
        assert placeholder in text

    config = yaml.safe_load(text)

    assert config["global"]["smtp_auth_password_file"] == (
        "/etc/alertmanager/secrets/smtp_password"
    )
    assert "smtp_auth_password" not in config["global"]

    # No literal email address anywhere in the tracked config.
    assert not re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text)


def test_alertmanager_routes_to_email_with_resolved_notices():

    config = yaml.safe_load(
        (ALERTMANAGER / "alertmanager.yml").read_text()
    )

    assert config["route"]["receiver"] == "email"

    [receiver] = config["receivers"]
    [email] = receiver["email_configs"]

    assert email["send_resolved"] is True


def test_credentials_are_ignored_by_git():

    gitignore = (PROJECT_ROOT / ".gitignore").read_text()

    # .env.alertmanager falls under .env.*; only the example is kept.
    assert ".env.*" in gitignore
    assert "!.env.alertmanager.example" in gitignore
    assert "monitoring/alertmanager/secrets/*" in gitignore

    assert (PROJECT_ROOT / ".env.alertmanager.example").exists()
