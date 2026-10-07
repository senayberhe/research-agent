from app.core.config import settings


def test_settings_loaded():
    assert settings.database_url
    assert settings.tavily_api_key
    assert settings.openai_api_key
    assert settings.environment in{
        "development",
        "production",
        "testing",
    }
    assert settings.log_level




def test_agent_limit_defaults():
    assert settings.agent_max_iterations >= 1
    assert settings.agent_max_tool_calls >= 1
    assert settings.agent_tool_timeout_seconds > 0
    assert settings.agent_max_tool_retries >= 0
    assert settings.agent_max_query_length >= 1
    assert settings.agent_max_result_length >= 1


def test_worker_settings():
    assert settings.worker_poll_interval_seconds > 0
    assert settings.worker_lease_seconds >= 1
    assert settings.worker_max_attempts >= 1


def test_worker_settings_are_validated(monkeypatch):

    import pytest
    from pydantic import ValidationError

    from app.core.config import Settings

    # A lease of 0 seconds or no attempts at all would make no sense.
    monkeypatch.setenv("WORKER_LEASE_SECONDS", "0")

    with pytest.raises(ValidationError, match="worker_lease_seconds"):
        Settings()

    monkeypatch.setenv("WORKER_LEASE_SECONDS", "300")
    monkeypatch.setenv("WORKER_MAX_ATTEMPTS", "0")

    with pytest.raises(ValidationError, match="worker_max_attempts"):
        Settings()
