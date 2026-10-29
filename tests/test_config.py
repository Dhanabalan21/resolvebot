"""Configuration loading."""
from datetime import date
from pathlib import Path

import pytest

from app import config


def test_defaults(monkeypatch):
    for key in ("APP_TODAY", "OPENAI_MODEL", "MAX_TOOL_CALLS", "DB_PATH"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(config, "dotenv_values", lambda path: {})
    settings = config.Settings.from_env()
    assert settings.app_today == date(2026, 7, 1)
    assert settings.openai_model == "gpt-4o-mini"
    assert settings.max_tool_calls == 6
    assert settings.db_path == config.PROJECT_ROOT / "data" / "shopsphere.db"


def test_env_file_read_from_project_root_regardless_of_cwd(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=test-key-from-file\nAPP_TODAY=2026-07-15\n")
    monkeypatch.setattr(config, "ENV_FILE", env_file)
    monkeypatch.setenv("OPENAI_API_KEY", "")  # a blank shell variable must not hide the file value
    monkeypatch.delenv("APP_TODAY", raising=False)
    monkeypatch.chdir(Path("/"))
    settings = config.Settings.from_env()
    assert settings.openai_api_key == "test-key-from-file"
    assert settings.app_today == date(2026, 7, 15)
    assert settings.llm_config_problem() is None


def test_real_environment_wins_over_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_MODEL=from-file\n")
    monkeypatch.setattr(config, "ENV_FILE", env_file)
    monkeypatch.setenv("OPENAI_MODEL", "from-env")
    assert config.Settings.from_env().openai_model == "from-env"


def test_key_never_in_repr(monkeypatch):
    monkeypatch.setattr(config, "dotenv_values", lambda path: {})
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-leak")
    assert "sk-should-not-leak" not in repr(config.Settings.from_env())


def test_missing_key_is_explained(monkeypatch):
    monkeypatch.setattr(config, "dotenv_values", lambda path: {})
    monkeypatch.setenv("OPENAI_API_KEY", "")
    assert "OPENAI_API_KEY" in config.Settings.from_env().llm_config_problem()


def test_invalid_app_today(monkeypatch):
    monkeypatch.setenv("APP_TODAY", "01/07/2026")
    with pytest.raises(config.ConfigError):
        config.Settings.from_env()
