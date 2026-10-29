"""Environment-based configuration.

The ``.env`` file is always read from the project root, no matter which
directory the server is started from. Real environment variables win over
``.env`` values, except that a variable set to an empty string is treated as
unset (an empty ``OPENAI_API_KEY=`` exported in a shell must not hide the key
stored in ``.env``).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from pathlib import Path

from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

DEFAULT_APP_TODAY = "2026-07-01"
DEFAULT_MODEL = "gpt-4o-mini"
SUPPORTED_PROVIDERS = ("openai",)


class ConfigError(RuntimeError):
    """Raised when the configuration is unusable. Never contains secret values."""


def _read_env() -> dict[str, str]:
    values: dict[str, str] = {
        k: v for k, v in dotenv_values(ENV_FILE).items() if v is not None
    }
    for key, value in os.environ.items():
        if value.strip():
            values[key] = value
    return values


def _resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


@dataclass(frozen=True)
class Settings:
    policies_path: Path
    manifest_path: Path
    db_path: Path
    app_today: date
    llm_provider: str
    openai_model: str
    max_tool_calls: int
    # repr=False keeps the key out of logs and tracebacks.
    openai_api_key: str = field(default="", repr=False)

    def llm_config_problem(self) -> str | None:
        """Operator-facing description of what is missing, or None if usable."""
        if self.llm_provider not in SUPPORTED_PROVIDERS:
            return (
                f"LLM_PROVIDER={self.llm_provider!r} is not supported "
                f"(supported: {', '.join(SUPPORTED_PROVIDERS)})."
            )
        if not self.openai_api_key:
            return (
                "OPENAI_API_KEY is not set. Add it to the .env file in the project "
                "root (or export it) and restart the server."
            )
        return None

    def validate_paths(self) -> None:
        for label, path in (
            ("POLICIES_PATH", self.policies_path),
            ("MANIFEST_PATH", self.manifest_path),
            ("DB_PATH", self.db_path),
        ):
            if not path.is_file():
                raise ConfigError(f"{label} does not point to a file: {path}")

    @classmethod
    def from_env(cls) -> "Settings":
        env = _read_env()

        today_raw = env.get("APP_TODAY", DEFAULT_APP_TODAY).strip()
        try:
            app_today = date.fromisoformat(today_raw)
        except ValueError as exc:
            raise ConfigError(
                f"APP_TODAY must be an ISO date (YYYY-MM-DD), got {today_raw!r}"
            ) from exc

        try:
            max_calls = int(env.get("MAX_TOOL_CALLS", "6"))
        except ValueError as exc:
            raise ConfigError("MAX_TOOL_CALLS must be an integer") from exc

        return cls(
            policies_path=_resolve_path(env.get("POLICIES_PATH", "./data/policies.md")),
            manifest_path=_resolve_path(env.get("MANIFEST_PATH", "./data/manifest.json")),
            db_path=_resolve_path(env.get("DB_PATH", "./data/shopsphere.db")),
            app_today=app_today,
            llm_provider=env.get("LLM_PROVIDER", "openai").strip().lower(),
            openai_model=env.get("OPENAI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
            max_tool_calls=max(1, max_calls),
            openai_api_key=env.get("OPENAI_API_KEY", "").strip(),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
