"""Shared fixtures. Every test runs against a temporary copy of data/, so the
real assessment database is never modified."""
from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

import pytest

from app import chat as chat_module
from app.config import PROJECT_ROOT, get_settings
from app.llm import AssistantTurn, ToolCall

DATA_DIR = PROJECT_ROOT / "data"


@pytest.fixture
def data_copy(tmp_path, monkeypatch) -> Path:
    target = tmp_path / "data"
    target.mkdir()
    for name in ("shopsphere.db", "policies.md", "manifest.json"):
        shutil.copy(DATA_DIR / name, target / name)
    monkeypatch.setenv("DB_PATH", str(target / "shopsphere.db"))
    monkeypatch.setenv("POLICIES_PATH", str(target / "policies.md"))
    monkeypatch.setenv("MANIFEST_PATH", str(target / "manifest.json"))
    monkeypatch.setenv("APP_TODAY", "2026-07-01")
    monkeypatch.setenv("MAX_TOOL_CALLS", "6")
    get_settings.cache_clear()
    yield target
    get_settings.cache_clear()


@pytest.fixture
def settings(data_copy):
    return get_settings()


@pytest.fixture
def db(data_copy):
    conn = sqlite3.connect(data_copy / "shopsphere.db")
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


def count_writes(conn: sqlite3.Connection) -> tuple[int, int]:
    return (
        conn.execute("SELECT COUNT(*) FROM return_requests").fetchone()[0],
        conn.execute("SELECT COUNT(*) FROM support_tickets").fetchone()[0],
    )


def tool(name: str, call_id: str = "c1", **args) -> AssistantTurn:
    return AssistantTurn(content=None, tool_calls=[ToolCall(id=call_id, name=name, arguments=json.dumps(args))])


def final(answer: str, route: str = "answer", citations=None) -> AssistantTurn:
    return AssistantTurn(content=json.dumps({"answer": answer, "route": route, "citations": citations or []}))


class ScriptedModel:
    """Fake LLM that replays a fixed list of turns and records what it was sent."""

    def __init__(self, turns: list[AssistantTurn]):
        self.turns = list(turns)
        self.calls: list[list[dict]] = []

    def complete(self, messages, tools, timeout):
        self.calls.append(list(messages))
        if not self.turns:
            raise AssertionError("ScriptedModel ran out of turns")
        return self.turns.pop(0)


@pytest.fixture
def script(monkeypatch):
    """script([...turns]) makes the next /chat call use those LLM turns."""
    holder: dict = {}

    def install(turns: list[AssistantTurn]) -> ScriptedModel:
        model = ScriptedModel(turns)
        holder["model"] = model
        return model

    monkeypatch.setattr(chat_module, "build_model", lambda settings: holder["model"])
    return install


@pytest.fixture
def client(data_copy):
    from fastapi.testclient import TestClient

    from main import app

    with TestClient(app) as test_client:
        yield test_client
