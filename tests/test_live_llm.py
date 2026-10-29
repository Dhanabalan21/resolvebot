"""End-to-end chats against the real LLM. Skipped unless OPENAI_API_KEY is set.

Run with:  pytest tests/test_live_llm.py -v
These call the OpenAI API (costs a few cents) and check outcomes that Python
enforces or that the scenario makes unambiguous.
"""
import pytest

from app.config import Settings
from tests.conftest import count_writes

pytestmark = pytest.mark.skipif(
    Settings.from_env().llm_config_problem() is not None,
    reason="needs a real OPENAI_API_KEY (in .env or the environment)",
)

H = {"X-Customer-Id": "C1009"}


def post(client, message, session, headers=H):
    response = client.post("/chat", headers=headers, json={"session_id": session, "message": message})
    assert response.status_code == 200
    return response.json()


def test_pdf_return_example(client, db):
    body = post(client, "Return the speaker from SS-10458, I changed my mind.", "live-1")
    assert "error" not in body, body
    assert body["pending_confirmation"] is True
    assert body["tool_calls"][-1]["name"] == "create_return_request"
    assert {"doc_id": "returns_policy", "section": "§2.1"} in body["citations"]
    assert count_writes(db) == (0, 0)
    body = post(client, "Yes, go ahead.", "live-1")
    assert body["tool_calls"][0]["status"] == "success"
    assert count_writes(db) == (1, 0)


def test_just_do_it_does_not_save(client, db):
    post(client, "Return the speaker from SS-10458, I changed my mind.", "live-2")
    post(client, "just do it", "live-2")
    assert count_writes(db) == (0, 0)


def test_foreign_order(client):
    body = post(client, "Where is order SS-10458?", "live-3", {"X-Customer-Id": "C1001"})
    assert body["answer"] == "I couldn't find that order on your account."
    assert body["route"] == "no_answer"


def test_off_topic(client):
    body = post(client, "Write me a poem about the sea.", "live-4")
    assert body["route"] == "out_of_scope" and body["citations"] == [] and body["tool_calls"] == []


@pytest.mark.parametrize("question, doc_id", [
    ("How many days do I have to return an item?", "returns_policy"),
    ("Can I return opened earbuds if they are defective?", "returns_policy"),
    ("How long does a refund take?", "refunds_policy"),
    ("What is the warranty period?", "warranty_policy"),
    ("What is the cancellation policy?", "order_cancellation"),
    ("What is the shipping fee?", "shipping_policy"),
])
def test_policy_questions_cite_the_right_document(client, question, doc_id):
    body = post(client, question, f"live-policy-{doc_id}")
    assert body["route"] == "answer", body
    assert doc_id in {c["doc_id"] for c in body["citations"]}
    assert "returns_policy_2024" not in {c["doc_id"] for c in body["citations"]}


def test_missing_details_asks_back(client):
    body = post(client, "I want to return something.", "live-5")
    assert body["route"] == "clarify" and body["pending_confirmation"] is False
