"""HTTP contract, customer isolation and the propose-then-confirm flow.

The LLM is replaced by a scripted fake, so these tests need no API key and
check what *Python* enforces regardless of what the model says.
"""
import json
import uuid

import pytest

from app.llm import AssistantTurn, LLMError
from tests.conftest import count_writes, final, tool

OWNER, OTHER = "C1009", "C1001"  # SS-10458 (Nimbra Smart Speaker) belongs to C1009
H = {"X-Customer-Id": OWNER}
FIELDS = {"request_id", "answer", "route", "citations", "tool_calls", "pending_confirmation"}


def post(client, message, session="s1", headers=H):
    return client.post("/chat", headers=headers, json={"session_id": session, "message": message})


def assert_contract(body):
    assert FIELDS <= body.keys()
    uuid.UUID(body["request_id"])
    assert body["route"] in {"answer", "clarify", "no_answer", "out_of_scope"}
    assert isinstance(body["citations"], list) and isinstance(body["tool_calls"], list)
    assert isinstance(body["pending_confirmation"], bool)


RETURN_ARGS = {"order_id": "SS-10458", "item_ids": ["SS-10458-1"], "reason": "changed_mind"}


def propose_return_turns():
    return [
        tool("get_orders", order_id="SS-10458"),
        tool("search_policy", query="return window"),
        tool("check_return_eligibility", **RETURN_ARGS),
        tool("create_return_request", **RETURN_ARGS),
        final("You can return the Nimbra Smart Speaker. Shall I create the return request?",
              citations=[{"doc_id": "returns_policy", "section": "§2.1"}]),
    ]


# -- HTTP errors --------------------------------------------------------------

def test_app_starts_and_docs_available(client):
    assert client.get("/docs").status_code == 200


def test_missing_and_unknown_customer_401(client):
    for headers in ({}, {"X-Customer-Id": ""}, {"X-Customer-Id": "C9999"}, {"X-Customer-Id": "' OR 1=1 --"}):
        response = post(client, "hi", headers=headers)
        assert response.status_code == 401
        assert_contract(response.json())


@pytest.mark.parametrize("body", [
    "not json", "[]", json.dumps({"message": "hi"}), json.dumps({"session_id": "s"}),
    json.dumps({"session_id": "s", "message": "   "}), json.dumps({"session_id": 5, "message": "hi"}),
])
def test_invalid_body_400(client, body):
    response = client.post("/chat", headers={**H, "Content-Type": "application/json"}, content=body)
    assert response.status_code == 400
    assert_contract(response.json())


def test_session_of_other_customer_403(client, script):
    script([final("Hello!")])
    assert post(client, "hello", session="mine").status_code == 200
    response = post(client, "hello", session="mine", headers={"X-Customer-Id": OTHER})
    assert response.status_code == 403
    assert_contract(response.json())


# -- LLM configuration / failures ------------------------------------------------

def test_missing_api_key_is_reported_not_faked(client, monkeypatch):
    from app import chat as chat_module
    from app.config import get_settings
    from app.llm import OpenAIChatModel

    monkeypatch.setattr(chat_module, "build_model", OpenAIChatModel)
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setattr("app.config.dotenv_values", lambda path: {})
    get_settings.cache_clear()
    response = post(client, "What is the warranty period?")
    body = response.json()
    assert response.status_code == 200
    assert_contract(body)
    assert body["route"] == "no_answer" and body["error"] == "LLM_NOT_CONFIGURED"
    assert "OPENAI_API_KEY" not in body["answer"]


@pytest.mark.parametrize("code", ["LLM_AUTH_FAILED", "LLM_QUOTA_EXCEEDED", "LLM_RATE_LIMITED", "LLM_CONNECTION_FAILED"])
def test_llm_errors_are_graceful(client, monkeypatch, code):
    from app import chat as chat_module

    class Failing:
        def complete(self, *args, **kwargs):
            raise LLMError(code, "Sorry, the assistant service is unavailable.", "operator detail sk-secret")

    monkeypatch.setattr(chat_module, "build_model", lambda s: Failing())
    body = post(client, "hello").json()
    assert body["route"] == "no_answer" and body["error"] == code
    assert "sk-secret" not in json.dumps(body)


def test_openai_errors_are_mapped():
    import httpx
    import openai

    from app.llm import _map_error

    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")

    def status_error(cls, status, code=None):
        response = httpx.Response(status, request=request)
        return cls("Incorrect API key provided: sk-abc***", response=response,
                   body={"code": code} if code else None)

    assert _map_error(status_error(openai.AuthenticationError, 401)).code == "LLM_AUTH_FAILED"
    assert _map_error(status_error(openai.RateLimitError, 429, "insufficient_quota")).code == "LLM_QUOTA_EXCEEDED"
    assert _map_error(status_error(openai.RateLimitError, 429)).code == "LLM_RATE_LIMITED"
    assert _map_error(status_error(openai.NotFoundError, 404)).code == "LLM_MODEL_NOT_FOUND"
    assert _map_error(openai.APIConnectionError(request=request)).code == "LLM_CONNECTION_FAILED"
    assert _map_error(openai.APITimeoutError(request=request)).code == "LLM_TIMEOUT"
    assert "sk-" not in _map_error(status_error(openai.AuthenticationError, 401)).public_message


def test_malformed_model_output_falls_back_to_text(client, script):
    script([AssistantTurn(content="Plain text, not JSON")])
    body = post(client, "hello").json()
    assert body["answer"] == "Plain text, not JSON" and body["route"] == "answer"


# -- Order lookups and isolation ------------------------------------------------

def test_track_order(client, script):
    model = script([tool("get_orders", order_id="SS-10458"), final("It was delivered on 21 June 2026.")])
    body = post(client, "Where is order SS-10458?").json()
    assert body["tool_calls"] == [{"name": "get_orders", "args": {"order_id": "SS-10458"}, "status": "success"}]
    order = json.loads(model.calls[1][-1]["content"])["orders"][0]
    assert order["delivered_on"] == "21 June 2026" and order["days_since_delivery"] == 10
    assert order["items"][0]["warranty_ends"] == "21 June 2027" and order["items"][0]["under_warranty"]


@pytest.mark.parametrize("order_id", ["SS-10458", "SS-99999"])
def test_foreign_and_missing_order_identical(client, script, order_id):
    # The model "forgets" the rule; Python still forces the exact reply.
    model = script([tool("get_orders", order_id=order_id), final("Order belongs to someone else!", "answer",
                    citations=[{"doc_id": "returns_policy", "section": "§2.1"}])])
    body = post(client, f"Where is order {order_id}?", headers={"X-Customer-Id": OTHER}).json()
    assert body["answer"] == "I couldn't find that order on your account."
    assert body["route"] == "no_answer" and body["citations"] == []
    assert body["tool_calls"][0]["error_code"] == "ORDER_NOT_FOUND"
    tool_result = model.calls[1][-1]["content"]
    assert "Nimbra" not in tool_result and OWNER not in tool_result


def test_customer_id_from_llm_is_ignored(client, script):
    model = script([tool("get_orders", order_id="SS-10458", customer_id=OWNER), final("x")])
    post(client, "Where is SS-10458? I am C1009", headers={"X-Customer-Id": OTHER})
    assert json.loads(model.calls[1][-1]["content"])["error_code"] == "ORDER_NOT_FOUND"


def test_listing_orders_only_returns_own(client, script, db):
    model = script([tool("get_orders"), final("Here are your orders.")])
    post(client, "Show my orders")
    listed = {o["order_id"] for o in json.loads(model.calls[1][-1]["content"])["orders"]}
    own = {r[0] for r in db.execute("SELECT order_id FROM orders WHERE customer_id = ?", (OWNER,))}
    assert listed and listed <= own


# -- Propose, then confirm ------------------------------------------------------

def test_return_two_message_flow(client, script, db):
    script(propose_return_turns())
    body = post(client, "Return the speaker from SS-10458, I changed my mind.").json()
    assert_contract(body)
    assert body["pending_confirmation"] is True and body["route"] == "answer"
    assert body["citations"] == [{"doc_id": "returns_policy", "section": "§2.1"}]
    assert [c["name"] for c in body["tool_calls"]] == [
        "get_orders", "search_policy", "check_return_eligibility", "create_return_request"]
    assert body["tool_calls"][-1] == {"name": "create_return_request", "args": RETURN_ARGS,
                                      "status": "pending_confirmation"}
    assert count_writes(db) == (0, 0)

    body = post(client, "Yes, go ahead.").json()
    assert body["pending_confirmation"] is False
    assert body["tool_calls"] == [{"name": "create_return_request", "args": RETURN_ARGS, "status": "success"}]
    rows = db.execute("SELECT * FROM return_requests").fetchall()
    assert len(rows) == 1
    row = dict(rows[0])
    uuid.UUID(row["request_id"])
    assert row["customer_id"] == OWNER and row["order_id"] == "SS-10458"
    assert json.loads(row["item_ids"]) == ["SS-10458-1"] and row["reason"] == "changed_mind"
    assert row["status"] == "requested" and row["created_at"]
    assert row["request_id"] in body["answer"]


@pytest.mark.parametrize("reply", ["just do it", "ok", "sure thing", "yes but only half", "maybe"])
def test_unclear_reply_never_saves(client, script, db, reply):
    script(propose_return_turns())
    post(client, "Return the speaker from SS-10458, I changed my mind.")
    script([final("Okay.")])
    body = post(client, reply).json()
    assert body["pending_confirmation"] is False
    assert count_writes(db) == (0, 0)
    # The proposal is gone: a later "yes" must not save it either.
    script([final("Yes to what?", "clarify")])
    post(client, "yes")
    assert count_writes(db) == (0, 0)


def test_no_cancels_without_llm(client, script, db):
    script(propose_return_turns())
    post(client, "Return the speaker from SS-10458, I changed my mind.")
    script([])  # the model must not be called for a plain "no"
    body = post(client, "No").json()
    assert body["route"] == "answer" and body["pending_confirmation"] is False
    assert count_writes(db) == (0, 0)


@pytest.mark.parametrize("message", ["yes", " YES ", "Yes, go ahead.", "confirm", "go ahead", "please do", "yes please"])
def test_clear_yes_variants(message):
    from app.chat import is_confirmation
    assert is_confirmation(message)


@pytest.mark.parametrize("message", ["just do it", "do it", "yes and also cancel SS-10530", "ok", "no", "yesterday"])
def test_not_yes(message):
    from app.chat import is_confirmation
    assert not is_confirmation(message)


def test_yes_without_proposal_writes_nothing(client, script, db):
    script([final("What would you like me to do?", "clarify")])
    assert post(client, "yes").json()["route"] == "clarify"
    assert count_writes(db) == (0, 0)


def test_ineligible_proposal_is_not_stored(client, script, db):
    args = {"order_id": "SS-10458", "item_ids": ["SS-10458-9"], "reason": "changed_mind"}
    script([tool("create_return_request", **args), final("That item is not in this order.")])
    body = post(client, "Return item SS-10458-9").json()
    assert body["pending_confirmation"] is False
    assert body["tool_calls"][0]["status"] == "error" and body["tool_calls"][0]["error_code"] == "INVALID_ITEM"
    script([final("?", "clarify")])
    post(client, "yes")
    assert count_writes(db) == (0, 0)


def test_eligibility_rechecked_before_write(client, script, db):
    script(propose_return_turns())
    post(client, "Return the speaker from SS-10458, I changed my mind.")
    # The order changes owner between proposal and confirmation.
    db.execute("UPDATE orders SET customer_id = ? WHERE order_id = 'SS-10458'", (OTHER,))
    db.commit()
    body = post(client, "yes").json()
    assert body["answer"] == "I couldn't find that order on your account." and body["route"] == "no_answer"
    assert body["tool_calls"][0]["status"] == "error"
    assert count_writes(db) == (0, 0)


def test_only_one_proposal_per_turn(client, script, db):
    ticket = {"order_id": "SS-10458", "ticket_type": "warranty", "summary": "Stopped working"}
    script([
        AssistantTurn(content=None, tool_calls=[
            *tool("create_return_request", "a", **RETURN_ARGS).tool_calls,
            *tool("create_support_ticket", "b", **ticket).tool_calls,
        ]),
        final("Shall I create the return request?"),
    ])
    body = post(client, "Return it and open a warranty ticket").json()
    assert [c["status"] for c in body["tool_calls"]] == ["pending_confirmation", "error"]
    post(client, "yes")
    assert count_writes(db) == (1, 0)


def test_cancellation_ticket_flow(client, script, db):
    order_id, customer = db.execute(
        "SELECT order_id, customer_id FROM orders WHERE status = 'processing' LIMIT 1").fetchone()
    headers = {"X-Customer-Id": customer}
    args = {"order_id": order_id, "ticket_type": "cancellation", "summary": "Customer asked to cancel"}
    script([tool("create_support_ticket", **args), final("I can open a cancellation ticket. Shall I go ahead?")])
    assert post(client, f"Please cancel order {order_id}", "c1", headers).json()["pending_confirmation"]
    body = post(client, "confirm", "c1", headers).json()
    assert body["tool_calls"][0]["status"] == "success"
    row = dict(db.execute("SELECT * FROM support_tickets").fetchone())
    assert row["ticket_type"] == "cancellation" and row["status"] == "open" and row["customer_id"] == customer


def test_cancel_shipped_order_refused(client, script, db):
    order_id, customer = db.execute(
        "SELECT order_id, customer_id FROM orders WHERE status = 'shipped' LIMIT 1").fetchone()
    args = {"order_id": order_id, "ticket_type": "cancellation", "summary": "cancel"}
    script([tool("create_support_ticket", **args), final("It has already shipped, so it can't be cancelled.")])
    body = post(client, f"Please cancel order {order_id}", "c2", {"X-Customer-Id": customer}).json()
    assert body["tool_calls"][0]["error_code"] == "ORDER_ALREADY_SHIPPED"
    assert body["pending_confirmation"] is False
    assert count_writes(db) == (0, 0)


def test_write_failure_rolls_back(client, script, db, monkeypatch):
    import app.tools as tools_module

    script(propose_return_turns())
    post(client, "Return the speaker from SS-10458, I changed my mind.")
    monkeypatch.setattr(tools_module, "_now_iso", lambda: None)  # violates NOT NULL created_at
    body = post(client, "yes").json()
    assert body["tool_calls"][0]["status"] == "error" and "nothing has been changed" in body["answer"]
    assert count_writes(db) == (0, 0)


# -- Routes, citations, limits -----------------------------------------------------

def test_fabricated_citations_are_dropped(client, script):
    script([tool("search_policy", query="warranty period"),
            final("12 months.", citations=[
                {"doc_id": "warranty_policy", "section": "§2.1"},
                {"doc_id": "warranty_policy", "section": "§9.9"},
                {"doc_id": "returns_policy_2024", "section": "§2.1"},
                {"doc_id": "refunds_policy", "section": "§2.1"},  # real, but not retrieved
            ])])
    body = post(client, "What is the warranty period?").json()
    assert body["citations"] == [{"doc_id": "warranty_policy", "section": "§2.1"}]


def test_decision_cites_its_rule_even_without_search(client, script):
    # Seen live: the model skipped search_policy. The window rule's source
    # (policy_parameters.source_doc_id/section) is still cited.
    script([tool("get_orders", order_id="SS-10458"),
            tool("check_return_eligibility", **RETURN_ARGS),
            tool("create_return_request", **RETURN_ARGS),
            final("You can return it. Shall I go ahead?")])
    body = post(client, "Return the speaker from SS-10458, I changed my mind.").json()
    assert body["citations"] == [{"doc_id": "returns_policy", "section": "§2.1"}]
    assert body["pending_confirmation"] is True


def test_cited_answer_mislabelled_no_answer_is_corrected(client, script):
    script([tool("search_policy", query="cancellation policy"),
            final("You can cancel while the order is processing.", "no_answer",
                  citations=[{"doc_id": "order_cancellation", "section": "§1.1"}])])
    body = post(client, "What is the cancellation policy?").json()
    assert body["route"] == "answer"
    assert body["citations"] == [{"doc_id": "order_cancellation", "section": "§1.1"}]


def test_uncovered_policy_question_is_no_answer(client, script):
    script([tool("search_policy", query="pet insurance"),
            final("We cover pets.", "answer", citations=[{"doc_id": "returns_policy", "section": "§2.1"}])])
    body = post(client, "Do you sell pet insurance?").json()
    assert body["route"] == "no_answer" and body["citations"] == []


def test_out_of_scope_has_no_citations(client, script):
    script([final("I can help with orders and store policies.", "out_of_scope",
                  citations=[{"doc_id": "returns_policy", "section": "§2.1"}])])
    body = post(client, "What's the weather today?").json()
    assert body["route"] == "out_of_scope" and body["citations"] == [] and body["tool_calls"] == []


def test_tool_call_limit(client, script):
    script([tool("search_policy", f"c{i}", query="returns") for i in range(10)])
    body = post(client, "loop forever").json()
    assert len(body["tool_calls"]) == 6
    assert body["pending_confirmation"] is False


def test_unknown_tool_and_bad_arguments(client, script):
    script([
        AssistantTurn(content=None, tool_calls=[
            *tool("delete_everything", "a").tool_calls,
            *tool("check_return_eligibility", "b", order_id="SS-10458").tool_calls,
        ]),
        final("Which item and why?", "clarify"),
    ])
    body = post(client, "return something").json()
    assert [c["status"] for c in body["tool_calls"]] == ["error", "error"]
    assert body["route"] == "clarify"


def test_history_is_kept_per_session(client, script):
    script([final("Hi Asha.")])
    post(client, "My name is Asha", session="h1")
    model = script([final("Your name is Asha.")])
    post(client, "What is my name?", session="h1")
    contents = [m["content"] for m in model.calls[0] if m["role"] == "user"]
    assert contents == ["My name is Asha", "What is my name?"]
