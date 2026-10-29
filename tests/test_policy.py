"""Policy retrieval: document filtering, sections, ranking and the required queries."""
from datetime import date

import pytest

from app.policy import get_index

TODAY = date(2026, 7, 1)


@pytest.fixture
def index(settings):
    return get_index(settings.policies_path, settings.manifest_path)


def _keys(result, field="matches"):
    return [(m["doc_id"], m["section"]) for m in result[field]]


def test_documents_match_manifest(index):
    assert len(index.documents) == 10
    assert index.documents["returns_policy_2024"].status == "superseded"
    assert index.documents["shipping_policy_update"].effective_date == date(2026, 8, 1)


def test_superseded_never_returned(index):
    for query in ["return window", "never returnable", "what is your return policy"]:
        result = index.search(query, TODAY)
        docs = {m["doc_id"] for m in result["matches"] + result["upcoming_changes"]}
        assert "returns_policy_2024" not in docs


def test_future_policy_not_applied_before_effective_date(index):
    result = index.search("What is the shipping fee?", TODAY)
    assert _keys(result)[0] == ("shipping_policy", "§2.1")
    assert all(d != "shipping_policy_update" for d, _ in _keys(result))
    assert ("shipping_policy_update", "§1.1") in _keys(result, "upcoming_changes")
    # From the effective date the update becomes a current match.
    later = index.search("What is the shipping fee?", date(2026, 8, 1))
    assert ("shipping_policy_update", "§1.1") in _keys(later)


@pytest.mark.parametrize("query, expected", [
    ("What is your return policy?", ("returns_policy", "§2.1")),
    ("How many days do I have to return an item?", ("returns_policy", "§2.1")),
    ("Can I return an opened item?", ("returns_policy", "§3.1")),
    ("Can I return opened earbuds if they are defective?", ("returns_policy", "§5.1")),
    ("Can I return opened earbuds if they are defective?", ("returns_policy", "§4.2")),
    ("How long does a refund take?", ("refunds_policy", "§2.1")),
    ("What is the warranty period?", ("warranty_policy", "§2.1")),
    ("What is the cancellation policy?", ("order_cancellation", "§1.1")),
    ("What is the shipping fee?", ("shipping_policy", "§2.1")),
])
def test_required_queries(index, query, expected):
    assert expected in _keys(index.search(query, TODAY))


def test_future_change_query(index):
    result = index.search("What changes after the future policy's effective date?", TODAY)
    upcoming = result["upcoming_changes"]
    assert upcoming and all(m["doc_id"] == "shipping_policy_update" for m in upcoming)
    assert {"§1.1", "§1.2"} <= {m["section"] for m in upcoming}
    assert all(m["effective_date"] == "2026-08-01" for m in upcoming)


def test_policy_sections_listed_before_faq(index):
    matches = index.search("How long does a refund take?", TODAY)["matches"]
    types = [m["doc_type"] for m in matches]
    assert types == sorted(types, key=lambda t: t != "policy")


@pytest.mark.parametrize("query", ["What's the weather in Paris?", "Write me a poem"])
def test_unrelated_queries_return_nothing(index, query):
    result = index.search(query, TODAY)
    assert result["matches"] == [] and result["upcoming_changes"] == []


def test_section_text_is_only_that_section(index):
    match = next(m for m in index.search("standard return window", TODAY)["matches"]
                 if (m["doc_id"], m["section"]) == ("returns_policy", "§2.1"))
    assert "30 days" in match["text"] and "§2.2" not in match["text"]
