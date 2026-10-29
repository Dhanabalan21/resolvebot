"""Business rules (§2.3) on synthetic data, including the PDF's example table."""
from datetime import date, timedelta

import pytest

from app.rules import RuleError, evaluate_return, evaluate_ticket, warranty_end

TODAY = date(2026, 7, 1)
PARAMS = {
    "return_window_days": {"value": "30"},
    "standard_warranty_months": {"value": "12"},
    "extended_warranty_months": {"value": "24"},
}


def _case(product: dict, *, opened: bool, days: int, reason, params=PARAMS, status="delivered"):
    delivered = TODAY - timedelta(days=days)
    order = {"order_id": "SS-1", "status": status}
    items = [{"item_id": "SS-1-1", "opened": int(opened), **product}]
    shipment = {"delivered_at": f"{delivered.isoformat()}T10:00:00"}
    return evaluate_return(order=order, items=items, shipment=shipment, item_ids=["SS-1-1"],
                           reason=reason, params=params, today=TODAY)


SPEAKER = {"name": "Speaker", "returnable": "yes", "warranty_class": "standard"}
EARBUDS = {"name": "Earbuds", "returnable": "unopened_only", "warranty_class": "standard"}
GIFT_CARD = {"name": "Gift card", "returnable": "no", "warranty_class": "none"}


@pytest.mark.parametrize("product, opened, days, reason, expected", [
    (SPEAKER, False, 10, "changed_mind", None),
    (SPEAKER, True, 30, "changed_mind", None),
    (SPEAKER, False, 31, "changed_mind", "RETURN_WINDOW_EXPIRED"),
    (EARBUDS, True, 5, "changed_mind", "OPENED_NOT_RETURNABLE"),
    (EARBUDS, True, 5, "defective", None),
    (GIFT_CARD, False, 2, "changed_mind", "NON_RETURNABLE"),
    (EARBUDS, False, 5, "changed_mind", None),
    (SPEAKER, False, 0, "changed_mind", None),
])
def test_pdf_return_examples(product, opened, days, reason, expected):
    if expected is None:
        result = _case(product, opened=opened, days=days, reason=reason)
        assert result["eligible"] is True and result["days_since_delivery"] == days
    else:
        with pytest.raises(RuleError) as err:
            _case(product, opened=opened, days=days, reason=reason)
        assert err.value.error_code == expected


def test_window_comes_from_parameters_not_code():
    params = {**PARAMS, "return_window_days": {"value": "14"}}
    assert _case(SPEAKER, opened=False, days=14, reason="changed_mind", params=params)["eligible"]
    with pytest.raises(RuleError, match="14 days"):
        _case(SPEAKER, opened=False, days=15, reason="changed_mind", params=params)


def test_check_order_stops_at_first_failure():
    # Gift card outside the window: check 2 (NON_RETURNABLE) comes before check 3.
    with pytest.raises(RuleError) as err:
        _case(GIFT_CARD, opened=False, days=90, reason="changed_mind")
    assert err.value.error_code == "NON_RETURNABLE"
    # Opened earbuds outside the window: check 3 comes before check 4.
    with pytest.raises(RuleError) as err:
        _case(EARBUDS, opened=True, days=40, reason="changed_mind")
    assert err.value.error_code == "RETURN_WINDOW_EXPIRED"


@pytest.mark.parametrize("reason", ["bored", "", None, "damaged"])
def test_invalid_reason(reason):
    with pytest.raises(RuleError) as err:
        _case(SPEAKER, opened=False, days=1, reason=reason)
    assert err.value.error_code == "INVALID_REASON"


def test_reason_normalisation():
    assert _case(SPEAKER, opened=False, days=1, reason=" Changed-Mind ")["reason"] == "changed_mind"


def test_not_delivered():
    with pytest.raises(RuleError) as err:
        _case(SPEAKER, opened=False, days=1, reason="changed_mind", status="shipped")
    assert err.value.error_code == "NOT_DELIVERED"


def test_invalid_item():
    order = {"order_id": "SS-1", "status": "delivered"}
    with pytest.raises(RuleError) as err:
        evaluate_return(order=order, items=[{"item_id": "SS-1-1", **SPEAKER, "opened": 0}],
                        shipment={"delivered_at": "2026-06-30T10:00:00"}, item_ids=["SS-1-9"],
                        reason="changed_mind", params=PARAMS, today=TODAY)
    assert err.value.error_code == "INVALID_ITEM"
    assert err.value.message == "Item SS-1-9 is not in this order"


@pytest.mark.parametrize("delivered, months, expected", [
    (date(2026, 3, 15), 12, date(2027, 3, 15)),
    (date(2026, 1, 31), 1, date(2026, 2, 28)),
    (date(2024, 1, 31), 1, date(2024, 2, 29)),
])
def test_warranty_end(delivered, months, expected):
    assert warranty_end(delivered, months) == expected


def _ticket(items, delivered: date | None, ticket_type="warranty", status="delivered"):
    shipment = {"delivered_at": f"{delivered.isoformat()}T09:00:00"} if delivered else None
    return evaluate_ticket(order={"order_id": "SS-1", "status": status}, items=items, shipment=shipment,
                           ticket_type=ticket_type, params=PARAMS, today=TODAY)


def test_warranty_last_day_is_covered():
    items = [{"item_id": "SS-1-1", "name": "Phone", "warranty_class": "standard"}]
    assert _ticket(items, date(2025, 7, 1))["ticket_type"] == "warranty"
    with pytest.raises(RuleError) as err:
        _ticket(items, date(2025, 6, 30))
    assert err.value.error_code == "WARRANTY_EXPIRED"


def test_extended_warranty_and_no_warranty():
    laptop = [{"item_id": "SS-1-1", "name": "Laptop", "warranty_class": "extended"}]
    assert _ticket(laptop, date(2025, 1, 1))["covered_items"][0]["item_id"] == "SS-1-1"
    gift = [{"item_id": "SS-1-1", "name": "Gift card", "warranty_class": "none"}]
    with pytest.raises(RuleError) as err:
        _ticket(gift, date(2026, 6, 1))
    assert err.value.error_code == "NO_WARRANTY"


def test_warranty_needs_delivery():
    items = [{"item_id": "SS-1-1", "name": "Phone", "warranty_class": "standard"}]
    with pytest.raises(RuleError) as err:
        _ticket(items, None, status="shipped")
    assert err.value.error_code == "NOT_DELIVERED"


@pytest.mark.parametrize("status, code", [
    ("processing", None), ("shipped", "ORDER_ALREADY_SHIPPED"), ("delivered", "ORDER_ALREADY_SHIPPED"),
])
def test_cancellation(status, code):
    if code is None:
        assert _ticket([], None, "cancellation", status)["ticket_type"] == "cancellation"
    else:
        with pytest.raises(RuleError) as err:
            _ticket([], None, "cancellation", status)
        assert err.value.error_code == code


def test_invalid_ticket_type():
    with pytest.raises(RuleError) as err:
        _ticket([], None, "refund", "processing")
    assert err.value.error_code == "INVALID_TICKET_TYPE"
