"""Business rules (assessment §2.3), written as pure functions.

All numbers come from the ``policy_parameters`` table; all date logic uses the
configured APP_TODAY passed in by the caller, never the system clock.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from dateutil.relativedelta import relativedelta

VALID_REASONS = ("changed_mind", "defective")
VALID_TICKET_TYPES = ("warranty", "cancellation")
ORDER_STATUSES = ("processing", "shipped", "delivered", "cancelled")

ORDER_NOT_FOUND_MESSAGE = "I couldn't find that order on your account."


class PolicyParameterError(RuntimeError):
    """A required policy parameter is missing or not numeric."""


@dataclass
class RuleError(Exception):
    error_code: str
    message: str
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"status": "error", "error_code": self.error_code, "message": self.message, **self.details}


def order_not_found() -> RuleError:
    return RuleError("ORDER_NOT_FOUND", ORDER_NOT_FOUND_MESSAGE)


def int_parameter(params: dict[str, dict], key: str) -> int:
    row = params.get(key)
    if row is None:
        raise PolicyParameterError(f"policy parameter {key!r} is missing")
    value = str(row["value"]).strip()
    if not re.fullmatch(r"\d+", value):
        raise PolicyParameterError(f"policy parameter {key!r} is not a whole number")
    return int(value)


def parse_date(value: str | None) -> date | None:
    """Date part of an ISO date or timestamp; None if missing or malformed."""
    if not value:
        return None
    try:
        return date.fromisoformat(str(value).strip()[:10])
    except ValueError:
        return None


def format_date(value: date | None) -> str | None:
    """'3 July 2026' style, as the assessment asks."""
    return f"{value.day} {value:%B %Y}" if value else None


def delivered_on(order: dict, shipment: dict | None, today: date) -> date | None:
    """Delivery date, or None if the order is not delivered as of ``today``."""
    if str(order.get("status", "")).lower() != "delivered" or not shipment:
        return None
    delivered = parse_date(shipment.get("delivered_at"))
    if delivered is None or delivered > today:
        return None
    return delivered


def warranty_months(warranty_class: str, params: dict[str, dict]) -> int | None:
    cls = str(warranty_class or "").strip().lower()
    if cls == "standard":
        return int_parameter(params, "standard_warranty_months")
    if cls == "extended":
        return int_parameter(params, "extended_warranty_months")
    return None  # 'none' or unknown: no warranty


def warranty_end(delivered: date, months: int) -> date:
    """Same calendar day ``months`` later; relativedelta clamps to month end
    (31 January + 1 month = 28/29 February). The end day itself is covered."""
    return delivered + relativedelta(months=months)


def normalise_reason(reason: object) -> str | None:
    if not isinstance(reason, str):
        return None
    value = re.sub(r"[\s\-]+", "_", reason.strip().lower())
    return value if value in VALID_REASONS else None


def evaluate_return(
    *,
    order: dict,
    items: list[dict],
    shipment: dict | None,
    item_ids: list[str],
    reason: object,
    params: dict[str, dict],
    today: date,
) -> dict:
    """Run the §2.3 return checks in order, stopping at the first failure.

    ``order`` must already be verified as belonging to the customer.
    Raises RuleError on failure; returns eligibility details on success.
    """
    reason_key = normalise_reason(reason)
    if reason_key is None:
        raise RuleError(
            "INVALID_REASON",
            f"Reason must be one of: {', '.join(VALID_REASONS)}.",
        )

    requested = list(dict.fromkeys(i.strip().upper() for i in item_ids if isinstance(i, str) and i.strip()))
    if not requested:
        raise RuleError("INVALID_ITEM", "Please say which item(s) from the order to return.")
    by_id = {item["item_id"]: item for item in items}
    for item_id in requested:
        if item_id not in by_id:
            raise RuleError("INVALID_ITEM", f"Item {item_id} is not in this order")
    selected = [by_id[i] for i in requested]

    # Check 1 (continued): delivered.
    delivered = delivered_on(order, shipment, today)
    if delivered is None:
        raise RuleError(
            "NOT_DELIVERED",
            f"Order {order['order_id']} has not been delivered yet (status: {order.get('status')}).",
        )

    # Check 2: product is returnable at all.
    for item in selected:
        if str(item.get("returnable", "")).lower() == "no":
            raise RuleError(
                "NON_RETURNABLE",
                f"{item['name']} ({item['item_id']}) cannot be returned.",
                {"item_id": item["item_id"]},
            )

    # Check 3: within the return window (day 0 = delivery date, last day inclusive).
    window = int_parameter(params, "return_window_days")
    days = (today - delivered).days
    if days > window:
        raise RuleError(
            "RETURN_WINDOW_EXPIRED",
            f"It has been {days} days since delivery; the return window is {window} days.",
            {"days_since_delivery": days, "return_window_days": window},
        )

    # Check 4: opened + changed_mind + unopened_only product.
    if reason_key == "changed_mind":
        for item in selected:
            if bool(item.get("opened")) and str(item.get("returnable", "")).lower() == "unopened_only":
                raise RuleError(
                    "OPENED_NOT_RETURNABLE",
                    f"{item['name']} ({item['item_id']}) has been opened and can only be "
                    "returned for a change of mind while sealed.",
                    {"item_id": item["item_id"]},
                )

    return {
        "status": "success",
        "eligible": True,
        "order_id": order["order_id"],
        "item_ids": requested,
        "items": [{"item_id": i["item_id"], "name": i["name"]} for i in selected],
        "reason": reason_key,
        "delivered_on": format_date(delivered),
        "days_since_delivery": days,
        "return_window_days": window,
        "last_day_to_return": format_date(delivered + relativedelta(days=window)),
    }


def evaluate_ticket(
    *,
    order: dict,
    items: list[dict],
    shipment: dict | None,
    ticket_type: object,
    params: dict[str, dict],
    today: date,
) -> dict:
    """Decide if a support ticket may be created. Raises RuleError if not."""
    kind = ticket_type.strip().lower() if isinstance(ticket_type, str) else ""
    if kind not in VALID_TICKET_TYPES:
        raise RuleError(
            "INVALID_TICKET_TYPE",
            f"Ticket type must be one of: {', '.join(VALID_TICKET_TYPES)}.",
        )

    status = str(order.get("status", "")).lower()
    if kind == "cancellation":
        if status == "processing":
            return {"status": "success", "ticket_type": "cancellation", "order_status": status}
        if status == "cancelled":
            raise RuleError("ORDER_ALREADY_CANCELLED", f"Order {order['order_id']} is already cancelled.")
        raise RuleError(
            "ORDER_ALREADY_SHIPPED",
            f"Order {order['order_id']} has already been {status}, so it can no longer be cancelled.",
            {"order_status": status},
        )

    delivered = delivered_on(order, shipment, today)
    if delivered is None:
        raise RuleError(
            "NOT_DELIVERED",
            f"Order {order['order_id']} has not been delivered yet, so a warranty claim is not possible.",
        )

    covered, expired = [], []
    for item in items:
        months = warranty_months(item.get("warranty_class"), params)
        if months is None:
            continue
        end = warranty_end(delivered, months)
        entry = {"item_id": item["item_id"], "name": item["name"], "warranty_ends": format_date(end)}
        (covered if today <= end else expired).append(entry)

    if covered:
        return {"status": "success", "ticket_type": "warranty", "covered_items": covered}
    if expired:
        raise RuleError(
            "WARRANTY_EXPIRED",
            "The warranty has ended for every item on this order.",
            {"expired_items": expired},
        )
    raise RuleError("NO_WARRANTY", "None of the items on this order has a warranty.")
