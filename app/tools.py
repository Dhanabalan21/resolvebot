"""The five assistant tools (assessment §2.2).

A ``ToolRunner`` is bound to the customer from the ``X-Customer-Id`` header;
the LLM never supplies a customer ID. Read tools run immediately. The two write
tools are split in two: ``propose`` validates without writing (used when the
LLM asks for the tool) and ``create_*`` re-checks and writes in one transaction
(used only after the customer says yes; see app/chat.py).
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Callable, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.config import Settings
from app.db import (
    connection,
    fetch_customer_order,
    fetch_customer_orders,
    fetch_order_items,
    fetch_policy_parameters,
    fetch_shipment,
)
from app.policy import get_index
from app.rules import (
    ORDER_STATUSES,
    PolicyParameterError,
    RuleError,
    delivered_on,
    evaluate_return,
    evaluate_ticket,
    format_date,
    int_parameter,
    order_not_found,
    parse_date,
    warranty_end,
    warranty_months,
)

logger = logging.getLogger("resolvebot.tools")

READ_TOOLS = ("get_orders", "search_policy", "check_return_eligibility")
WRITE_TOOLS = ("create_return_request", "create_support_ticket")
TOOL_NAMES = READ_TOOLS + WRITE_TOOLS


# ---------------------------------------------------------------------------
# Input schemas. Extra keys (e.g. a "customer_id" the LLM invents) are ignored.
# ---------------------------------------------------------------------------

class _Args(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)


class GetOrdersArgs(_Args):
    order_id: Optional[str] = Field(default=None, max_length=40)
    status: Optional[str] = None

    @field_validator("order_id")
    @classmethod
    def _upper(cls, v: Optional[str]) -> Optional[str]:
        return v.upper() if v else None

    @field_validator("status")
    @classmethod
    def _status(cls, v: Optional[str]) -> Optional[str]:
        if not v:
            return None
        v = v.lower()
        if v not in ORDER_STATUSES:
            raise ValueError(f"status must be one of {', '.join(ORDER_STATUSES)}")
        return v


class SearchPolicyArgs(_Args):
    query: str = Field(min_length=1, max_length=500)


class ReturnArgs(_Args):
    order_id: str = Field(min_length=1, max_length=40)
    item_ids: list[str] = Field(max_length=50)
    reason: str = Field(max_length=40)

    @field_validator("order_id")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()

    @field_validator("item_ids")
    @classmethod
    def _items(cls, v: list[str]) -> list[str]:
        return list(dict.fromkeys(i.strip().upper() for i in v if i.strip()))


class TicketArgs(_Args):
    order_id: str = Field(min_length=1, max_length=40)
    ticket_type: str = Field(max_length=40)
    summary: str = Field(default="", max_length=1000)

    @field_validator("order_id")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()


ARG_MODELS: dict[str, type[_Args]] = {
    "get_orders": GetOrdersArgs,
    "search_policy": SearchPolicyArgs,
    "check_return_eligibility": ReturnArgs,
    "create_return_request": ReturnArgs,
    "create_support_ticket": TicketArgs,
}


def _validation_error(name: str, exc: ValidationError) -> RuleError:
    fields = {str(e["loc"][0]) for e in exc.errors() if e.get("loc")}
    if "order_id" in fields:
        return order_not_found()
    if "reason" in fields:
        return RuleError("INVALID_REASON", "Reason must be changed_mind or defective.")
    if "item_ids" in fields:
        return RuleError("INVALID_ITEM", "item_ids must be a list of item IDs from the order.")
    if "ticket_type" in fields:
        return RuleError("INVALID_TICKET_TYPE", "Ticket type must be warranty or cancellation.")
    return RuleError("INVALID_ARGUMENTS", f"Invalid arguments for {name}: {', '.join(sorted(fields)) or 'unknown'}.")


def parse_args(name: str, raw: object) -> tuple[_Args | None, dict, RuleError | None]:
    """Validate raw LLM arguments. Returns (model, args_for_log, error)."""
    model_cls = ARG_MODELS[name]
    if not isinstance(raw, dict):
        return None, {}, RuleError("INVALID_ARGUMENTS", f"Arguments for {name} must be a JSON object.")
    shown = {k: v for k, v in raw.items() if k in model_cls.model_fields}
    try:
        model = model_cls.model_validate(raw)
    except ValidationError as exc:
        return None, shown, _validation_error(name, exc)
    return model, model.model_dump(exclude_none=True), None


# ---------------------------------------------------------------------------
# OpenAI tool definitions
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_orders",
            "description": (
                "Look up the logged-in customer's orders. Pass order_id for one order, or omit it "
                "to list recent orders (optionally filtered by status). Returns status, items "
                "(item_id, name, opened, returnable, warranty end date), shipment and ETA, the "
                "delivery date and days since delivery. Use it to find item IDs before checking a return."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "Order ID such as SS-10458."},
                    "status": {"type": "string", "enum": list(ORDER_STATUSES)},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_policy",
            "description": (
                "Search ShopSphere's current policy documents. Returns the best matching sections "
                "with doc_id and section number to cite, plus any scheduled future changes. Use "
                "for every policy question and before explaining a return or warranty decision."
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Short search phrase, e.g. 'return window opened earbuds'."}},
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_return_eligibility",
            "description": (
                "Check (read-only) whether items from an order can be returned. Returns "
                "eligible=true or an error_code such as NON_RETURNABLE, RETURN_WINDOW_EXPIRED, "
                "OPENED_NOT_RETURNABLE, NOT_DELIVERED, INVALID_ITEM."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "item_ids": {"type": "array", "items": {"type": "string"}, "description": "Item IDs such as SS-10458-1."},
                    "reason": {"type": "string", "enum": ["changed_mind", "defective"],
                               "description": "changed_mind = no longer wanted; defective = broken, damaged or wrong item."},
                },
                "required": ["order_id", "item_ids", "reason"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_return_request",
            "description": (
                "Propose a return request. Call only when the customer asks for the return to be "
                "made. Nothing is saved now: the system holds it as a proposal and saves it only "
                "if the customer's next message is a clear yes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "item_ids": {"type": "array", "items": {"type": "string"}},
                    "reason": {"type": "string", "enum": ["changed_mind", "defective"]},
                },
                "required": ["order_id", "item_ids", "reason"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_support_ticket",
            "description": (
                "Propose a support ticket: 'warranty' (delivered order with an item still under "
                "warranty) or 'cancellation' (order still processing). Call only when the customer "
                "asks for it. Nothing is saved now: it is saved only if the customer's next message "
                "is a clear yes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "ticket_type": {"type": "string", "enum": ["warranty", "cancellation"]},
                    "summary": {"type": "string", "description": "One-sentence description of the problem or request."},
                },
                "required": ["order_id", "ticket_type", "summary"],
                "additionalProperties": False,
            },
        },
    },
]


def _now_iso() -> str:
    # Real time for audit timestamps only; business dates use APP_TODAY.
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ToolRunner:
    """Tool implementations bound to one authenticated customer."""

    def __init__(self, customer_id: str, settings: Settings):
        self.customer_id = customer_id
        self.settings = settings
        self.today = settings.app_today

    # -- dispatch -----------------------------------------------------------
    def run_read_tool(self, name: str, args: _Args) -> dict:
        handlers: dict[str, Callable[[], dict]] = {
            "get_orders": lambda: self.get_orders(args.order_id, args.status),
            "search_policy": lambda: self.search_policy(args.query),
            "check_return_eligibility": lambda: self.check_return_eligibility(args.order_id, args.item_ids, args.reason),
        }
        return self._safely(name, handlers[name])

    def propose(self, name: str, args: _Args) -> dict:
        """Validate a write tool call without writing anything."""
        if name == "create_return_request":
            return self._safely(name, lambda: self.check_return_eligibility(args.order_id, args.item_ids, args.reason))

        def ticket() -> dict:
            with connection() as conn:
                return self._evaluate_ticket(conn, args.order_id, args.ticket_type, args.summary)
        return self._safely(name, ticket)

    def execute_write(self, name: str, args: dict) -> dict:
        """Run a confirmed write. Called only after the customer's yes."""
        if name == "create_return_request":
            return self._safely(name, lambda: self.create_return_request(args["order_id"], args["item_ids"], args["reason"]))
        if name == "create_support_ticket":
            return self._safely(name, lambda: self.create_support_ticket(args["order_id"], args["ticket_type"], args["summary"]))
        return RuleError("UNKNOWN_TOOL", f"Unknown action {name}.").to_dict()

    def _safely(self, name: str, fn: Callable[[], dict]) -> dict:
        try:
            return fn()
        except RuleError as err:
            return err.to_dict()
        except PolicyParameterError as exc:
            logger.error("Tool %s: %s", name, exc)
            return RuleError("POLICY_CONFIG_ERROR", "A store policy setting is missing, so I can't decide this right now.").to_dict()
        except sqlite3.Error as exc:
            logger.error("Tool %s database error: %s", name, type(exc).__name__)
            return RuleError("DATABASE_ERROR", "The order system is unavailable right now.").to_dict()

    # -- 1. get_orders ------------------------------------------------------
    def get_orders(self, order_id: str | None = None, status: str | None = None) -> dict:
        with connection() as conn:
            params = fetch_policy_parameters(conn)
            if order_id:
                order = fetch_customer_order(conn, self.customer_id, order_id)
                if order is None:
                    raise order_not_found()
                orders = [order] if not status or order["status"] == status else []
            else:
                orders = fetch_customer_orders(conn, self.customer_id, status)
            described = [
                self._describe_order(o, fetch_order_items(conn, o["order_id"]), fetch_shipment(conn, o["order_id"]), params)
                for o in orders
            ]
        return {"status": "success", "today": format_date(self.today), "count": len(described), "orders": described}

    def _describe_order(self, order: dict, items: list[dict], shipment: dict | None, params: dict) -> dict:
        delivered = delivered_on(order, shipment, self.today)
        window = int_parameter(params, "return_window_days")
        described_items = []
        for item in items:
            entry = {
                "item_id": item["item_id"],
                "name": item["name"],
                "category": item["category"],
                "qty": item["qty"],
                "unit_price": item["unit_price"],
                "opened": bool(item["opened"]),
                "returnable": item["returnable"],
                "warranty_class": item["warranty_class"],
                "warranty_ends": None,
                "under_warranty": False,
            }
            months = warranty_months(item["warranty_class"], params)
            if delivered and months is not None:
                end = warranty_end(delivered, months)
                entry["warranty_ends"] = format_date(end)
                entry["under_warranty"] = self.today <= end
            described_items.append(entry)

        days = (self.today - delivered).days if delivered else None
        out = {
            "order_id": order["order_id"],
            "order_date": format_date(parse_date(order["order_date"])),
            "status": order["status"],
            "total": order["total"],
            "items": described_items,
            "delivered_on": format_date(delivered),
            "days_since_delivery": days,
            "return_window_days": window,
            "return_window_open": days is not None and days <= window,
        }
        if shipment:
            out["shipment"] = {
                "carrier": shipment["carrier"],
                "tracking_no": shipment["tracking_no"],
                "status": shipment["status"],
                "shipped_on": format_date(parse_date(shipment["shipped_at"])),
                "eta": format_date(parse_date(shipment["eta"])),
                "original_eta": format_date(parse_date(shipment["original_eta"])),
            }
        return out

    # -- 2. search_policy ---------------------------------------------------
    def search_policy(self, query: str) -> dict:
        index = get_index(self.settings.policies_path, self.settings.manifest_path)
        result = {"status": "success", **index.search(query, self.today)}
        if not result["matches"] and not result["upcoming_changes"]:
            result["note"] = (
                "No current policy covers this. Say the policies do not cover it and suggest "
                "emailing the support team. Do not guess."
            )
        return result

    # -- 3. check_return_eligibility ------------------------------------------
    def check_return_eligibility(self, order_id: str, item_ids: list[str], reason: str) -> dict:
        with connection() as conn:
            return self._evaluate_return(conn, order_id, item_ids, reason)

    def _policy_basis(self, params: dict[str, dict], keys: list[str]) -> list[dict]:
        """The policy sections a decision rests on, taken from the source columns
        of the policy_parameters it used (only sections of current documents)."""
        index = get_index(self.settings.policies_path, self.settings.manifest_path)
        basis, seen = [], set()
        for key in keys:
            row = params.get(key)
            if not row:
                continue
            ref = (row["source_doc_id"], row["source_section"])
            if ref in seen or not index.is_current(ref[0], self.today):
                continue
            section = index.get_section(*ref)
            if section is not None:
                seen.add(ref)
                basis.append({"doc_id": ref[0], "section": ref[1], "text": section.text})
        return basis

    def _evaluate_return(self, conn: sqlite3.Connection, order_id: str, item_ids: list[str], reason: str) -> dict:
        order = fetch_customer_order(conn, self.customer_id, order_id)
        if order is None:
            raise order_not_found()
        params = fetch_policy_parameters(conn)
        basis = self._policy_basis(params, ["return_window_days"])
        try:
            decision = evaluate_return(
                order=order,
                items=fetch_order_items(conn, order_id),
                shipment=fetch_shipment(conn, order_id),
                item_ids=item_ids,
                reason=reason,
                params=params,
                today=self.today,
            )
        except RuleError as err:
            err.details.setdefault("policy_basis", basis)
            raise
        return {**decision, "policy_basis": basis}

    def _evaluate_ticket(self, conn: sqlite3.Connection, order_id: str, ticket_type: str, summary: str) -> dict:
        order = fetch_customer_order(conn, self.customer_id, order_id)
        if order is None:
            raise order_not_found()
        params = fetch_policy_parameters(conn)
        is_warranty = isinstance(ticket_type, str) and ticket_type.strip().lower() == "warranty"
        basis = self._policy_basis(params, ["standard_warranty_months", "extended_warranty_months"]) if is_warranty else []
        try:
            decision = evaluate_ticket(
                order=order,
                items=fetch_order_items(conn, order_id),
                shipment=fetch_shipment(conn, order_id),
                ticket_type=ticket_type,
                params=params,
                today=self.today,
            )
        except RuleError as err:
            if basis:
                err.details.setdefault("policy_basis", basis)
            raise
        if not summary.strip():
            raise RuleError("INVALID_SUMMARY", "Please describe the problem or request in a sentence.")
        return {**decision, "policy_basis": basis} if basis else decision

    # -- 4. create_return_request (write; only after "yes") -------------------
    def create_return_request(self, order_id: str, item_ids: list[str], reason: str) -> dict:
        with connection(write=True) as conn:
            # Re-check inside the write transaction so the decision cannot be stale.
            decision = self._evaluate_return(conn, order_id, item_ids, reason)
            request_id = str(uuid4())
            conn.execute(
                """
                INSERT INTO return_requests
                    (request_id, order_id, customer_id, item_ids, reason, status, created_at)
                VALUES (?, ?, ?, ?, ?, 'requested', ?)
                """,
                (request_id, order_id, self.customer_id, json.dumps(decision["item_ids"]),
                 decision["reason"], _now_iso()),
            )
            saved = dict(conn.execute(
                "SELECT * FROM return_requests WHERE request_id = ? AND customer_id = ?",
                (request_id, self.customer_id),
            ).fetchone())
        saved["item_ids"] = json.loads(saved["item_ids"])
        return {"status": "success", "return_request": saved, "items": decision["items"]}

    # -- 5. create_support_ticket (write; only after "yes") -------------------
    def create_support_ticket(self, order_id: str, ticket_type: str, summary: str) -> dict:
        with connection(write=True) as conn:
            decision = self._evaluate_ticket(conn, order_id, ticket_type, summary)
            ticket_id = str(uuid4())
            conn.execute(
                """
                INSERT INTO support_tickets
                    (ticket_id, order_id, customer_id, ticket_type, summary, status, created_at)
                VALUES (?, ?, ?, ?, ?, 'open', ?)
                """,
                (ticket_id, order_id, self.customer_id, decision["ticket_type"], summary.strip(), _now_iso()),
            )
            saved = dict(conn.execute(
                "SELECT * FROM support_tickets WHERE ticket_id = ? AND customer_id = ?",
                (ticket_id, self.customer_id),
            ).fetchone())
        return {"status": "success", "support_ticket": saved}
