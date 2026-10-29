"""Conversation orchestration (assessment §2.5).

Flow for one message:
1. If the session has a waiting proposal and the message is a clear yes, run the
   write tool (which re-checks the rules) and report. A plain "no" cancels it.
   Any other message drops the proposal and is handled normally.
2. Otherwise run the LLM tool loop. Read tools run immediately. When the LLM
   asks for a write tool, Python validates it and stores it as the session's
   single proposal; nothing is written until the next message is a yes.
3. Python post-checks the LLM's final JSON: route, citations (only sections
   actually retrieved this turn), the exact ORDER_NOT_FOUND reply, and the
   pending_confirmation flag.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Callable
from uuid import uuid4

from app.config import Settings
from app.llm import ChatModel, LLMError, build_model
from app.policy import get_index
from app.rules import ORDER_NOT_FOUND_MESSAGE, format_date
from app.sessions import Session
from app.tools import READ_TOOLS, TOOL_DEFINITIONS, TOOL_NAMES, ToolRunner, parse_args

logger = logging.getLogger("resolvebot.chat")

ROUTES = ("answer", "clarify", "no_answer", "out_of_scope")
HISTORY_FOR_LLM = 12
TURN_BUDGET_SECONDS = 27.0  # the API must reply within 30 seconds

_YES_RE = re.compile(
    r"^(?:"
    r"(?:yes|yeah|yep|yup)(?: please)?(?: (?:go ahead|proceed|confirm|confirmed|do it|please do|please go ahead|please proceed))?"
    r"|(?:please )?(?:go ahead|proceed)(?: please)?"
    r"|(?:i )?confirm(?:ed)?"
    r"|please do"
    r")$"
)
_NO_RE = re.compile(
    r"^(?:no|nope|nah|no thanks|no thank you|cancel|cancel it|don t|do not|never mind|nevermind|stop)(?: please| thanks)?$"
)


def _normalise(message: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", message.lower())).strip()


def is_confirmation(message: str) -> bool:
    """A clear yes ('yes', 'confirm', 'go ahead', 'please do'). 'just do it' is not."""
    return bool(_YES_RE.match(_normalise(message)))


def is_refusal(message: str) -> bool:
    return bool(_NO_RE.match(_normalise(message)))


def _response(answer: str, route: str, *, citations=None, tool_calls=None,
              pending: bool = False, error: str | None = None) -> dict:
    body = {
        "request_id": str(uuid4()),
        "answer": answer,
        "route": route,
        "citations": citations or [],
        "tool_calls": tool_calls or [],
        "pending_confirmation": pending,
    }
    if error:
        body["error"] = error
    return body


def _log_entry(name: str, args: dict, result: dict, status: str | None = None) -> dict:
    failed = result.get("status") == "error"
    entry = {"name": name, "args": args, "status": status or ("error" if failed else "success")}
    if failed:
        entry["error_code"] = result.get("error_code")
    return entry


def system_prompt(settings: Settings) -> str:
    today = settings.app_today
    return f"""You are ResolveBot, the customer support assistant for ShopSphere, an online electronics store.
Today is {format_date(today)} ({today.isoformat()}). Use this date, never any other "today".
A date without a year means {today.year}. Write dates like "3 July 2026".

The customer is already logged in; their identity is handled by the system. Never ask for or use a customer ID.

TOOLS
- get_orders: order status, ETA, items (with item_id), delivery date, days since delivery, warranty end dates.
- search_policy: policy sections. Use it for every policy question and to quote the rule behind a decision.
- check_return_eligibility: the store's return decision. Python decides; you only explain the result.
  Its policy_basis lists the policy sections the decision rests on: cite them.
- create_return_request / create_support_ticket: these only PROPOSE an action. The system saves it only if
  the customer's next message is a clear yes.

RULES
1. Never invent orders, items, dates, policy rules, numbers or citations. Use only tool results.
2. If a tool returns error_code ORDER_NOT_FOUND, reply exactly: "{ORDER_NOT_FOUND_MESSAGE}" with route no_answer.
3. Returns: find the item IDs with get_orders, then call check_return_eligibility. Reason must be changed_mind
   (no longer wanted) or defective (broken, damaged or wrong item). If the order, the item or the reason is
   missing or unclear, ask for everything that is missing in one message (route clarify). If several orders or
   items match ("my charger"), list them and ask which one. Never guess.
4. Propose (call create_return_request / create_support_ticket) only when the customer asks for the action
   ("please return it", "open a ticket", "cancel it"). If they only ask a question ("can I return it?"), answer
   and offer to do it, without calling the create tool. If the rules say no, explain why and propose nothing.
5. After a proposal, describe exactly what will be done (order, items, reason or ticket type) and end with
   "Shall I go ahead?". Never say it is done or saved.
6. Warranty: a warranty ticket needs a delivered order with an item still under warranty. Cancellation: only
   orders with status processing; otherwise explain it has already shipped.
7. Policy answers: answer only from search_policy matches and cite each section you used. If the FAQ and a
   policy disagree, follow the policy. Entries in upcoming_changes are not in force yet: never apply them as
   current rules, but you may mention them with their start date. If no section covers the question, say the
   policies do not cover it, suggest emailing the support team, and use route no_answer with no citations.
8. If the message asks several things, answer all of them.
9. Topics not about ShopSphere orders or policies (weather, poems, general knowledge): route out_of_scope,
   call no tools, and reply in one line saying what you can help with (orders, returns, warranty,
   cancellations and store policies).

OUTPUT
When you are done with tools, reply with a JSON object only:
{{"answer": "<reply to the customer>", "route": "answer|clarify|no_answer|out_of_scope",
  "citations": [{{"doc_id": "<doc_id>", "section": "<§x.y>"}}]}}
route: answer = normal reply, a "no, because...", a proposal, or a result; clarify = you need more details;
no_answer = order not found or policies do not cover it; out_of_scope = not about the store.
citations: only sections returned by search_policy in this turn that you actually used; [] if none."""


def _parse_final(content: str | None) -> dict | None:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", (content or "").strip())
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("answer"), str) or not data["answer"].strip():
        return None
    return data


class _Turn:
    """Bookkeeping for one assistant turn."""

    def __init__(self) -> None:
        self.tool_log: list[dict] = []
        self.retrieved: list[tuple[str, str]] = []  # (doc_id, section) in retrieval order
        self.searches = 0
        self.searches_with_results = 0
        self.other_tool_success = False
        self.order_not_found = False
        self.proposal: dict | None = None
        self.basis: list[tuple[str, str]] = []  # sections a rule decision was based on

    def collect_basis(self, result: dict) -> None:
        for ref in result.get("policy_basis") or []:
            key = (ref["doc_id"], ref["section"])
            if key not in self.basis:
                self.basis.append(key)
                self.retrieved.append(key)


def _clean_citations(raw: object, turn: _Turn, settings: Settings) -> list[dict]:
    """Keep only citations to sections retrieved this turn from non-superseded documents."""
    index = get_index(settings.policies_path, settings.manifest_path)
    allowed = set(turn.retrieved)
    out, seen = [], set()
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        doc_id = str(item.get("doc_id", "")).strip()
        section = str(item.get("section", "")).strip()
        if section and not section.startswith("§"):
            section = "§" + section
        key = (doc_id, section)
        if key in allowed and key not in seen and index.has_section(doc_id, section, settings.app_today):
            seen.add(key)
            out.append({"doc_id": doc_id, "section": section})
    return out


def _run_tool_call(call, runner: ToolRunner, turn: _Turn) -> dict:
    """Execute one LLM tool call; returns the result sent back to the LLM."""
    name = call.name
    if name not in TOOL_NAMES:
        result = {"status": "error", "error_code": "UNKNOWN_TOOL", "message": f"There is no tool called {name}."}
        turn.tool_log.append(_log_entry(name, {}, result))
        return result

    try:
        raw_args = json.loads(call.arguments) if call.arguments.strip() else {}
    except json.JSONDecodeError:
        raw_args = None
    args, shown_args, error = parse_args(name, raw_args)
    if error:
        result = error.to_dict()
    elif name in READ_TOOLS:
        result = runner.run_read_tool(name, args)
    elif turn.proposal is not None:
        result = {"status": "error", "error_code": "PROPOSAL_ALREADY_PENDING",
                  "message": "Only one action can wait for confirmation at a time. Ask about the first one."}
    else:
        result = runner.propose(name, args)
        turn.collect_basis(result)
        if result.get("status") == "success":
            stored = dict(shown_args)
            if name == "create_return_request":  # store the normalised, checked values
                stored.update(item_ids=result["item_ids"], reason=result["reason"])
            turn.proposal = {"name": name, "args": stored}
            turn.tool_log.append(_log_entry(name, stored, result, status="pending_confirmation"))
            return {
                "status": "pending_confirmation",
                "message": "NOT saved yet. Describe this action to the customer and ask 'Shall I go ahead?'.",
                "checked": result,
            }

    turn.collect_basis(result)
    if name == "search_policy":
        turn.searches += 1
    if result.get("status") == "error":
        if result.get("error_code") == "ORDER_NOT_FOUND":
            turn.order_not_found = True
    elif name == "search_policy":
        if result["matches"] or result["upcoming_changes"]:
            turn.searches_with_results += 1
        for match in result["matches"] + result["upcoming_changes"]:
            turn.retrieved.append((match["doc_id"], match["section"]))
    else:
        turn.other_tool_success = True
    turn.tool_log.append(_log_entry(name, shown_args, result))
    return result


def _confirmed_action(session: Session, proposal: dict, runner: ToolRunner) -> dict:
    name, args = proposal["name"], proposal["args"]
    result = runner.execute_write(name, args)
    log = [_log_entry(name, args, result)]

    if result.get("status") == "success":
        if name == "create_return_request":
            saved = result["return_request"]
            items = ", ".join(f"{i['name']} ({i['item_id']})" for i in result["items"])
            answer = (f"Your return request is saved: {items} from order {saved['order_id']}. "
                      f"Reference: {saved['request_id']}. Status: {saved['status']}.")
        else:
            saved = result["support_ticket"]
            answer = (f"Your {saved['ticket_type']} support ticket for order {saved['order_id']} is open. "
                      f"Reference: {saved['ticket_id']}. Status: {saved['status']}.")
        route = "answer"
    elif result.get("error_code") == "ORDER_NOT_FOUND":
        answer, route = ORDER_NOT_FOUND_MESSAGE, "no_answer"
    else:
        answer = f"I couldn't save that, so nothing has been changed. {result.get('message', '')}".strip()
        route = "answer"

    session.remember("assistant", answer)
    return _response(answer, route, tool_calls=log)


def handle_chat(
    session: Session,
    message: str,
    settings: Settings,
    model_factory: Callable[[Settings], ChatModel] | None = None,
) -> dict:
    """Handle one customer message. The caller holds ``session.lock``."""
    runner = ToolRunner(session.customer_id, settings)
    message = message.strip()

    proposal, session.proposal = session.proposal, None  # a proposal lives for one message only
    if proposal is not None:
        if is_confirmation(message):
            session.remember("user", message)
            return _confirmed_action(session, proposal, runner)
        if is_refusal(message):
            session.remember("user", message)
            answer = "Okay, I haven't done anything and nothing was saved. Is there anything else I can help with?"
            session.remember("assistant", answer)
            return _response(answer, "answer")
        # Anything else: the proposal is forgotten and the message handled normally.

    try:
        model = (model_factory or build_model)(settings)
    except LLMError as err:
        logger.error("LLM unavailable (%s): %s", err.code, err.operator_hint)
        return _response(err.public_message, "no_answer", error=err.code)

    turn = _Turn()
    messages = [
        {"role": "system", "content": system_prompt(settings)},
        *session.messages[-HISTORY_FOR_LLM:],
        {"role": "user", "content": message},
    ]
    deadline = time.monotonic() + TURN_BUDGET_SECONDS
    calls_used = 0
    raw_content: str | None = None
    finished = False

    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining < 2:
                raise LLMError("LLM_TIMEOUT", "Sorry, that took too long to answer. Please try again.",
                               "Turn time budget exhausted.")
            reply = model.complete(messages, TOOL_DEFINITIONS, timeout=remaining)
            if not reply.tool_calls:
                raw_content, finished = reply.content, True
                break
            if calls_used >= settings.max_tool_calls:
                break
            messages.append({
                "role": "assistant",
                "content": reply.content,
                "tool_calls": [
                    {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                    for c in reply.tool_calls
                ],
            })
            for call in reply.tool_calls:
                if calls_used >= settings.max_tool_calls:
                    result = {"status": "error", "error_code": "TOOL_LIMIT_REACHED",
                              "message": "Tool limit reached for this message. Answer with what you have."}
                else:
                    calls_used += 1
                    result = _run_tool_call(call, runner, turn)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result, default=str)})
    except LLMError as err:
        logger.error("LLM call failed (%s): %s", err.code, err.operator_hint)
        return _response(err.public_message, "no_answer", tool_calls=turn.tool_log, error=err.code)

    session.remember("user", message)

    if not finished:
        # The model kept asking for tools past the limit; any proposal is dropped.
        answer = ("Sorry, I couldn't finish that in one go. Could you ask about one thing at a time, "
                  "including the order ID if it's about an order?")
        session.remember("assistant", answer)
        return _response(answer, "answer", tool_calls=turn.tool_log)

    final = _parse_final(raw_content)
    if final is None:
        # Model ignored the JSON format: keep its text, cite only the top retrieved section.
        logger.warning("LLM final message was not valid JSON; using it as plain text")
        text = (raw_content or "").strip()
        final = {"answer": text or "Sorry, I couldn't put an answer together. Could you rephrase that?",
                 "route": "answer" if text else "clarify",
                 "citations": [{"doc_id": d, "section": s} for d, s in turn.retrieved[:1]]}

    answer = final["answer"].strip()
    route = final.get("route") if final.get("route") in ROUTES else "answer"
    citations = _clean_citations(final.get("citations"), turn, settings)
    if not citations and turn.basis:
        # The reply explains a rule decision; cite the sections that rule comes from.
        citations = [{"doc_id": d, "section": s} for d, s in turn.basis]
    pending = False

    if turn.order_not_found:
        if ORDER_NOT_FOUND_MESSAGE not in answer:
            answer = ORDER_NOT_FOUND_MESSAGE
        route = "no_answer"
    elif turn.proposal is not None:
        session.proposal = turn.proposal
        pending, route = True, "answer"
        if "?" not in answer:
            answer += " Shall I go ahead?"
    elif (route == "answer" and turn.searches and not turn.searches_with_results
          and not turn.other_tool_success):
        route = "no_answer"  # nothing in the current policies supports an answer
    elif route == "no_answer" and citations:
        # Seen live: the model sometimes labels a fully cited policy answer no_answer.
        # Verified citations to retrieved sections mean the policies do cover it.
        route = "answer"

    if route in ("no_answer", "out_of_scope"):
        citations = []

    session.remember("assistant", answer)
    return _response(answer, route, citations=citations, tool_calls=turn.tool_log, pending=pending)
