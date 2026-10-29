# ResolveBot

A customer-support chat API for ShopSphere, a made-up electronics store. A logged-in customer
can track orders, ask about store policies, check whether an item can be returned, and open a
return request or a warranty/cancellation ticket. Nothing is saved until the customer says yes.

## Install and run

Python 3.10+.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then set OPENAI_API_KEY in .env
uvicorn main:app --port 8000
```

`.env` is always read from the project root, and relative paths such as `./data/...` are resolved
from the project root, so the server can be started from any directory. Real environment variables
override `.env` (a variable exported as an empty string is treated as unset).

| Variable | Default |
|---|---|
| `POLICIES_PATH` | `./data/policies.md` |
| `MANIFEST_PATH` | `./data/manifest.json` |
| `DB_PATH` | `./data/shopsphere.db` |
| `APP_TODAY` | `2026-07-01` |
| `LLM_PROVIDER` | `openai` (the only provider implemented) |
| `OPENAI_API_KEY` | none, required for chat replies |
| `OPENAI_MODEL` | `gpt-4o-mini` |
| `MAX_TOOL_CALLS` | `6` |

Without a key the service still starts (with a warning in the log); chat replies then come back
as `route: no_answer` with `"error": "LLM_NOT_CONFIGURED"`. A pending "yes"/"no" confirmation still
works without the LLM, because it is handled in Python.

### Example

```bash
curl -s localhost:8000/chat -H 'X-Customer-Id: C1009' -H 'Content-Type: application/json' \
  -d '{"session_id":"s1","message":"Return the speaker from SS-10458, I changed my mind."}'
```

### Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests copy `data/` to a temp folder; the real database is never written. `tests/test_live_llm.py`
calls the real OpenAI API and is skipped automatically when no key is configured. All other tests use
a scripted fake LLM and need no key.

> If pytest crashes while loading an unrelated plugin (e.g. ROS `launch_testing` on `PYTHONPATH`),
> run `env -u PYTHONPATH pytest` or `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest`.

## How it works

```text
main.py           HTTP: 400/401/403 checks, session ownership, JSON contract
app/chat.py       conversation: confirm-or-cancel step, LLM tool loop, post-checks on the reply
app/tools.py      the five tools, their input schemas and OpenAI definitions
app/rules.py      business rules (return checks, warranty, cancellation), pure functions
app/policy.py     policy parsing and search (RAG)
app/db.py         parameterised SQLite queries, write transactions
app/sessions.py   per-session owner, history and the single waiting proposal
app/llm.py        OpenAI client wrapper and provider error mapping
app/config.py     settings from .env / environment
```

### Tools

| Tool | What it does |
|---|---|
| `get_orders` | The customer's orders (one by `order_id`, or a list, optionally by `status`) with items, shipment, ETA, delivery date, days since delivery, each item's warranty end date and whether it is still covered. |
| `search_policy` | Best-matching policy sections with `doc_id` and `section`, plus any scheduled future changes (labelled, not applied). |
| `check_return_eligibility` | Runs the §2.3 checks in order and returns `eligible` or the first failing error code. |
| `create_return_request` | When the LLM calls it, Python validates and stores a **proposal** only. It is executed after a yes. |
| `create_support_ticket` | Same, for `warranty` and `cancellation` tickets. |

**Safety.** Every tool is bound to the customer from the `X-Customer-Id` header. The LLM's
arguments never include a customer ID (one it invents is ignored). Every order query has
`WHERE order_id = ? AND customer_id = ?`, so a missing order and someone else's order give the same
`ORDER_NOT_FOUND`. If any tool returns `ORDER_NOT_FOUND`, Python forces the reply to exactly
"I couldn't find that order on your account." with route `no_answer`, whatever the LLM wrote.

**Business rules** are Python only. All numbers (`return_window_days`, `standard_warranty_months`,
`extended_warranty_months`) are read from `policy_parameters` on every call; all dates use
`APP_TODAY`. Return checks run in the order from the brief and stop at the first failure:

1. Order belongs to the customer (`ORDER_NOT_FOUND`), the reason is valid (`INVALID_REASON`),
   the items are in the order (`INVALID_ITEM`), and the order is delivered (`NOT_DELIVERED`).
2. The product's `returnable` is not `no` (`NON_RETURNABLE`).
3. Days since delivery ≤ the window (`RETURN_WINDOW_EXPIRED`). Day 0 is the delivery date.
4. For `changed_mind` only: an opened `unopened_only` product is refused (`OPENED_NOT_RETURNABLE`).

The warranty end is `delivery date + relativedelta(months=n)`, and the end day is covered. A warranty
ticket needs a delivered order with at least one covered item (`WARRANTY_EXPIRED` / `NO_WARRANTY`). A
cancellation ticket needs status `processing` (`ORDER_ALREADY_SHIPPED`).

### Policy search (RAG)

`policies.md` is split at each top-level `#` heading and matched to `manifest.json` by the
`Document ID` line. A document is used as a rule only if its status is `current` and its
`effective_date` ≤ `APP_TODAY`. A document marked superseded in either the manifest or its own
header is never used. Current documents that start later are returned separately under
`upcoming_changes` with their start date, so the assistant can mention them but not apply them.

Each `##`/`###` heading with a `§` number becomes one searchable section. Search is BM25 keyword
ranking with light stemming and a small synonym list (e.g. earbuds → earphones, cost → fee). It runs
on a normal CPU with no extra dependencies. Sections from documents the question names get a boost.
FAQ sections are down-weighted and listed after policy sections, because the policy wins on
conflicts. Only the matching sections are sent to the LLM, never the whole file.

The LLM returns its citations. Python keeps only those that were actually retrieved in this turn
and belong to a non-superseded document, so made-up citations are dropped. Routes `no_answer` and
`out_of_scope` always have empty citations. If the policy search found nothing and no other tool
succeeded, an `answer` route is changed to `no_answer`. The reverse also applies: a reply the model
labels `no_answer` while citing verified, retrieved sections is changed to `answer`. If the model
cites nothing for a return or warranty decision, the reply cites the sections the rule's numbers
come from (`policy_parameters.source_doc_id` / `source_section`).

### The yes-step

1. When the LLM calls a write tool, Python runs the same checks. If they fail, the error goes back
   to the LLM and nothing is stored. If they pass, the call is stored as the session's single
   proposal, logged with status `pending_confirmation`, and `pending_confirmation` is `true`.
2. The proposal lives for exactly one message. If that message is a clear yes (`yes`, `yes please`,
   `confirm`, `go ahead`, `please do`, `yes, go ahead`…, case and punctuation ignored), the write
   runs. It re-checks ownership and all rules inside one `BEGIN IMMEDIATE` transaction, inserts with
   a UUID4 ID, status `requested`/`open` and a UTC timestamp, then returns the saved row. Any error
   rolls back.
3. A plain `no`/`cancel` drops it with a fixed reply. Any other message ("just do it", "ok", a new
   request, changed details) drops it and is handled normally, which may create a new proposal.
   Only one proposal can wait at a time; a second write call in the same turn is refused.

### Conversation and API

* Sessions are keyed by `session_id` and owned by the first customer who used it (another customer
  gets 403). History (last 20 messages) and the proposal are kept per session; requests in the same
  session are processed one at a time.
* The tool loop stops after `MAX_TOOL_CALLS` tool calls and replies politely. The whole turn has a
  27-second budget (the brief asks for replies within 30 s). The LLM runs at temperature 0.
* The LLM must reply with JSON `{answer, route, citations}`. If it does not, its text is used as
  the answer.
* OpenAI errors (bad key, no quota, rate limit, connection, timeout, unknown model) become a polite
  `no_answer` reply with an `error` code. The detail goes to the server log only; provider messages
  are never shown to the client, since they can echo part of the key.
* HTTP errors use the same JSON shape plus `error` and `detail`: 400 for invalid JSON or missing or
  non-string `session_id`/`message`, 401 for a missing or unknown `X-Customer-Id`, 403 for another
  customer's session. Everything else is 200.

## Assumptions

* The customer header is checked before the body, so an invalid body without a header gives 401.
* "Clear yes" follows the brief's examples (yes / confirm / go ahead / please do). "ok", "sure" and
  "just do it" are not confirmation.
* Return reasons are only `changed_mind` and `defective`. The LLM maps "broken", "damaged" or
  "wrong item" to `defective`, as the brief describes.
* The delivery date is the date part of `shipments.delivered_at`. An order counts as delivered only
  if `orders.status` is `delivered` and that date is on or before `APP_TODAY`.
* Ticket types are only `warranty` and `cancellation` (the brief's tool inputs). Cancelling an
  already-cancelled order returns an extra code, `ORDER_ALREADY_CANCELLED`.
* `created_at` uses the real clock (UTC, ISO 8601), because the brief asks for "the current time";
  every business date uses `APP_TODAY`.
* Extra error codes beyond the brief's list exist only for cases it doesn't cover:
  `INVALID_ARGUMENTS`, `INVALID_SUMMARY`, `PROPOSAL_ALREADY_PENDING`, `UNKNOWN_TOOL`,
  `TOOL_LIMIT_REACHED`, `DATABASE_ERROR`, `POLICY_CONFIG_ERROR`.

## What does not work yet / limitations

* Sessions are in memory. A restart forgets chat history and any waiting proposal, and sessions are
  not shared between several server processes. Run a single worker.
* Only OpenAI is implemented as the LLM provider.
* Whether the assistant *proposes* (an action request) or only *answers* (a question such as "can I
  return it?") is decided by the LLM, guided by the prompt. Python guarantees nothing is saved
  without a yes, but cannot stop an over-eager proposal.
* The same is true for clarifying questions ("return my charger" with several chargers): the LLM is
  told to list the options and ask, and Python cannot enforce that.
* Keyword search can miss questions phrased with words that are not in the policies or the synonym
  list. In that case the assistant says the policies do not cover it.
* Duplicate return requests for the same item are not blocked (the brief does not require it).

## Example chats

The flows below were run against the running server with `gpt-4o-mini`, and are also covered by
`tests/test_live_llm.py` (live, needs a key) and `tests/test_api.py` (scripted LLM, no key).

* Return the speaker from SS-10458 (customer C1009) → `get_orders`, `search_policy`,
  `check_return_eligibility`, then `create_return_request` with status `pending_confirmation`.
  Citation `returns_policy §2.1`, and nothing is saved. Then "Yes, go ahead." → one
  `return_requests` row (`["SS-10458-1"]`, `changed_mind`, `requested`).
* Same proposal, then "just do it" → nothing is saved, and a later "yes" does not save it either.
* Customer C1001 asks "Where is order SS-10458?" → "I couldn't find that order on your account."
  (`no_answer`), the same reply as for a non-existent order.
* Cancel a `shipped` order → `ORDER_ALREADY_SHIPPED`, with no proposal.
* Opened earbuds, `changed_mind` → `OPENED_NOT_RETURNABLE`; with `defective` → eligible.

## AI assistance

This project was built with the help of Claude Code (Anthropic), working from the assessment brief
and the original scaffold: implementation, the policy search, and the test suite.
