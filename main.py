"""ResolveBot HTTP API entry point: ``uvicorn main:app --port 8000``."""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Annotated
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, StringConstraints, ValidationError

from app.chat import handle_chat
from app.config import get_settings
from app.db import customer_exists
from app.policy import get_index
from app.sessions import SessionForbidden, SessionStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("resolvebot")


class ChatRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    session_id: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)]
    message: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.validate_paths()
    get_index(settings.policies_path, settings.manifest_path)  # fail fast on unreadable policy files
    problem = settings.llm_config_problem()
    if problem:
        logger.warning("LLM is not configured: %s Chat replies will return route no_answer.", problem)
    else:
        logger.info("LLM provider=%s model=%s", settings.llm_provider, settings.openai_model)
    logger.info("APP_TODAY=%s", settings.app_today.isoformat())
    app.state.sessions = SessionStore()
    yield


app = FastAPI(title="ResolveBot", version="1.0.0", lifespan=lifespan)


def _error(status: int, code: str, message: str) -> JSONResponse:
    """HTTP errors keep the same JSON shape as normal replies."""
    return JSONResponse(status_code=status, content={
        "request_id": str(uuid4()),
        "answer": message,
        "route": "no_answer",
        "citations": [],
        "tool_calls": [],
        "pending_confirmation": False,
        "error": code,
        "detail": message,
    })


@app.post("/chat")
async def chat(request: Request):
    customer_id = request.headers.get("X-Customer-Id", "").strip()
    if not customer_id or not await run_in_threadpool(customer_exists, customer_id):
        return _error(401, "UNAUTHORIZED", "Missing or unknown X-Customer-Id header.")

    try:
        body = ChatRequest.model_validate(json.loads(await request.body()))
    except (json.JSONDecodeError, UnicodeDecodeError, ValidationError):
        return _error(400, "INVALID_REQUEST",
                      "Body must be JSON with non-empty string fields 'session_id' and 'message'.")

    try:
        session = request.app.state.sessions.claim(body.session_id, customer_id)
    except SessionForbidden:
        return _error(403, "FORBIDDEN", "This session belongs to another customer.")

    def run() -> dict:
        with session.lock:
            return handle_chat(session, body.message, get_settings())

    return await run_in_threadpool(run)


@app.exception_handler(Exception)
async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    # Log the type only; never send stack traces or internals to the client.
    logger.exception("Unhandled error: %s", type(exc).__name__)
    return _error(500, "INTERNAL_ERROR", "Something went wrong on our side. Please try again.")
