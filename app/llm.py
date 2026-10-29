"""Thin wrapper around the OpenAI chat-completions API (openai>=1.x).

Provider errors are turned into ``LLMError`` with a stable code, a short
customer-safe message and an operator hint for the server log. Raw provider
messages are never passed on, because some of them echo part of the API key.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

import openai

from app.config import Settings

logger = logging.getLogger("resolvebot.llm")


class LLMError(Exception):
    def __init__(self, code: str, public_message: str, operator_hint: str):
        super().__init__(code)
        self.code = code
        self.public_message = public_message
        self.operator_hint = operator_hint


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class AssistantTurn:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)


class ChatModel(Protocol):
    def complete(self, messages: list[dict], tools: list[dict], timeout: float) -> AssistantTurn: ...


UNAVAILABLE = "Sorry, I can't answer right now because the assistant service is unavailable. Please try again in a few minutes."


def _map_error(exc: Exception) -> LLMError:
    # APITimeoutError subclasses APIConnectionError, so it is checked first.
    if isinstance(exc, openai.APITimeoutError):
        return LLMError("LLM_TIMEOUT", "Sorry, that took too long to answer. Please try again.",
                        "OpenAI request timed out.")
    if isinstance(exc, openai.APIConnectionError):
        return LLMError("LLM_CONNECTION_FAILED", UNAVAILABLE,
                        "Could not connect to the OpenAI API (network/DNS/proxy).")
    if isinstance(exc, openai.AuthenticationError):
        return LLMError("LLM_AUTH_FAILED", UNAVAILABLE,
                        "OpenAI rejected the API key (401). Check OPENAI_API_KEY.")
    if isinstance(exc, openai.PermissionDeniedError):
        return LLMError("LLM_PERMISSION_DENIED", UNAVAILABLE,
                        "OpenAI denied access (403). Check the key's project/organisation permissions.")
    if isinstance(exc, openai.RateLimitError):
        if getattr(exc, "code", None) == "insufficient_quota":
            return LLMError("LLM_QUOTA_EXCEEDED", UNAVAILABLE,
                            "OpenAI quota exhausted (insufficient_quota). Check billing for this key.")
        return LLMError("LLM_RATE_LIMITED", "Sorry, I'm getting too many requests right now. Please try again shortly.",
                        "OpenAI rate limit hit (429).")
    if isinstance(exc, openai.NotFoundError):
        return LLMError("LLM_MODEL_NOT_FOUND", UNAVAILABLE,
                        "OpenAI returned 404. Check OPENAI_MODEL is a valid model for this key.")
    if isinstance(exc, openai.BadRequestError):
        return LLMError("LLM_BAD_REQUEST", UNAVAILABLE, "OpenAI rejected the request (400).")
    if isinstance(exc, openai.APIStatusError):
        return LLMError("LLM_UNAVAILABLE", UNAVAILABLE, f"OpenAI returned HTTP {exc.status_code}.")
    return LLMError("LLM_ERROR", UNAVAILABLE, f"Unexpected LLM error: {type(exc).__name__}.")


class OpenAIChatModel:
    def __init__(self, settings: Settings):
        problem = settings.llm_config_problem()
        if problem:
            raise LLMError("LLM_NOT_CONFIGURED",
                           "Sorry, the chat assistant is not configured on the server yet, so I can't answer that. "
                           "Please try again later.",
                           problem)
        self._model = settings.openai_model
        self._client = openai.OpenAI(api_key=settings.openai_api_key, max_retries=1)

    def complete(self, messages: list[dict], tools: list[dict], timeout: float) -> AssistantTurn:
        try:
            completion = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0,
                response_format={"type": "json_object"},
                timeout=timeout,
            )
        except openai.OpenAIError as exc:
            raise _map_error(exc) from None
        if not completion.choices:
            raise LLMError("LLM_EMPTY_RESPONSE", UNAVAILABLE, "OpenAI returned no choices.")
        message = completion.choices[0].message
        return AssistantTurn(
            content=message.content,
            tool_calls=[
                ToolCall(id=c.id, name=c.function.name, arguments=c.function.arguments or "")
                for c in (message.tool_calls or [])
                if getattr(c, "function", None) is not None
            ],
        )


def build_model(settings: Settings) -> ChatModel:
    return OpenAIChatModel(settings)
