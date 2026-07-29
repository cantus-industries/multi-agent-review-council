"""Reviewer backends.

Two implementations satisfy the same protocol:

``RuleBasedReviewer``
    Deterministic regex checks from ``agents_config.json``. Runs offline, needs
    no credentials, and is what the test suite exercises.

``ClaudeReviewer``
    Sends the document to Claude with the persona's system prompt and a strict
    output schema. Requires the ``anthropic`` package and API credentials.

Both raise ``ReviewError`` on failure. Neither returns a default-approve
critique when it cannot do its job.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from config import Persona
from models import (
    Critique,
    Finding,
    ReviewError,
    ReviewRequest,
    Severity,
    sort_findings,
)

__all__ = [
    "Reviewer",
    "RuleBasedReviewer",
    "ClaudeReviewer",
    "GeminiReviewer",
    "CRITIQUE_SCHEMA",
    "build_reviewers",
]

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5"
DEFAULT_EFFORT = "high"
DEFAULT_MAX_TOKENS = 16_000

DEFAULT_GEMINI_MODEL = "gemini-2.5-pro"

CRITIQUE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Short SCREAMING_SNAKE identifier.",
                    },
                    "severity": {
                        "type": "string",
                        "enum": [s.label for s in Severity],
                    },
                    "message": {
                        "type": "string",
                        "description": "One sentence stating the objection.",
                    },
                    "evidence": {
                        "type": "string",
                        "description": "Verbatim excerpt the finding rests on.",
                    },
                    "financial_cost": {
                        "type": "number",
                        "description": "Estimated cost exposure in USD, or 0.",
                    },
                },
                "required": [
                    "code",
                    "severity",
                    "message",
                    "evidence",
                    "financial_cost",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["findings"],
    "additionalProperties": False,
}


@runtime_checkable
class Reviewer(Protocol):
    """What the council requires of a council member."""

    name: str
    backend: str

    def review(self, request: ReviewRequest) -> Critique:
        """Return a critique, or raise ReviewError. Never return a default."""
        ...


class RuleBasedReviewer:
    """Applies a persona's configured regex rules to the document."""

    backend = "rules"

    def __init__(self, persona: Persona) -> None:
        self.persona = persona
        self.name = persona.name

    def review(self, request: ReviewRequest) -> Critique:
        started = time.perf_counter()
        findings: list[Finding] = []

        for rule in self.persona.rules:
            match = rule.pattern.search(request.document)
            if match is None:
                continue
            findings.append(
                Finding(
                    code=rule.code,
                    severity=rule.severity,
                    message=rule.message,
                    evidence=_excerpt(request.document, match.start(), match.end()),
                    financial_cost=rule.financial_cost,
                )
            )

        elapsed = (time.perf_counter() - started) * 1000
        return Critique(
            persona=self.name,
            findings=sort_findings(findings),
            backend=self.backend,
            duration_ms=elapsed,
        )


class ClaudeReviewer:
    """Sends the document to Claude under the persona's system prompt."""

    backend = "claude"

    def __init__(
        self,
        persona: Persona,
        *,
        client: Any | None = None,
        model: str = DEFAULT_MODEL,
        effort: str = DEFAULT_EFFORT,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self.persona = persona
        self.name = persona.name
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self._client = client if client is not None else _build_client()

    def review(self, request: ReviewRequest) -> Critique:
        started = time.perf_counter()
        system = self.persona.system_prompt
        if request.governance_rules:
            system = (
                f"{system}\n\n"
                "Governance standards currently in force:\n"
                f"{request.governance_rules}"
            )

        try:
            response = self._client.beta.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                system=system,
                output_config={
                    "effort": self.effort,
                    "format": {"type": "json_schema", "schema": CRITIQUE_SCHEMA},
                },
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Review the following document from your assigned "
                            "viewpoint. Report only objections you can ground "
                            "in a verbatim excerpt. Return an empty findings "
                            "list if you have none.\n\n"
                            f"<document>\n{request.document}\n</document>"
                        ),
                    }
                ],
            )
        except Exception as exc:  # noqa: BLE001 - re-raised as ReviewError below
            raise _translate_api_error(self.name, exc) from exc

        if getattr(response, "stop_reason", None) == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None)
            raise ReviewError(
                f"{self.name}: the model declined this review"
                + (f" (category {category})" if category else "")
            )

        payload = _extract_json(self.name, response)
        findings = _parse_findings(self.name, payload)
        elapsed = (time.perf_counter() - started) * 1000
        return Critique(
            persona=self.name,
            findings=sort_findings(findings),
            backend=self.backend,
            duration_ms=elapsed,
        )


class GeminiReviewer:
    """Sends the document to Gemini under the persona's system prompt."""

    backend = "gemini"

    def __init__(
        self,
        persona: Persona,
        *,
        client: Any | None = None,
        model: str = DEFAULT_GEMINI_MODEL,
    ) -> None:
        self.persona = persona
        self.name = persona.name
        self.model = model
        self._client = client if client is not None else _build_gemini_client()

    def review(self, request: ReviewRequest) -> Critique:
        started = time.perf_counter()
        system = self.persona.system_prompt
        if request.governance_rules:
            system = (
                f"{system}\n\n"
                "Governance standards currently in force:\n"
                f"{request.governance_rules}"
            )

        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=(
                    "Review the following document from your assigned "
                    "viewpoint. Report only objections you can ground in a "
                    "verbatim excerpt. Return an empty findings list if you "
                    f"have none.\n\n<document>\n{request.document}\n</document>"
                ),
                config={
                    "system_instruction": system,
                    "response_mime_type": "application/json",
                    "response_schema": _gemini_schema(CRITIQUE_SCHEMA),
                },
            )
        except Exception as exc:  # noqa: BLE001 - re-raised as ReviewError below
            raise _translate_api_error(self.name, exc) from exc

        text = getattr(response, "text", None)
        if not text:
            raise ReviewError(
                f"{self.name}: the model returned no text (it may have been "
                "blocked by a safety filter)"
            )
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ReviewError(
                f"{self.name}: response was not valid JSON despite the "
                f"response schema: {exc}"
            ) from exc

        findings = _parse_findings(self.name, payload)
        elapsed = (time.perf_counter() - started) * 1000
        return Critique(
            persona=self.name,
            findings=sort_findings(findings),
            backend=self.backend,
            duration_ms=elapsed,
        )


def build_reviewers(
    personas: Sequence[Persona], backend: str, **kwargs: Any
) -> tuple[Reviewer, ...]:
    """Instantiate one reviewer per persona for the named backend."""
    factories = {
        "rules": RuleBasedReviewer,
        "claude": ClaudeReviewer,
        "gemini": GeminiReviewer,
    }
    factory = factories.get(backend)
    if factory is None:
        raise ReviewError(
            f"unknown backend {backend!r}; expected one of {sorted(factories)}"
        )
    if backend == "rules":
        return tuple(factory(p) for p in personas)
    return tuple(factory(p, **kwargs) for p in personas)


def _gemini_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Strip JSON Schema keywords the Gemini response schema rejects."""
    unsupported = {"additionalProperties", "$schema", "const"}
    result: dict[str, Any] = {}
    for key, value in schema.items():
        if key in unsupported:
            continue
        if isinstance(value, Mapping):
            result[key] = _gemini_schema(value)
        elif isinstance(value, list) and key != "required" and key != "enum":
            result[key] = [
                _gemini_schema(v) if isinstance(v, Mapping) else v for v in value
            ]
        else:
            result[key] = value
    return result


def _build_client() -> Any:
    try:
        import anthropic
    except ImportError as exc:
        raise ReviewError(
            "the Claude backend requires the 'anthropic' package; "
            "install it with `pip install anthropic` or use --backend rules"
        ) from exc
    return anthropic.Anthropic()


def _build_gemini_client() -> Any:
    try:
        from google import genai
    except ImportError as exc:
        raise ReviewError(
            "the Gemini backend requires the 'google-genai' package; "
            "install it with `pip install google-genai` or use --backend rules"
        ) from exc
    return genai.Client()


def _translate_api_error(persona: str, exc: Exception) -> ReviewError:
    """Map SDK exceptions onto ReviewError without importing anthropic eagerly."""
    name = type(exc).__name__
    if name == "RateLimitError":
        return ReviewError(f"{persona}: rate limited by the API")
    if name in {"AuthenticationError", "PermissionDeniedError"}:
        return ReviewError(f"{persona}: credentials rejected ({name})")
    if name == "NotFoundError":
        return ReviewError(f"{persona}: unknown model or endpoint")
    if name == "APIConnectionError":
        return ReviewError(f"{persona}: could not reach the API")
    return ReviewError(f"{persona}: review call failed ({name}: {exc})")


def _extract_json(persona: str, response: Any) -> Any:
    content = getattr(response, "content", None)
    if not content:
        raise ReviewError(f"{persona}: the model returned no content")

    for block in content:
        if getattr(block, "type", None) != "text":
            continue
        try:
            return json.loads(block.text)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ReviewError(
                f"{persona}: response was not valid JSON despite the output "
                f"schema: {exc}"
            ) from exc

    raise ReviewError(f"{persona}: response contained no text block")


def _parse_findings(persona: str, payload: Any) -> list[Finding]:
    if not isinstance(payload, dict):
        raise ReviewError(
            f"{persona}: expected a JSON object, got {type(payload).__name__}"
        )
    raw_findings = payload.get("findings")
    if not isinstance(raw_findings, list):
        raise ReviewError(f"{persona}: 'findings' must be a list")

    findings: list[Finding] = []
    for item in raw_findings:
        if not isinstance(item, dict):
            raise ReviewError(
                f"{persona}: each finding must be an object, got "
                f"{type(item).__name__}"
            )
        try:
            findings.append(
                Finding(
                    code=str(item.get("code", "")),
                    severity=Severity.parse(item.get("severity")),
                    message=str(item.get("message", "")),
                    evidence=str(item.get("evidence", "")),
                    financial_cost=item.get("financial_cost", 0.0),
                )
            )
        except Exception as exc:
            raise ReviewError(f"{persona}: malformed finding: {exc}") from exc
    return findings


def _excerpt(document: str, start: int, end: int, window: int = 40) -> str:
    left = max(0, start - window)
    right = min(len(document), end + window)
    snippet = document[left:right].replace("\n", " ").strip()
    prefix = "..." if left > 0 else ""
    suffix = "..." if right < len(document) else ""
    return f"{prefix}{snippet}{suffix}"
