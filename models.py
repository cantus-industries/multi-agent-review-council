"""Typed domain objects and the error hierarchy for the critic council."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Mapping, Sequence

__all__ = [
    "CouncilError",
    "ConfigError",
    "ReviewError",
    "ReviewTimeout",
    "LedgerError",
    "Severity",
    "Verdict",
    "Finding",
    "Critique",
    "ReviewRequest",
]

MAX_DOCUMENT_LENGTH = 200_000
MAX_FINDINGS_PER_CRITIQUE = 100


class CouncilError(Exception):
    """Base class for every recoverable council failure."""


class ConfigError(CouncilError):
    """Raised when the persona configuration is missing or malformed."""


class ReviewError(CouncilError):
    """Raised when a reviewer cannot produce a critique."""


class ReviewTimeout(ReviewError):
    """Raised when a reviewer exceeds its wall-clock budget."""


class LedgerError(CouncilError):
    """Raised when the ledger is used out of order or given bad input."""


class Severity(IntEnum):
    """Ordered severity scale. Higher is worse."""

    INFO = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

    @property
    def label(self) -> str:
        return self.name.lower()

    @classmethod
    def parse(cls, value: object) -> "Severity":
        if isinstance(value, Severity):
            return value
        if isinstance(value, str):
            try:
                return cls[value.strip().upper()]
            except KeyError as exc:
                raise ConfigError(
                    f"unknown severity {value!r}; expected one of "
                    f"{[s.label for s in cls]}"
                ) from exc
        raise ConfigError(
            f"severity must be a string, got {type(value).__name__}"
        )


class Verdict(IntEnum):
    """Council outcome. Higher is more restrictive."""

    APPROVED = 0
    APPROVED_WITH_CONDITIONS = 1
    BLOCKED = 2

    @property
    def label(self) -> str:
        return self.name


@dataclass(frozen=True, slots=True)
class Finding:
    """A single objection raised by one council member."""

    code: str
    severity: Severity
    message: str
    evidence: str = ""
    financial_cost: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code.strip():
            raise ReviewError("Finding.code must be a non-empty string")
        if not isinstance(self.message, str) or not self.message.strip():
            raise ReviewError("Finding.message must be a non-empty string")
        if not isinstance(self.severity, Severity):
            raise ReviewError(
                f"Finding.severity must be a Severity, got "
                f"{type(self.severity).__name__}"
            )
        cost = self.financial_cost
        if isinstance(cost, bool) or not isinstance(cost, (int, float)):
            raise ReviewError(
                f"Finding.financial_cost must be a real number, got "
                f"{type(cost).__name__}"
            )
        if not math.isfinite(cost):
            raise ReviewError(
                f"Finding.financial_cost must be finite, got {cost!r}"
            )
        if cost < 0:
            raise ReviewError(
                f"Finding.financial_cost must not be negative, got {cost}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity.label,
            "message": self.message,
            "evidence": self.evidence,
            "financial_cost": float(self.financial_cost),
        }


@dataclass(frozen=True, slots=True)
class Critique:
    """One council member's complete response to a review request."""

    persona: str
    findings: tuple[Finding, ...] = ()
    backend: str = "unknown"
    duration_ms: float = 0.0
    error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.persona, str) or not self.persona.strip():
            raise ReviewError("Critique.persona must be a non-empty string")
        if len(self.findings) > MAX_FINDINGS_PER_CRITIQUE:
            raise ReviewError(
                f"critique from {self.persona} returned {len(self.findings)} "
                f"findings, exceeding the {MAX_FINDINGS_PER_CRITIQUE} cap"
            )
        if self.error is not None and self.findings:
            raise ReviewError(
                "a failed critique must not also carry findings; "
                "the reviewer's partial output cannot be trusted"
            )

    @property
    def failed(self) -> bool:
        return self.error is not None

    @property
    def max_severity(self) -> Severity | None:
        if not self.findings:
            return None
        return max(f.severity for f in self.findings)

    @property
    def financial_cost(self) -> float:
        return sum(f.financial_cost for f in self.findings)

    @classmethod
    def failure(cls, persona: str, reason: str, backend: str = "unknown",
                duration_ms: float = 0.0) -> "Critique":
        return cls(
            persona=persona,
            findings=(),
            backend=backend,
            duration_ms=duration_ms,
            error=reason,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "persona": self.persona,
            "findings": [f.to_dict() for f in self.findings],
            "backend": self.backend,
            "duration_ms": round(self.duration_ms, 3),
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class ReviewRequest:
    """The artifact under review, plus the governance rules in force."""

    document: str
    governance_rules: str = ""
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.document, str):
            raise ReviewError(
                f"document must be a string, got {type(self.document).__name__}"
            )
        if not self.document.strip():
            raise ReviewError("document must not be blank")
        if len(self.document) > MAX_DOCUMENT_LENGTH:
            raise ReviewError(
                f"document exceeds {MAX_DOCUMENT_LENGTH} characters "
                f"(got {len(self.document)})"
            )
        if not isinstance(self.governance_rules, str):
            raise ReviewError("governance_rules must be a string")


def sort_findings(findings: Sequence[Finding]) -> tuple[Finding, ...]:
    """Order findings most severe first, then by code for stable output."""
    return tuple(sorted(findings, key=lambda f: (-int(f.severity), f.code)))
