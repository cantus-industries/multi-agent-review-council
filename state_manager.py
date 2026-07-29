"""Aggregation of parallel critiques into a single, auditable verdict.

``ConsensusLedger`` is the one object touched from multiple threads, so every
mutation is guarded. Its central rule: **a run that did not hear from every
council member cannot be approved.** A reviewer that timed out, crashed, or was
never recorded degrades the run, and a degraded run is BLOCKED. There is no
code path that returns APPROVED on incomplete evidence.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

from erp_state_ledger import ProjectState, RiskEntry
from models import Critique, LedgerError, Severity, Verdict

__all__ = ["GovernanceRules", "LedgerSnapshot", "ConsensusLedger"]

BLOCKING_SEVERITY = Severity.HIGH


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class GovernanceRules:
    """The standards text in force, with an append-only revision history."""

    text: str
    version: int = 1
    history: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise LedgerError("governance rules text must be a string")
        if isinstance(self.version, bool) or not isinstance(self.version, int):
            raise LedgerError("governance rules version must be an integer")
        if self.version < 1:
            raise LedgerError("governance rules version must be >= 1")

    def sync(self, updated_text: str) -> "GovernanceRules":
        """Return the next revision. Re-syncing identical text is a no-op."""
        if not isinstance(updated_text, str) or not updated_text.strip():
            raise LedgerError("updated governance rules must be a non-empty string")
        if updated_text == self.text:
            return self
        return GovernanceRules(
            text=updated_text,
            version=self.version + 1,
            history=self.history + (self.text,),
        )


@dataclass(frozen=True, slots=True)
class LedgerSnapshot:
    """An immutable read of the ledger at one point in time."""

    verdict: Verdict
    critiques: tuple[Critique, ...]
    degraded: bool
    missing_personas: tuple[str, ...]
    failed_personas: tuple[str, ...]
    financial_impact: float
    max_severity: Severity | None
    compliance_adherence: bool
    governance_version: int
    generated_at: str

    @property
    def findings(self) -> tuple:
        return tuple(f for c in self.critiques for f in c.findings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.label,
            "degraded": self.degraded,
            "missing_personas": list(self.missing_personas),
            "failed_personas": list(self.failed_personas),
            "financial_impact": round(self.financial_impact, 2),
            "max_severity": (
                self.max_severity.label if self.max_severity is not None else None
            ),
            "compliance_adherence": self.compliance_adherence,
            "governance_version": self.governance_version,
            "generated_at": self.generated_at,
            "critiques": [c.to_dict() for c in self.critiques],
        }


class ConsensusLedger:
    """Collects critiques from parallel reviewers and computes the verdict."""

    def __init__(
        self,
        expected_personas: Iterable[str],
        blocking_personas: Iterable[str] = (),
        governance: GovernanceRules | None = None,
    ) -> None:
        expected = tuple(expected_personas)
        if not expected:
            raise LedgerError("a ledger needs at least one expected persona")
        duplicates = {p for p in expected if expected.count(p) > 1}
        if duplicates:
            raise LedgerError(f"duplicate expected personas {sorted(duplicates)}")

        blocking = frozenset(blocking_personas)
        unknown = blocking - set(expected)
        if unknown:
            raise LedgerError(
                f"blocking personas not in the roster: {sorted(unknown)}"
            )

        self._expected: tuple[str, ...] = expected
        self._blocking: frozenset[str] = blocking
        self._governance = governance or GovernanceRules(text="")
        self._critiques: dict[str, Critique] = {}
        self._lock = threading.Lock()

    @property
    def governance(self) -> GovernanceRules:
        with self._lock:
            return self._governance

    def sync_governance_rules(self, updated_text: str) -> GovernanceRules:
        """Replace the standards text, bumping the version and keeping history."""
        with self._lock:
            self._governance = self._governance.sync(updated_text)
            return self._governance

    def record(self, critique: Critique) -> None:
        """Record one member's critique. Safe to call from worker threads."""
        if not isinstance(critique, Critique):
            raise LedgerError(
                f"expected a Critique, got {type(critique).__name__}"
            )
        with self._lock:
            if critique.persona not in self._expected:
                raise LedgerError(
                    f"critique from unexpected persona {critique.persona!r}; "
                    f"roster is {list(self._expected)}"
                )
            if critique.persona in self._critiques:
                raise LedgerError(
                    f"duplicate critique from {critique.persona!r}"
                )
            self._critiques[critique.persona] = critique

    def snapshot(self) -> LedgerSnapshot:
        """Compute the verdict from everything recorded so far."""
        with self._lock:
            critiques = tuple(
                self._critiques[name]
                for name in self._expected
                if name in self._critiques
            )
            missing = tuple(
                name for name in self._expected if name not in self._critiques
            )
            governance_version = self._governance.version
            blocking = self._blocking

        failed = tuple(c.persona for c in critiques if c.failed)
        degraded = bool(missing or failed)

        severities = [
            f.severity for c in critiques for f in c.findings
        ]
        max_severity = max(severities) if severities else None
        financial_impact = sum(c.financial_cost for c in critiques)

        blocking_hit = any(
            c.persona in blocking and f.severity >= BLOCKING_SEVERITY
            for c in critiques
            for f in c.findings
        )

        if degraded or blocking_hit:
            verdict = Verdict.BLOCKED
        elif severities:
            verdict = Verdict.APPROVED_WITH_CONDITIONS
        else:
            verdict = Verdict.APPROVED

        return LedgerSnapshot(
            verdict=verdict,
            critiques=critiques,
            degraded=degraded,
            missing_personas=missing,
            failed_personas=failed,
            financial_impact=financial_impact,
            max_severity=max_severity,
            compliance_adherence=not (degraded or blocking_hit),
            governance_version=governance_version,
            generated_at=_utc_now().isoformat(),
        )

    def apply_to(
        self, state: ProjectState, snapshot: LedgerSnapshot | None = None
    ) -> LedgerSnapshot:
        """Write the run's outcome into the ERP project state."""
        if not isinstance(state, ProjectState):
            raise LedgerError(
                f"expected a ProjectState, got {type(state).__name__}"
            )
        result = snapshot if snapshot is not None else self.snapshot()

        for critique in result.critiques:
            for finding in critique.findings:
                state.risks.add(
                    RiskEntry(
                        code=finding.code,
                        severity=finding.severity,
                        description=finding.message,
                        source_persona=critique.persona,
                        financial_exposure=finding.financial_cost,
                    )
                )

        state.knowledge.set("last_verdict", result.verdict.label)
        state.knowledge.set("last_review_at", result.generated_at)
        state.knowledge.set("run_degraded", "true" if result.degraded else "false")
        state.knowledge.set(
            "governance_version", str(result.governance_version)
        )
        return result
