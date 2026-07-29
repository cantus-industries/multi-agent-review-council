"""Structure of the project state the council audits against and updates.

Three things live here:

``Milestone``      one unit of tracked delivery work
``RiskRegister``   the accumulated, deduplicated set of open risks
``KnowledgeState`` versioned key/value context the council reads and writes

``ProjectState`` bundles all three. It is mutated from the orchestrator thread
only, after every reviewer has finished, so it is deliberately not locked --
``state_manager.ConsensusLedger`` owns the concurrent half of the problem.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable, Iterator, Mapping

from models import LedgerError, Severity

__all__ = [
    "MilestoneStatus",
    "Milestone",
    "RiskEntry",
    "RiskRegister",
    "KnowledgeState",
    "ProjectState",
]

MAX_MILESTONES = 500
MAX_RISKS = 2_000
MAX_KNOWLEDGE_KEYS = 1_000


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class MilestoneStatus(Enum):
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETE = "complete"

    @classmethod
    def parse(cls, value: object) -> "MilestoneStatus":
        if isinstance(value, MilestoneStatus):
            return value
        if isinstance(value, str):
            try:
                return cls(value.strip().lower())
            except ValueError as exc:
                raise LedgerError(
                    f"unknown milestone status {value!r}; expected one of "
                    f"{[s.value for s in cls]}"
                ) from exc
        raise LedgerError(
            f"milestone status must be a string, got {type(value).__name__}"
        )


@dataclass(frozen=True, slots=True)
class Milestone:
    """One tracked unit of delivery."""

    id: str
    name: str
    status: MilestoneStatus = MilestoneStatus.PLANNED
    owner: str = ""

    def __post_init__(self) -> None:
        for attr in ("id", "name"):
            value = getattr(self, attr)
            if not isinstance(value, str) or not value.strip():
                raise LedgerError(f"Milestone.{attr} must be a non-empty string")
        if not isinstance(self.status, MilestoneStatus):
            raise LedgerError(
                f"Milestone.status must be a MilestoneStatus, got "
                f"{type(self.status).__name__}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status.value,
            "owner": self.owner,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Milestone":
        if not isinstance(payload, Mapping):
            raise LedgerError(
                f"milestone must be an object, got {type(payload).__name__}"
            )
        missing = {"id", "name"} - set(payload)
        if missing:
            raise LedgerError(f"milestone missing keys {sorted(missing)}")
        return cls(
            id=str(payload["id"]),
            name=str(payload["name"]),
            status=MilestoneStatus.parse(payload.get("status", "planned")),
            owner=str(payload.get("owner", "")),
        )


@dataclass(frozen=True, slots=True)
class RiskEntry:
    """One open risk, traceable to the council member that raised it."""

    code: str
    severity: Severity
    description: str
    source_persona: str
    financial_exposure: float = 0.0
    opened_at: str = field(default_factory=lambda: _utc_now().isoformat())

    def __post_init__(self) -> None:
        for attr in ("code", "description", "source_persona"):
            value = getattr(self, attr)
            if not isinstance(value, str) or not value.strip():
                raise LedgerError(f"RiskEntry.{attr} must be a non-empty string")
        if not isinstance(self.severity, Severity):
            raise LedgerError(
                f"RiskEntry.severity must be a Severity, got "
                f"{type(self.severity).__name__}"
            )
        exposure = self.financial_exposure
        if isinstance(exposure, bool) or not isinstance(exposure, (int, float)):
            raise LedgerError("RiskEntry.financial_exposure must be a number")
        if not math.isfinite(exposure) or exposure < 0:
            raise LedgerError(
                "RiskEntry.financial_exposure must be finite and non-negative"
            )

    @property
    def key(self) -> tuple[str, str]:
        """Identity for deduplication: one risk per persona per code."""
        return (self.source_persona, self.code)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity.label,
            "description": self.description,
            "source_persona": self.source_persona,
            "financial_exposure": float(self.financial_exposure),
            "opened_at": self.opened_at,
        }


class RiskRegister:
    """The set of open risks, deduplicated by (persona, code)."""

    def __init__(self, entries: Iterable[RiskEntry] = ()) -> None:
        self._entries: dict[tuple[str, str], RiskEntry] = {}
        for entry in entries:
            self.add(entry)

    def add(self, entry: RiskEntry) -> bool:
        """Record a risk. Returns False if an identical one is already open."""
        if not isinstance(entry, RiskEntry):
            raise LedgerError(
                f"expected a RiskEntry, got {type(entry).__name__}"
            )
        if entry.key in self._entries:
            return False
        if len(self._entries) >= MAX_RISKS:
            raise LedgerError(
                f"risk register is full at {MAX_RISKS} entries; "
                "close resolved risks before adding more"
            )
        self._entries[entry.key] = entry
        return True

    def close(self, source_persona: str, code: str) -> bool:
        """Remove a risk. Returns False if it was not open."""
        return self._entries.pop((source_persona, code), None) is not None

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[RiskEntry]:
        return iter(
            sorted(
                self._entries.values(),
                key=lambda e: (-int(e.severity), e.source_persona, e.code),
            )
        )

    @property
    def max_severity(self) -> Severity | None:
        if not self._entries:
            return None
        return max(e.severity for e in self._entries.values())

    @property
    def total_exposure(self) -> float:
        return sum(e.financial_exposure for e in self._entries.values())

    def to_list(self) -> list[dict[str, Any]]:
        return [entry.to_dict() for entry in self]


class KnowledgeState:
    """Versioned key/value context. Every write bumps the version."""

    def __init__(self, initial: Mapping[str, str] | None = None) -> None:
        self._data: dict[str, str] = {}
        self._version = 0
        if initial:
            for key, value in initial.items():
                self.set(key, value)

    @property
    def version(self) -> int:
        return self._version

    def get(self, key: str, default: str | None = None) -> str | None:
        return self._data.get(key, default)

    def set(self, key: str, value: str) -> None:
        if not isinstance(key, str) or not key.strip():
            raise LedgerError("knowledge key must be a non-empty string")
        if not isinstance(value, str):
            raise LedgerError(
                f"knowledge value for {key!r} must be a string, got "
                f"{type(value).__name__}"
            )
        if key not in self._data and len(self._data) >= MAX_KNOWLEDGE_KEYS:
            raise LedgerError(
                f"knowledge state is full at {MAX_KNOWLEDGE_KEYS} keys"
            )
        if self._data.get(key) == value:
            return
        self._data[key] = value
        self._version += 1

    def to_dict(self) -> dict[str, Any]:
        return {"version": self._version, "entries": dict(sorted(self._data.items()))}


@dataclass
class ProjectState:
    """Everything the council reads from and writes back to."""

    milestones: dict[str, Milestone] = field(default_factory=dict)
    risks: RiskRegister = field(default_factory=RiskRegister)
    knowledge: KnowledgeState = field(default_factory=KnowledgeState)

    def add_milestone(self, milestone: Milestone) -> None:
        if milestone.id in self.milestones:
            raise LedgerError(f"duplicate milestone id {milestone.id!r}")
        if len(self.milestones) >= MAX_MILESTONES:
            raise LedgerError(
                f"milestone list is full at {MAX_MILESTONES} entries"
            )
        self.milestones[milestone.id] = milestone

    def set_milestone_status(
        self, milestone_id: str, status: MilestoneStatus
    ) -> Milestone:
        try:
            current = self.milestones[milestone_id]
        except KeyError as exc:
            raise LedgerError(f"no milestone with id {milestone_id!r}") from exc
        updated = replace(current, status=MilestoneStatus.parse(status))
        self.milestones[milestone_id] = updated
        return updated

    def to_dict(self) -> dict[str, Any]:
        return {
            "milestones": [
                m.to_dict() for m in sorted(self.milestones.values(), key=lambda m: m.id)
            ],
            "risks": self.risks.to_list(),
            "knowledge": self.knowledge.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProjectState":
        if not isinstance(payload, Mapping):
            raise LedgerError(
                f"project state must be an object, got {type(payload).__name__}"
            )
        raw_milestones = payload.get("milestones", [])
        if not isinstance(raw_milestones, list):
            raise LedgerError("project state 'milestones' must be a list")

        state = cls()
        for item in raw_milestones:
            state.add_milestone(Milestone.from_dict(item))

        raw_knowledge = payload.get("knowledge", {})
        if not isinstance(raw_knowledge, Mapping):
            raise LedgerError("project state 'knowledge' must be an object")
        entries = raw_knowledge.get("entries", raw_knowledge)
        if not isinstance(entries, Mapping):
            raise LedgerError("project state knowledge 'entries' must be an object")
        for key, value in entries.items():
            state.knowledge.set(str(key), str(value))

        return state
