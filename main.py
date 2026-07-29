"""Execution engine: runs the six council members in parallel over one document.

Exit codes
    0  approved
    1  approved with conditions
    2  blocked
    3  configuration or usage error
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Mapping, Sequence

from config import DEFAULT_CONFIG_PATH, Persona, load_governance_rules, load_personas
from erp_state_ledger import ProjectState
from models import (
    ConfigError,
    CouncilError,
    Critique,
    ReviewRequest,
    Verdict,
)
from reviewers import Reviewer, build_reviewers
from state_manager import ConsensusLedger, GovernanceRules, LedgerSnapshot

__all__ = ["Council", "main"]

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 120.0
MAX_TIMEOUT_SECONDS = 3_600.0

EXIT_APPROVED = 0
EXIT_CONDITIONS = 1
EXIT_BLOCKED = 2
EXIT_ERROR = 3

_VERDICT_EXIT = {
    Verdict.APPROVED: EXIT_APPROVED,
    Verdict.APPROVED_WITH_CONDITIONS: EXIT_CONDITIONS,
    Verdict.BLOCKED: EXIT_BLOCKED,
}


class Council:
    """Fans one document out to every reviewer concurrently."""

    def __init__(
        self,
        reviewers: Sequence[Reviewer],
        ledger: ConsensusLedger,
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        if not reviewers:
            raise ConfigError("a council needs at least one reviewer")
        if isinstance(timeout_seconds, bool) or not isinstance(
            timeout_seconds, (int, float)
        ):
            raise ConfigError("timeout_seconds must be a number")
        if not 0 < timeout_seconds <= MAX_TIMEOUT_SECONDS:
            raise ConfigError(
                f"timeout_seconds must be in (0, {MAX_TIMEOUT_SECONDS}], "
                f"got {timeout_seconds}"
            )
        self.reviewers = tuple(reviewers)
        self.ledger = ledger
        self.timeout_seconds = float(timeout_seconds)

    def review(self, request: ReviewRequest) -> LedgerSnapshot:
        """Run every reviewer in parallel and return the consensus snapshot."""
        started = time.monotonic()
        pool = ThreadPoolExecutor(
            max_workers=len(self.reviewers), thread_name_prefix="council"
        )
        try:
            futures: dict[Future[Critique], Reviewer] = {
                pool.submit(reviewer.review, request): reviewer
                for reviewer in self.reviewers
            }
            remaining = self.timeout_seconds - (time.monotonic() - started)
            done, not_done = wait(futures, timeout=max(remaining, 0.0))

            for future in done:
                reviewer = futures[future]
                try:
                    critique = future.result()
                except CouncilError as exc:
                    logger.warning("%s failed: %s", reviewer.name, exc)
                    critique = Critique.failure(
                        reviewer.name, str(exc), reviewer.backend
                    )
                except Exception as exc:  # noqa: BLE001 - never trust a reviewer
                    logger.exception("%s raised unexpectedly", reviewer.name)
                    critique = Critique.failure(
                        reviewer.name,
                        f"unexpected {type(exc).__name__}: {exc}",
                        reviewer.backend,
                    )
                self.ledger.record(critique)

            for future in not_done:
                future.cancel()
                reviewer = futures[future]
                logger.warning(
                    "%s exceeded the %.1fs budget", reviewer.name,
                    self.timeout_seconds,
                )
                self.ledger.record(
                    Critique.failure(
                        reviewer.name,
                        f"timed out after {self.timeout_seconds:.1f}s",
                        reviewer.backend,
                    )
                )
        finally:
            # Do not block on a hung reviewer; the ledger already recorded it.
            pool.shutdown(wait=False, cancel_futures=True)

        snapshot = self.ledger.snapshot()
        logger.info(
            "Council verdict %s (%d critiques, %d degraded, $%.2f exposure)",
            snapshot.verdict.label,
            len(snapshot.critiques),
            len(snapshot.failed_personas) + len(snapshot.missing_personas),
            snapshot.financial_impact,
        )
        return snapshot


def load_knowledge_base(path: str | Path | None) -> dict[str, str]:
    """Read the flat context parameters the council audits against."""
    if path is None:
        return {}
    kb_path = Path(path)
    try:
        text = kb_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ConfigError(f"no knowledge base at {kb_path}") from exc
    except OSError as exc:
        raise ConfigError(f"cannot read {kb_path}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise ConfigError(f"{kb_path} is not valid UTF-8: {exc}") from exc

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{kb_path} contains invalid JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ConfigError(
            f"{kb_path} must contain an object, got {type(payload).__name__}"
        )

    parameters = payload.get("parameters", payload)
    if not isinstance(parameters, Mapping):
        raise ConfigError(f"{kb_path} 'parameters' must be an object")
    return {str(k): str(v) for k, v in parameters.items()}


def build_council(
    personas: Sequence[Persona],
    backend: str,
    *,
    governance_text: str = "",
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    **backend_kwargs: Any,
) -> Council:
    reviewers = build_reviewers(personas, backend, **backend_kwargs)
    ledger = ConsensusLedger(
        expected_personas=[p.name for p in personas],
        blocking_personas=[p.name for p in personas if p.blocking],
        governance=GovernanceRules(text=governance_text),
    )
    return Council(reviewers, ledger, timeout_seconds=timeout_seconds)


def render_text(snapshot: LedgerSnapshot) -> str:
    lines = [f"VERDICT: {snapshot.verdict.label}"]
    if snapshot.degraded:
        lines.append(
            "RUN DEGRADED - the verdict is blocked because the council was "
            "incomplete, not because the document failed review."
        )
        if snapshot.failed_personas:
            lines.append(f"  failed:  {', '.join(snapshot.failed_personas)}")
        if snapshot.missing_personas:
            lines.append(f"  missing: {', '.join(snapshot.missing_personas)}")
    lines.append(f"Financial exposure: ${snapshot.financial_impact:,.2f}")
    lines.append("")

    for critique in snapshot.critiques:
        if critique.failed:
            lines.append(f"[{critique.persona}] ERROR: {critique.error}")
            continue
        if not critique.findings:
            lines.append(f"[{critique.persona}] no objections")
            continue
        lines.append(f"[{critique.persona}] {len(critique.findings)} finding(s)")
        for finding in critique.findings:
            lines.append(
                f"    {finding.severity.label.upper():<8} {finding.code}: "
                f"{finding.message}"
            )
            if finding.evidence:
                lines.append(f"             evidence: {finding.evidence}")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--document", help="Document text to review.")
    source.add_argument(
        "--document-file", help="Path to a UTF-8 file to review."
    )
    parser.add_argument(
        "--config", default=str(DEFAULT_CONFIG_PATH),
        help="Persona roster (default: agents_config.json beside this file).",
    )
    parser.add_argument(
        "--standards", help="Governance standards text injected into every review."
    )
    parser.add_argument(
        "--knowledge-base", help="JSON context parameters to seed project state."
    )
    parser.add_argument(
        "--backend", choices=("rules", "claude", "gemini"), default="rules",
        help="Reviewer backend. 'rules' runs offline with no credentials.",
    )
    parser.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS,
        help="Per-run wall-clock budget in seconds.",
    )
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument(
        "--log-level", default="WARNING",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        if args.document_file is not None:
            try:
                document = Path(args.document_file).read_text(encoding="utf-8")
            except OSError as exc:
                raise ConfigError(f"cannot read {args.document_file}: {exc}") from exc
            except UnicodeDecodeError as exc:
                raise ConfigError(
                    f"{args.document_file} is not valid UTF-8: {exc}"
                ) from exc
        else:
            document = args.document

        personas = load_personas(args.config)
        governance = load_governance_rules(args.standards)
        parameters = load_knowledge_base(args.knowledge_base)

        council = build_council(
            personas,
            args.backend,
            governance_text=governance,
            timeout_seconds=args.timeout,
        )
        snapshot = council.review(
            ReviewRequest(document=document, governance_rules=governance)
        )

        state = ProjectState()
        for key, value in parameters.items():
            state.knowledge.set(key, value)
        council.ledger.apply_to(state, snapshot)
    except CouncilError as exc:
        logger.error("%s", exc)
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    if args.format == "json":
        print(json.dumps(snapshot.to_dict(), indent=2))
    else:
        print(render_text(snapshot))

    return _VERDICT_EXIT[snapshot.verdict]


if __name__ == "__main__":
    sys.exit(main())
