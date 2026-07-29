"""Test suite for the autonomous critic council."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from config import Persona, Rule, load_governance_rules, load_personas
from erp_state_ledger import (
    KnowledgeState,
    Milestone,
    MilestoneStatus,
    ProjectState,
    RiskEntry,
    RiskRegister,
)
from main import Council, build_council, load_knowledge_base, main, render_text
from models import (
    ConfigError,
    Critique,
    Finding,
    LedgerError,
    ReviewError,
    ReviewRequest,
    Severity,
    Verdict,
)
from reviewers import (
    CRITIQUE_SCHEMA,
    ClaudeReviewer,
    GeminiReviewer,
    RuleBasedReviewer,
    _gemini_schema,
    build_reviewers,
)
from state_manager import ConsensusLedger, GovernanceRules

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "agents_config.json"

CLEAN_DOC = "We propose a scheduled report summarizing weekly ticket volume."
BLOCKING_DOC = "Deploy the scripts directly to production every Friday night."
ADVISORY_DOC = "This world-class rollout includes a manual step for handoff."


def make_persona(name="Auditor", blocking=False, rules=()):
    return Persona(
        name=name,
        title=f"{name} Title",
        blocking=blocking,
        system_prompt="Review the document.",
        rules=tuple(rules),
    )


def make_rule(code="R1", pattern="risky", severity=Severity.HIGH, cost=0.0):
    return Rule.from_dict(
        {
            "code": code,
            "pattern": pattern,
            "severity": severity.label,
            "message": f"{code} fired",
            "financial_cost": cost,
        },
        "TestPersona",
    )


class StubReviewer:
    """A reviewer whose behaviour the test controls exactly."""

    backend = "stub"

    def __init__(self, name, findings=(), raises=None, delay=0.0):
        self.name = name
        self._findings = tuple(findings)
        self._raises = raises
        self._delay = delay
        self.calls = 0

    def review(self, request):
        self.calls += 1
        if self._delay:
            time.sleep(self._delay)
        if self._raises is not None:
            raise self._raises
        return Critique(persona=self.name, findings=self._findings, backend=self.backend)


def ledger_for(reviewers, blocking=()):
    return ConsensusLedger(
        expected_personas=[r.name for r in reviewers], blocking_personas=blocking
    )


class TestSeverityAndVerdict:
    def test_severity_is_ordered(self):
        assert Severity.CRITICAL > Severity.HIGH > Severity.MEDIUM > Severity.LOW

    @pytest.mark.parametrize("text,expected", [("high", Severity.HIGH), ("  CRITICAL ", Severity.CRITICAL)])
    def test_parse_accepts_case_and_whitespace(self, text, expected):
        assert Severity.parse(text) is expected

    @pytest.mark.parametrize("bad", ["urgent", "", "None"])
    def test_parse_rejects_unknown_names(self, bad):
        with pytest.raises(ConfigError, match="unknown severity"):
            Severity.parse(bad)

    @pytest.mark.parametrize("bad", [None, 3, 2.5, ["high"]])
    def test_parse_rejects_non_strings(self, bad):
        with pytest.raises(ConfigError, match="must be a string"):
            Severity.parse(bad)

    def test_verdict_is_ordered_by_restrictiveness(self):
        assert Verdict.BLOCKED > Verdict.APPROVED_WITH_CONDITIONS > Verdict.APPROVED


class TestFinding:
    def test_round_trips_to_dict(self):
        finding = Finding("C1", Severity.HIGH, "bad", "excerpt", 10.0)
        assert finding.to_dict()["severity"] == "high"

    @pytest.mark.parametrize("code,message", [("", "m"), ("   ", "m"), ("C", ""), ("C", "  ")])
    def test_rejects_blank_strings(self, code, message):
        with pytest.raises(ReviewError):
            Finding(code, Severity.LOW, message)

    @pytest.mark.parametrize("cost", [-1.0, float("inf"), float("nan"), "5", True, None])
    def test_rejects_bad_financial_cost(self, cost):
        with pytest.raises(ReviewError):
            Finding("C", Severity.LOW, "m", financial_cost=cost)

    def test_rejects_non_severity(self):
        with pytest.raises(ReviewError, match="must be a Severity"):
            Finding("C", "high", "m")

    def test_is_immutable(self):
        finding = Finding("C", Severity.LOW, "m")
        with pytest.raises(AttributeError):
            finding.code = "other"


class TestCritique:
    def test_max_severity_and_cost_aggregate(self):
        critique = Critique(
            "P",
            (
                Finding("A", Severity.LOW, "m", financial_cost=5.0),
                Finding("B", Severity.CRITICAL, "m", financial_cost=7.5),
            ),
        )
        assert critique.max_severity is Severity.CRITICAL
        assert critique.financial_cost == 12.5

    def test_empty_critique_has_no_max_severity(self):
        assert Critique("P").max_severity is None

    def test_failure_factory_marks_failed(self):
        critique = Critique.failure("P", "boom")
        assert critique.failed and critique.error == "boom" and critique.findings == ()

    def test_failed_critique_cannot_carry_findings(self):
        with pytest.raises(ReviewError, match="must not also carry findings"):
            Critique("P", (Finding("A", Severity.LOW, "m"),), error="boom")

    def test_rejects_blank_persona(self):
        with pytest.raises(ReviewError, match="non-empty string"):
            Critique("  ")


class TestReviewRequest:
    def test_accepts_a_normal_document(self):
        assert ReviewRequest(CLEAN_DOC).document == CLEAN_DOC

    @pytest.mark.parametrize("doc", ["", "   ", "\n\t"])
    def test_rejects_blank_documents(self, doc):
        with pytest.raises(ReviewError, match="must not be blank"):
            ReviewRequest(doc)

    @pytest.mark.parametrize("doc", [None, 42, b"bytes", ["text"]])
    def test_rejects_non_string_documents(self, doc):
        with pytest.raises(ReviewError, match="must be a string"):
            ReviewRequest(doc)

    def test_rejects_oversized_document(self):
        with pytest.raises(ReviewError, match="exceeds"):
            ReviewRequest("x" * 200_001)

    def test_rejects_non_string_governance_rules(self):
        with pytest.raises(ReviewError, match="governance_rules"):
            ReviewRequest(CLEAN_DOC, governance_rules=["a"])


class TestConfigLoading:
    def test_shipped_config_defines_six_personas(self):
        personas = load_personas(CONFIG_PATH)
        assert len(personas) == 6
        assert sum(1 for p in personas if p.blocking) == 3

    def test_every_shipped_rule_compiles(self):
        for persona in load_personas(CONFIG_PATH):
            assert persona.rules, f"{persona.name} has no rules"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ConfigError, match="no persona config"):
            load_personas(tmp_path / "absent.json")

    def test_invalid_json_raises(self, tmp_path):
        path = tmp_path / "c.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(ConfigError, match="invalid JSON"):
            load_personas(path)

    def test_non_object_root_raises(self, tmp_path):
        path = tmp_path / "c.json"
        path.write_text("[1, 2]", encoding="utf-8")
        with pytest.raises(ConfigError, match="must contain an object"):
            load_personas(path)

    def test_zero_personas_raises(self, tmp_path):
        path = tmp_path / "c.json"
        path.write_text(json.dumps({"personas": []}), encoding="utf-8")
        with pytest.raises(ConfigError, match="zero personas"):
            load_personas(path)

    def test_invalid_regex_raises_at_load_not_at_review(self, tmp_path):
        path = tmp_path / "c.json"
        path.write_text(
            json.dumps(
                {
                    "personas": [
                        {
                            "name": "P",
                            "title": "T",
                            "system_prompt": "S",
                            "rules": [
                                {
                                    "code": "R",
                                    "pattern": "([unclosed",
                                    "severity": "high",
                                    "message": "m",
                                }
                            ],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ConfigError, match="invalid regex"):
            load_personas(path)

    def test_duplicate_persona_names_raise(self, tmp_path):
        persona = {"name": "P", "title": "T", "system_prompt": "S"}
        path = tmp_path / "c.json"
        path.write_text(json.dumps({"personas": [persona, persona]}), encoding="utf-8")
        with pytest.raises(ConfigError, match="duplicate persona names"):
            load_personas(path)

    def test_duplicate_rule_codes_raise(self, tmp_path):
        rule = {"code": "R", "pattern": "x", "severity": "low", "message": "m"}
        path = tmp_path / "c.json"
        path.write_text(
            json.dumps(
                {"personas": [{"name": "P", "title": "T", "system_prompt": "S", "rules": [rule, rule]}]}
            ),
            encoding="utf-8",
        )
        with pytest.raises(ConfigError, match="duplicate rule codes"):
            load_personas(path)

    @pytest.mark.parametrize("name", ["", "9lives", "has space", "has-dash", "x" * 65])
    def test_invalid_persona_names_raise(self, tmp_path, name):
        path = tmp_path / "c.json"
        path.write_text(
            json.dumps({"personas": [{"name": name, "title": "T", "system_prompt": "S"}]}),
            encoding="utf-8",
        )
        with pytest.raises(ConfigError, match="persona name"):
            load_personas(path)

    def test_non_bool_blocking_raises(self, tmp_path):
        path = tmp_path / "c.json"
        path.write_text(
            json.dumps(
                {"personas": [{"name": "P", "title": "T", "system_prompt": "S", "blocking": "yes"}]}
            ),
            encoding="utf-8",
        )
        with pytest.raises(ConfigError, match="blocking must be a bool"):
            load_personas(path)

    def test_negative_rule_cost_raises(self):
        with pytest.raises(ConfigError, match="must not be negative"):
            Rule.from_dict(
                {"code": "R", "pattern": "x", "severity": "low", "message": "m", "financial_cost": -1},
                "P",
            )

    def test_governance_rules_none_returns_empty(self):
        assert load_governance_rules(None) == ""

    def test_governance_rules_missing_file_raises(self, tmp_path):
        with pytest.raises(ConfigError, match="no governance rules file"):
            load_governance_rules(tmp_path / "absent.txt")

    def test_shipped_standards_file_loads(self):
        assert "SUBSTANTIATION" in load_governance_rules(
            REPO_ROOT / "brand_compliance_standards.txt"
        )


class TestRuleBasedReviewer:
    def test_fires_matching_rule_with_evidence(self):
        reviewer = RuleBasedReviewer(make_persona(rules=[make_rule(pattern="risky")]))
        critique = reviewer.review(ReviewRequest("This plan is risky indeed."))
        assert len(critique.findings) == 1
        assert "risky" in critique.findings[0].evidence

    def test_clean_document_produces_no_findings(self):
        reviewer = RuleBasedReviewer(make_persona(rules=[make_rule(pattern="risky")]))
        assert reviewer.review(ReviewRequest(CLEAN_DOC)).findings == ()

    def test_matching_is_case_insensitive(self):
        reviewer = RuleBasedReviewer(make_persona(rules=[make_rule(pattern="risky")]))
        assert reviewer.review(ReviewRequest("Totally RISKY move.")).findings

    def test_findings_are_sorted_most_severe_first(self):
        persona = make_persona(
            rules=[
                make_rule("LOW1", "alpha", Severity.LOW),
                make_rule("CRIT1", "beta", Severity.CRITICAL),
            ]
        )
        critique = RuleBasedReviewer(persona).review(ReviewRequest("alpha and beta"))
        assert [f.code for f in critique.findings] == ["CRIT1", "LOW1"]

    def test_shipped_compliance_rule_catches_direct_prod_push(self):
        personas = {p.name: p for p in load_personas(CONFIG_PATH)}
        critique = RuleBasedReviewer(personas["ComplianceAuditor"]).review(
            ReviewRequest(BLOCKING_DOC)
        )
        assert any(f.code == "CMP_UNREVIEWED_PROD_PUSH" for f in critique.findings)

    def test_backend_is_reported(self):
        critique = RuleBasedReviewer(make_persona()).review(ReviewRequest(CLEAN_DOC))
        assert critique.backend == "rules"


class TestGovernanceRules:
    def test_sync_bumps_version_and_keeps_history(self):
        rules = GovernanceRules(text="v1 text")
        updated = rules.sync("v2 text")
        assert updated.version == 2 and updated.history == ("v1 text",)

    def test_sync_with_identical_text_is_a_noop(self):
        rules = GovernanceRules(text="same")
        assert rules.sync("same") is rules

    def test_sync_rejects_blank_text(self):
        with pytest.raises(LedgerError, match="non-empty"):
            GovernanceRules(text="v1").sync("   ")

    def test_version_must_be_positive(self):
        with pytest.raises(LedgerError, match=">= 1"):
            GovernanceRules(text="x", version=0)

    def test_ledger_sync_is_visible_in_snapshot(self):
        ledger = ConsensusLedger(["P"])
        ledger.sync_governance_rules("new standards")
        ledger.record(Critique("P"))
        assert ledger.snapshot().governance_version == 2


class TestConsensusLedger:
    def test_all_clean_approves(self):
        ledger = ConsensusLedger(["A", "B"])
        ledger.record(Critique("A"))
        ledger.record(Critique("B"))
        snapshot = ledger.snapshot()
        assert snapshot.verdict is Verdict.APPROVED
        assert snapshot.compliance_adherence is True

    def test_missing_persona_blocks_and_marks_degraded(self):
        ledger = ConsensusLedger(["A", "B"])
        ledger.record(Critique("A"))
        snapshot = ledger.snapshot()
        assert snapshot.verdict is Verdict.BLOCKED
        assert snapshot.degraded and snapshot.missing_personas == ("B",)

    def test_failed_reviewer_blocks_even_with_no_findings(self):
        ledger = ConsensusLedger(["A", "B"])
        ledger.record(Critique("A"))
        ledger.record(Critique.failure("B", "timeout"))
        snapshot = ledger.snapshot()
        assert snapshot.verdict is Verdict.BLOCKED
        assert snapshot.failed_personas == ("B",)
        assert snapshot.compliance_adherence is False

    def test_high_finding_from_blocking_persona_blocks(self):
        ledger = ConsensusLedger(["A"], blocking_personas=["A"])
        ledger.record(Critique("A", (Finding("F", Severity.HIGH, "m"),)))
        assert ledger.snapshot().verdict is Verdict.BLOCKED

    def test_high_finding_from_advisory_persona_only_adds_conditions(self):
        ledger = ConsensusLedger(["A"], blocking_personas=[])
        ledger.record(Critique("A", (Finding("F", Severity.HIGH, "m"),)))
        snapshot = ledger.snapshot()
        assert snapshot.verdict is Verdict.APPROVED_WITH_CONDITIONS
        assert snapshot.compliance_adherence is True

    def test_medium_finding_from_blocking_persona_does_not_block(self):
        ledger = ConsensusLedger(["A"], blocking_personas=["A"])
        ledger.record(Critique("A", (Finding("F", Severity.MEDIUM, "m"),)))
        assert ledger.snapshot().verdict is Verdict.APPROVED_WITH_CONDITIONS

    def test_financial_impact_sums_across_personas(self):
        ledger = ConsensusLedger(["A", "B"])
        ledger.record(Critique("A", (Finding("F", Severity.LOW, "m", financial_cost=100.0),)))
        ledger.record(Critique("B", (Finding("G", Severity.LOW, "m", financial_cost=50.5),)))
        assert ledger.snapshot().financial_impact == pytest.approx(150.5)

    def test_duplicate_critique_raises(self):
        ledger = ConsensusLedger(["A"])
        ledger.record(Critique("A"))
        with pytest.raises(LedgerError, match="duplicate critique"):
            ledger.record(Critique("A"))

    def test_unknown_persona_raises(self):
        ledger = ConsensusLedger(["A"])
        with pytest.raises(LedgerError, match="unexpected persona"):
            ledger.record(Critique("Z"))

    def test_non_critique_raises(self):
        with pytest.raises(LedgerError, match="expected a Critique"):
            ConsensusLedger(["A"]).record({"persona": "A"})

    def test_empty_roster_raises(self):
        with pytest.raises(LedgerError, match="at least one expected persona"):
            ConsensusLedger([])

    def test_duplicate_roster_entries_raise(self):
        with pytest.raises(LedgerError, match="duplicate expected personas"):
            ConsensusLedger(["A", "A"])

    def test_blocking_persona_outside_roster_raises(self):
        with pytest.raises(LedgerError, match="not in the roster"):
            ConsensusLedger(["A"], blocking_personas=["Z"])

    def test_concurrent_records_are_all_retained(self):
        names = [f"P{i}" for i in range(50)]
        ledger = ConsensusLedger(names)
        barrier = threading.Barrier(len(names))

        def record(name):
            barrier.wait()
            ledger.record(Critique(name))

        threads = [threading.Thread(target=record, args=(n,)) for n in names]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        snapshot = ledger.snapshot()
        assert len(snapshot.critiques) == 50
        assert snapshot.verdict is Verdict.APPROVED

    def test_snapshot_is_ordered_by_roster(self):
        ledger = ConsensusLedger(["A", "B", "C"])
        for name in ("C", "A", "B"):
            ledger.record(Critique(name))
        assert [c.persona for c in ledger.snapshot().critiques] == ["A", "B", "C"]


class TestErpStateLedger:
    def test_risk_register_deduplicates_by_persona_and_code(self):
        register = RiskRegister()
        entry = RiskEntry("R", Severity.HIGH, "d", "P")
        assert register.add(entry) is True
        assert register.add(RiskEntry("R", Severity.HIGH, "d", "P")) is False
        assert len(register) == 1

    def test_same_code_from_different_personas_is_two_risks(self):
        register = RiskRegister()
        register.add(RiskEntry("R", Severity.HIGH, "d", "P1"))
        register.add(RiskEntry("R", Severity.HIGH, "d", "P2"))
        assert len(register) == 2

    def test_register_reports_max_severity_and_exposure(self):
        register = RiskRegister(
            [
                RiskEntry("A", Severity.LOW, "d", "P1", 10.0),
                RiskEntry("B", Severity.CRITICAL, "d", "P2", 5.0),
            ]
        )
        assert register.max_severity is Severity.CRITICAL
        assert register.total_exposure == 15.0

    def test_empty_register_has_no_max_severity(self):
        assert RiskRegister().max_severity is None

    def test_close_removes_a_risk(self):
        register = RiskRegister([RiskEntry("R", Severity.LOW, "d", "P")])
        assert register.close("P", "R") is True
        assert register.close("P", "R") is False

    def test_register_rejects_non_entries(self):
        with pytest.raises(LedgerError, match="expected a RiskEntry"):
            RiskRegister().add("not a risk")

    def test_knowledge_state_versions_only_on_change(self):
        state = KnowledgeState()
        state.set("k", "v")
        assert state.version == 1
        state.set("k", "v")
        assert state.version == 1
        state.set("k", "v2")
        assert state.version == 2

    @pytest.mark.parametrize("key,value", [("", "v"), ("  ", "v"), ("k", 5), ("k", None)])
    def test_knowledge_state_rejects_bad_input(self, key, value):
        with pytest.raises(LedgerError):
            KnowledgeState().set(key, value)

    def test_milestone_status_transitions(self):
        state = ProjectState()
        state.add_milestone(Milestone("M1", "Ship it"))
        updated = state.set_milestone_status("M1", MilestoneStatus.BLOCKED)
        assert updated.status is MilestoneStatus.BLOCKED

    def test_duplicate_milestone_raises(self):
        state = ProjectState()
        state.add_milestone(Milestone("M1", "a"))
        with pytest.raises(LedgerError, match="duplicate milestone"):
            state.add_milestone(Milestone("M1", "b"))

    def test_unknown_milestone_raises(self):
        with pytest.raises(LedgerError, match="no milestone with id"):
            ProjectState().set_milestone_status("nope", MilestoneStatus.COMPLETE)

    def test_unknown_status_raises(self):
        with pytest.raises(LedgerError, match="unknown milestone status"):
            MilestoneStatus.parse("almost")

    def test_project_state_round_trips(self):
        state = ProjectState()
        state.add_milestone(Milestone("M1", "Ship", MilestoneStatus.IN_PROGRESS, "ops"))
        state.knowledge.set("k", "v")
        restored = ProjectState.from_dict(state.to_dict())
        assert restored.to_dict()["milestones"] == state.to_dict()["milestones"]
        assert restored.knowledge.get("k") == "v"


class TestLedgerAppliesToState:
    def test_findings_become_risks(self):
        ledger = ConsensusLedger(["A"])
        ledger.record(Critique("A", (Finding("F", Severity.HIGH, "bad", financial_cost=25.0),)))
        state = ProjectState()
        ledger.apply_to(state)
        assert len(state.risks) == 1
        assert state.risks.total_exposure == 25.0

    def test_verdict_is_written_to_knowledge_state(self):
        ledger = ConsensusLedger(["A"])
        ledger.record(Critique("A"))
        state = ProjectState()
        ledger.apply_to(state)
        assert state.knowledge.get("last_verdict") == "APPROVED"
        assert state.knowledge.get("run_degraded") == "false"

    def test_degraded_run_is_recorded(self):
        ledger = ConsensusLedger(["A", "B"])
        ledger.record(Critique.failure("A", "boom"))
        ledger.record(Critique("B"))
        state = ProjectState()
        ledger.apply_to(state)
        assert state.knowledge.get("run_degraded") == "true"

    def test_reapplying_the_same_snapshot_is_idempotent(self):
        ledger = ConsensusLedger(["A"])
        ledger.record(Critique("A", (Finding("F", Severity.LOW, "m"),)))
        state = ProjectState()
        snapshot = ledger.apply_to(state)
        ledger.apply_to(state, snapshot)
        assert len(state.risks) == 1

    def test_rejects_non_project_state(self):
        ledger = ConsensusLedger(["A"])
        ledger.record(Critique("A"))
        with pytest.raises(LedgerError, match="expected a ProjectState"):
            ledger.apply_to({})


class TestCouncilExecution:
    def test_runs_every_reviewer_once(self):
        reviewers = [StubReviewer(f"P{i}") for i in range(6)]
        council = Council(reviewers, ledger_for(reviewers))
        council.review(ReviewRequest(CLEAN_DOC))
        assert all(r.calls == 1 for r in reviewers)

    def test_reviewers_actually_run_in_parallel(self):
        reviewers = [StubReviewer(f"P{i}", delay=0.20) for i in range(6)]
        council = Council(reviewers, ledger_for(reviewers), timeout_seconds=10)
        started = time.monotonic()
        council.review(ReviewRequest(CLEAN_DOC))
        elapsed = time.monotonic() - started
        assert elapsed < 0.60, f"serial execution suspected ({elapsed:.2f}s for 6x0.2s)"

    def test_clean_run_approves(self):
        reviewers = [StubReviewer("A"), StubReviewer("B")]
        council = Council(reviewers, ledger_for(reviewers))
        assert council.review(ReviewRequest(CLEAN_DOC)).verdict is Verdict.APPROVED

    def test_reviewer_exception_becomes_a_failed_critique_not_an_approval(self):
        reviewers = [StubReviewer("A"), StubReviewer("B", raises=ReviewError("api down"))]
        council = Council(reviewers, ledger_for(reviewers))
        snapshot = council.review(ReviewRequest(CLEAN_DOC))
        assert snapshot.verdict is Verdict.BLOCKED
        assert snapshot.failed_personas == ("B",)
        assert "api down" in dict(
            (c.persona, c.error) for c in snapshot.critiques
        )["B"]

    def test_unexpected_exception_is_also_contained(self):
        reviewers = [StubReviewer("A", raises=ZeroDivisionError("nope"))]
        council = Council(reviewers, ledger_for(reviewers))
        snapshot = council.review(ReviewRequest(CLEAN_DOC))
        assert snapshot.verdict is Verdict.BLOCKED
        assert "ZeroDivisionError" in snapshot.critiques[0].error

    def test_one_slow_reviewer_does_not_stall_the_others(self):
        reviewers = [StubReviewer("Fast"), StubReviewer("Slow", delay=1.0)]
        council = Council(reviewers, ledger_for(reviewers), timeout_seconds=0.15)
        started = time.monotonic()
        snapshot = council.review(ReviewRequest(CLEAN_DOC))
        elapsed = time.monotonic() - started
        assert elapsed < 0.8, "the council waited on the hung reviewer"
        assert snapshot.verdict is Verdict.BLOCKED
        assert snapshot.failed_personas == ("Slow",)
        assert "timed out" in snapshot.critiques[1].error

    def test_blocking_finding_blocks(self):
        reviewers = [StubReviewer("A", findings=(Finding("F", Severity.CRITICAL, "m"),))]
        council = Council(reviewers, ledger_for(reviewers, blocking=["A"]))
        assert council.review(ReviewRequest(CLEAN_DOC)).verdict is Verdict.BLOCKED

    def test_empty_reviewer_list_raises(self):
        with pytest.raises(ConfigError, match="at least one reviewer"):
            Council([], ConsensusLedger(["A"]))

    @pytest.mark.parametrize("timeout", [0, -1, 3601, "10", True, None])
    def test_invalid_timeout_raises(self, timeout):
        reviewers = [StubReviewer("A")]
        with pytest.raises(ConfigError):
            Council(reviewers, ledger_for(reviewers), timeout_seconds=timeout)


class TestBuildReviewers:
    def test_rules_backend_needs_no_credentials(self):
        personas = load_personas(CONFIG_PATH)
        reviewers = build_reviewers(personas, "rules")
        assert len(reviewers) == 6
        assert all(r.backend == "rules" for r in reviewers)

    def test_unknown_backend_raises(self):
        with pytest.raises(ReviewError, match="unknown backend"):
            build_reviewers([make_persona()], "gpt")

    def test_build_council_marks_blocking_personas(self):
        council = build_council(load_personas(CONFIG_PATH), "rules")
        snapshot = council.review(ReviewRequest(BLOCKING_DOC))
        assert snapshot.verdict is Verdict.BLOCKED

    def test_advisory_only_findings_yield_conditions(self):
        council = build_council(load_personas(CONFIG_PATH), "rules")
        snapshot = council.review(ReviewRequest(ADVISORY_DOC))
        assert snapshot.verdict is Verdict.APPROVED_WITH_CONDITIONS

    def test_clean_document_approves_end_to_end(self):
        council = build_council(load_personas(CONFIG_PATH), "rules")
        assert council.review(ReviewRequest(CLEAN_DOC)).verdict is Verdict.APPROVED


class TestClaudeReviewer:
    def _client(self, payload=None, stop_reason="end_turn", raises=None):
        text = json.dumps(payload if payload is not None else {"findings": []})

        def create(**kwargs):
            if raises is not None:
                raise raises
            return SimpleNamespace(
                stop_reason=stop_reason,
                stop_details=SimpleNamespace(category="cyber"),
                content=[SimpleNamespace(type="text", text=text)],
            )

        return SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))

    def test_parses_findings(self):
        payload = {
            "findings": [
                {
                    "code": "SEC_1",
                    "severity": "critical",
                    "message": "leaked key",
                    "evidence": "api_key=abc",
                    "financial_cost": 1000,
                }
            ]
        }
        reviewer = ClaudeReviewer(make_persona(), client=self._client(payload))
        critique = reviewer.review(ReviewRequest(CLEAN_DOC))
        assert critique.backend == "claude"
        assert critique.findings[0].severity is Severity.CRITICAL

    def test_empty_findings_is_a_clean_critique(self):
        reviewer = ClaudeReviewer(make_persona(), client=self._client())
        assert reviewer.review(ReviewRequest(CLEAN_DOC)).findings == ()

    def test_refusal_raises_rather_than_approving(self):
        reviewer = ClaudeReviewer(make_persona(), client=self._client(stop_reason="refusal"))
        with pytest.raises(ReviewError, match="declined"):
            reviewer.review(ReviewRequest(CLEAN_DOC))

    def test_invalid_json_raises(self):
        client = SimpleNamespace(
            beta=SimpleNamespace(
                messages=SimpleNamespace(
                    create=lambda **kw: SimpleNamespace(
                        stop_reason="end_turn",
                        content=[SimpleNamespace(type="text", text="not json")],
                    )
                )
            )
        )
        with pytest.raises(ReviewError, match="not valid JSON"):
            ClaudeReviewer(make_persona(), client=client).review(ReviewRequest(CLEAN_DOC))

    def test_empty_content_raises(self):
        client = SimpleNamespace(
            beta=SimpleNamespace(
                messages=SimpleNamespace(
                    create=lambda **kw: SimpleNamespace(stop_reason="end_turn", content=[])
                )
            )
        )
        with pytest.raises(ReviewError, match="no content"):
            ClaudeReviewer(make_persona(), client=client).review(ReviewRequest(CLEAN_DOC))

    def test_malformed_finding_raises(self):
        payload = {"findings": [{"code": "X", "severity": "urgent", "message": "m"}]}
        reviewer = ClaudeReviewer(make_persona(), client=self._client(payload))
        with pytest.raises(ReviewError, match="malformed finding"):
            reviewer.review(ReviewRequest(CLEAN_DOC))

    def test_findings_must_be_a_list(self):
        reviewer = ClaudeReviewer(make_persona(), client=self._client({"findings": {}}))
        with pytest.raises(ReviewError, match="must be a list"):
            reviewer.review(ReviewRequest(CLEAN_DOC))

    def test_sdk_errors_are_translated(self):
        class RateLimitError(Exception):
            pass

        reviewer = ClaudeReviewer(
            make_persona(), client=self._client(raises=RateLimitError("429"))
        )
        with pytest.raises(ReviewError, match="rate limited"):
            reviewer.review(ReviewRequest(CLEAN_DOC))

    def test_request_carries_schema_and_fallbacks(self):
        captured = {}

        def create(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                stop_reason="end_turn",
                content=[SimpleNamespace(type="text", text='{"findings": []}')],
            )

        client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))
        ClaudeReviewer(make_persona(), client=client).review(ReviewRequest(CLEAN_DOC))
        assert captured["model"] == "claude-opus-5"
        assert captured["fallbacks"] == "default"
        assert captured["output_config"]["format"]["schema"] is CRITIQUE_SCHEMA


class TestGeminiReviewer:
    def _client(self, text='{"findings": []}'):
        return SimpleNamespace(
            models=SimpleNamespace(
                generate_content=lambda **kw: SimpleNamespace(text=text)
            )
        )

    def test_parses_findings(self):
        payload = json.dumps(
            {"findings": [{"code": "OPS_1", "severity": "low", "message": "m", "evidence": "e", "financial_cost": 0}]}
        )
        critique = GeminiReviewer(make_persona(), client=self._client(payload)).review(
            ReviewRequest(CLEAN_DOC)
        )
        assert critique.backend == "gemini"
        assert critique.findings[0].code == "OPS_1"

    def test_empty_text_raises(self):
        with pytest.raises(ReviewError, match="no text"):
            GeminiReviewer(make_persona(), client=self._client("")).review(
                ReviewRequest(CLEAN_DOC)
            )

    def test_invalid_json_raises(self):
        with pytest.raises(ReviewError, match="not valid JSON"):
            GeminiReviewer(make_persona(), client=self._client("nope")).review(
                ReviewRequest(CLEAN_DOC)
            )

    def test_schema_strips_unsupported_keywords(self):
        stripped = _gemini_schema(CRITIQUE_SCHEMA)
        rendered = json.dumps(stripped)
        assert "additionalProperties" not in rendered
        assert stripped["properties"]["findings"]["items"]["required"]
        assert "critical" in stripped["properties"]["findings"]["items"]["properties"]["severity"]["enum"]


class TestKnowledgeBase:
    def test_shipped_knowledge_base_loads(self):
        params = load_knowledge_base(REPO_ROOT / "corporate_knowledge_base.json")
        assert params["engagement_model"]
        assert all(isinstance(v, str) for v in params.values())

    def test_none_path_returns_empty(self):
        assert load_knowledge_base(None) == {}

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ConfigError, match="no knowledge base"):
            load_knowledge_base(tmp_path / "absent.json")

    def test_invalid_json_raises(self, tmp_path):
        path = tmp_path / "kb.json"
        path.write_text("{", encoding="utf-8")
        with pytest.raises(ConfigError, match="invalid JSON"):
            load_knowledge_base(path)

    def test_non_object_raises(self, tmp_path):
        path = tmp_path / "kb.json"
        path.write_text("[]", encoding="utf-8")
        with pytest.raises(ConfigError, match="must contain an object"):
            load_knowledge_base(path)


class TestRendering:
    def test_degraded_run_is_explained_not_just_blocked(self):
        ledger = ConsensusLedger(["A", "B"])
        ledger.record(Critique("A"))
        ledger.record(Critique.failure("B", "timed out"))
        text = render_text(ledger.snapshot())
        assert "RUN DEGRADED" in text and "timed out" in text

    def test_findings_are_rendered_with_evidence(self):
        ledger = ConsensusLedger(["A"])
        ledger.record(Critique("A", (Finding("F", Severity.HIGH, "bad thing", "the excerpt"),)))
        text = render_text(ledger.snapshot())
        assert "HIGH" in text and "bad thing" in text and "the excerpt" in text

    def test_clean_persona_is_reported(self):
        ledger = ConsensusLedger(["A"])
        ledger.record(Critique("A"))
        assert "no objections" in render_text(ledger.snapshot())


class TestCli:
    def test_clean_document_exits_zero(self, capsys):
        assert main(["--document", CLEAN_DOC]) == 0
        assert "APPROVED" in capsys.readouterr().out

    def test_advisory_findings_exit_one(self):
        assert main(["--document", ADVISORY_DOC]) == 1

    def test_blocking_findings_exit_two(self):
        assert main(["--document", BLOCKING_DOC]) == 2

    def test_missing_config_exits_three(self, tmp_path, capsys):
        code = main(["--document", CLEAN_DOC, "--config", str(tmp_path / "absent.json")])
        assert code == 3
        assert "error:" in capsys.readouterr().err

    def test_json_output_is_parseable(self, capsys):
        main(["--document", BLOCKING_DOC, "--format", "json"])
        payload = json.loads(capsys.readouterr().out)
        assert payload["verdict"] == "BLOCKED"
        assert len(payload["critiques"]) == 6

    def test_document_file_is_read(self, tmp_path):
        path = tmp_path / "doc.txt"
        path.write_text(BLOCKING_DOC, encoding="utf-8")
        assert main(["--document-file", str(path)]) == 2

    def test_missing_document_file_exits_three(self, tmp_path):
        assert main(["--document-file", str(tmp_path / "absent.txt")]) == 3

    def test_blank_document_exits_three(self):
        assert main(["--document", "   "]) == 3

    def test_document_and_file_are_mutually_exclusive(self):
        with pytest.raises(SystemExit):
            main(["--document", "x", "--document-file", "y"])

    def test_document_is_required(self):
        with pytest.raises(SystemExit):
            main([])

    def test_standards_and_knowledge_base_are_wired_in(self, capsys):
        code = main(
            [
                "--document",
                BLOCKING_DOC,
                "--standards",
                str(REPO_ROOT / "brand_compliance_standards.txt"),
                "--knowledge-base",
                str(REPO_ROOT / "corporate_knowledge_base.json"),
                "--format",
                "json",
            ]
        )
        assert code == 2
        assert json.loads(capsys.readouterr().out)["governance_version"] == 1
