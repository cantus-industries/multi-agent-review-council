# Autonomous Critic Council

A framework for running one document past several specialist reviewers **in parallel**, aggregating their findings in a thread-safe ledger, and returning a single verdict with a full audit trail.

The design decision it exists to demonstrate: **an incomplete review cannot be an approval.** If a reviewer crashes, times out, or never reports, the run is marked degraded and the verdict is `BLOCKED` — never `APPROVED` on partial evidence. Most review automation fails the other way, returning a confident default when a component silently drops out.

## What it does

Six configurable personas (Legal, Compliance, Security, Finance, Brand, Operations) each evaluate the same document from their own viewpoint and return structured findings. Three are **blocking** — a HIGH or CRITICAL finding from Legal, Compliance, or Security stops the run. The other three are **advisory** — their findings downgrade the verdict to `APPROVED_WITH_CONDITIONS` but cannot block.

Verdicts:

| Verdict | Meaning | Exit code |
|---|---|---|
| `APPROVED` | Every reviewer reported, none had findings | 0 |
| `APPROVED_WITH_CONDITIONS` | Findings exist, none blocking | 1 |
| `BLOCKED` | A blocking finding, **or** the council was incomplete | 2 |
| — | Configuration or usage error | 3 |

## Install and run

```bash
git clone <your-fork-url> && cd multi-agent-review-council
pip install -e ".[dev]"

# Offline, no credentials needed:
python main.py --document "Deploy the scripts directly to production every Friday."
```

```
VERDICT: BLOCKED
Financial exposure: $75,000.00

[ChiefLegalOfficer] no objections
[ComplianceAuditor] 1 finding(s)
    HIGH     CMP_UNREVIEWED_PROD_PUSH: Changes reach production without an evidenced review gate.
             evidence: Deploy the scripts directly to production every Friday.
...
```

Other options:

```bash
python main.py --document-file proposal.md --format json
python main.py --document-file proposal.md \
  --standards brand_compliance_standards.txt \
  --knowledge-base corporate_knowledge_base.json
```

## Reviewer backends

All three satisfy the same `Reviewer` protocol, selected with `--backend`:

| Backend | What it is | Requires |
|---|---|---|
| `rules` *(default)* | Deterministic regex checks from `agents_config.json` | nothing |
| `claude` | Claude with a strict output schema, per persona | `pip install -e ".[claude]"`, `ANTHROPIC_API_KEY` |
| `gemini` | Gemini with a response schema, per persona | `pip install -e ".[gemini]"`, `GOOGLE_API_KEY` |

The `rules` backend is the default because it needs no credentials and runs deterministically — it is what the test suite exercises. The LLM backends are the ones you would run in production; they are structurally identical from the council's point of view, which is the point of the protocol.

## Tests

```bash
pytest
```

156 tests. Beyond the happy paths, the ones that matter:

- `test_reviewers_actually_run_in_parallel` — six 0.2 s reviewers must finish in under 0.6 s, so a regression to a serial loop fails the build.
- `test_one_slow_reviewer_does_not_stall_the_others` — a hung reviewer is abandoned at the deadline; the council still returns.
- `test_missing_persona_blocks_and_marks_degraded` and `test_failed_reviewer_blocks_even_with_no_findings` — the fail-loud rule, asserted directly.
- `test_concurrent_records_are_all_retained` — 50 threads recording through a barrier; nothing is lost.
- `test_invalid_regex_raises_at_load_not_at_review` — bad config fails at startup, not mid-audit.

The LLM backends are tested against injected fake clients (parsing, refusal handling, malformed JSON, SDK error translation). No network calls in the suite.

## Layout

```
main.py                        parallel execution engine + CLI
state_manager.py               ConsensusLedger: aggregation, verdict, governance versioning
erp_state_ledger.py            Milestone / RiskRegister / KnowledgeState / ProjectState
reviewers.py                   Reviewer protocol + rules, Claude, and Gemini backends
config.py                      persona and rule loading with validation
models.py                      typed domain objects and the error hierarchy
agents_config.json             the six personas, their rules, and blocking status
corporate_knowledge_base.json  context parameters seeded into project state
brand_compliance_standards.txt standards text injected into every review
tests/test_council.py          the suite
```

## Configuring your own council

Personas live in `agents_config.json`. Each needs a `name`, `title`, `system_prompt` (used by the LLM backends), optional `blocking` flag, and optional `rules` for the offline backend:

```json
{
  "name": "ComplianceAuditor",
  "title": "Compliance Auditor",
  "blocking": true,
  "system_prompt": "You review documents for control failures...",
  "rules": [
    {
      "code": "CMP_APPROVAL_BYPASS",
      "pattern": "\\b(bypass\\w*|skip\\w*)\\s+(the\\s+)?(review|approval)",
      "severity": "critical",
      "message": "An approval control is explicitly bypassed.",
      "financial_cost": 150000
    }
  ]
}
```

Every regex is compiled at load time, so a malformed pattern fails immediately with the persona and rule code named, rather than throwing mid-review.

## Scope and limits

Stated plainly, because they matter if you are evaluating this:

- **The persona system prompts here are short illustrative stubs.** The tuned production prompts and finding rubrics are proprietary and are omitted from this public reference. The offline rules backend, which the test suite exercises, does not depend on them.
- **The regex backend is a demonstration, not a compliance control.** It matches surface patterns. Real control coverage needs the LLM backends and a rubric you have validated against your own corpus.
- **A timed-out reviewer's thread is abandoned, not killed.** Python cannot kill a thread. The council stops waiting and records the timeout, but the worker runs to completion in the background. For hard resource bounds, run reviewers as separate processes.
- **The financial figures are illustrative.** `financial_cost` on each rule is a placeholder for whatever exposure model you actually use.
- **`ProjectState` is not persisted.** It is built per run. Wiring it to a store is left to the integrator.
- **The Gemini backend is untested against a live endpoint.** It is written to the documented `google-genai` surface and covered by fake-client tests; verify the model ID and response-schema handling against current docs before relying on it.

## License

MIT. Built by **Cantus Industries** — multi-agent systems that audit their own work.
