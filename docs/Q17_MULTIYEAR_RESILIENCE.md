# Q17 — Multi-year resilience

Status: shipped 2026-05-16 as a single batch.
Tracking: PROGRAM.md §52.
Tests: `tests/test_q17_multiyear_resilience.py` — 38 tests.
Regression: Q13→Q17 = 240 pass / 0 fail.

Eight observational subsystems that close the surviving gaps for
installing the system once and running it for many years.

## At a glance

| Item     | Module                                          | Default | Surface |
|----------|-------------------------------------------------|---------|---------|
| Q17.1    | `app/warm_spare/`                               | OFF     | Manifest + rsync recipe + claim-canonical state machine |
| Q17.2    | `app/resilience_drills/drills/local_only.py`    | ON      | 6th drill in the Q6 registry |
| Q17.3    | `app/healing/monitors/bit_rot_scan.py`          | ON      | Bit-rot healing monitor |
| Q17.4    | `app/operator_transition/`                      | ON      | Five-phase state machine + successor declaration file |
| Q17.5    | `app/self_model/agreement_ledger.py`            | ON      | JSONL ledger + briefing section |
| Q17.6    | `app/healing/monitors/kb_contradiction.py`      | ON      | KB-contradiction monitor |
| Q17.7    | `app/creativity/synthesis_pass.py`              | ON      | Weekly idle daemon + briefing section |
| Q17.8    | `app/conversation_memory/`                      | ON      | Temporal index + recall agent tool |

All eight emit `q17_landmark` events into the identity continuity
ledger on alert / transition.

## Q17.1 Warm-spare partner-host

Closes the existential gap: today's system is single-host. Fire,
theft, RAID failure end continuity regardless of backups.

This subsystem provides primitives, not automation. The operator
provisions the partner host, configures SSH, and owns the
LaunchAgent that runs rsync. We never store partner-host credentials
in the gateway.

### One-time setup

```bash
# 1. Provision partner host (Raspberry Pi / Mac / €5/mo VPS).
# 2. Make passwordless SSH work from canonical → partner:
ssh-copy-id user@partner-host

# 3. Materialise the manifest:
python -m app.warm_spare manifest

# 4. Configure partner target (via React /cp/settings, or workspace/warm_spare/activation.json):
cat > workspace/warm_spare/activation.json <<EOF
{
  "enabled": true,
  "partner_target": "user@partner-host:~/andrusai-mirror/"
}
EOF

# 5. Install the hourly LaunchAgent:
./scripts/install_warm_spare.sh install
./scripts/install_warm_spare.sh start   # smoke-test one immediate pass
```

### Failover

On the surviving partner host:

```bash
cat workspace/warm_spare/canonical_heartbeat.json   # verify silence
python -c "
from app.warm_spare.failover import claim_canonical
print(claim_canonical('CLAIM CANONICAL'))
"
```

Must type the literal phrase. State transitions to CLAIMING, then
auto-promotes to CANONICAL after a 5-minute confirmation window.

### Activation file vs runtime_settings

The sync script (`scripts/warm_spare_sync.sh`) reads from
`workspace/warm_spare/activation.json` first (operator-owned), then
falls back to `runtime_settings.{warm_spare_enabled, warm_spare_partner_target}`.
The activation file is preferred because the gateway can overwrite
`runtime_settings.json` on any setter call.

## Q17.2 Local-Only Day drill

Sixth drill in the Q6 registry. Risk LOW. Cadence 90 days.

```bash
python -m app.resilience_drills run local_only
```

DRY-RUN: never issues live LLM calls. Probes Ollama TCP + per-vendor
key format. Passes when ≥50% of non-dominant providers are ready;
below that, files a CR for operator attention.

## Q17.3 Bit-rot scan

Daily probe, weekly internal cadence. 11 identity-critical JSONL
files. Records SHA-256 + 1MB-prefix hash + line count baseline at
`workspace/healing/bit_rot_baseline.json`. Detects:

- **shrunk** → ALERT (truncation)
- **inplace_mutated** → ALERT (bit flip / silent rewrite)
- **prefix_mutated** → ALERT (non-append mutation of append-only file)
- **append_ok** / **ok** / **new** → no alert

## Q17.4 Operator-transition protocol

Five phases over operator presence:

| Phase         | Trigger                          |
|---------------|----------------------------------|
| ACTIVE        | recent operator activity         |
| ABSENT_30D    | 30d no operator activity         |
| ABSENT_90D    | 90d                              |
| READ_MOSTLY   | 180d auto, or manually-set       |
| TRANSITIONED  | successor has assumed role       |

Successor declaration at `workspace/operator_transition/successor.json`
is operator-authored, human-read. The system never acts on it.

## Q17.5 Agreement ledger

Counterweight to Goodhart guard. Every proactive suggestion gets
recorded with PENDING; operator response transitions appended.

```python
from app.self_model import record_suggestion, record_response, AgreementResponse

sid = record_suggestion(category="library_radar", summary="adopt rich")
record_response(sid, AgreementResponse.ACCEPTED)

from app.self_model import rolling_rate
print(rolling_rate("library_radar"))
```

90-day rolling window — short streaks can't move the needle. The
daily briefing's `briefing_section()` is the only sanctioned read
site; no subsystem reads the ledger to bias its output toward
acceptance.

## Q17.6 KB contradiction probe

Weekly probe. Samples up to 200 epistemic claims, groups by subject
key, runs pairwise negation-pair check (20-pair lexicon: `is/isn't`,
`>/<`, etc.). Surfaces structural contradictions to Signal + ledger.

Structural, not semantic. Catches the gross cases worth an
operator's attention. Semantic contradiction would need an LLM call
per pair and isn't justified at this volume.

## Q17.7 Cross-subsystem synthesis pass

Weekly idle daemon. Picks 2 random pairs from a 20-entry
`SUBSYSTEM_DESCRIPTORS` list, feeds them to
`concept_blend.blend_concepts`, scores results with `novelty_wrap` +
`aesthetic_score`, persists to
`workspace/creativity/synthesis_candidates.jsonl`.

Top 3 recent surface in the weekly briefing under
"💡 Synthesis candidates this week".

Cost: ~$0.10/week worst case.

## Q17.8 Cross-conversation continuity

```python
from app.conversation_memory import recall

refs = recall("rich logging library", window_months=24, top_k=5)
for r in refs:
    print(f"{r.ts} {r.kind}: {r.preview}")
```

Token-overlap search over `workspace/conversation_memory/index.jsonl`,
which is an incremental scan of `audit.log` with PII redacted
(emails + phone numbers stripped before tokenisation).

Deliberately not a vector index — robust against vendor embedding-
model rotation.

Agent tool: `app.tools.recall_past_conversation.RecallPastConversationTool`.

## Wiring

- 1 new identity-event kind: `q17_landmark`
- 2 new healing monitors (bit_rot_scan, kb_contradiction)
- 1 new resilience drill (local_only)
- 9 new master switches + getters/setters in `runtime_settings.py`
- Boot anchors in `app/healing/__init__.py` for synthesis_pass + local_only
- 38 tests pass; Q13→Q17 cumulative 240 pass / 0 fail
- No TIER_IMMUTABLE files modified

## When to revisit Q17

Re-open only on a specific concrete concern:

1. **A live bit-rot detection.** Audit how it happened + extend the watch list.
2. **A failover-drill failure.** Treat as P0.
3. **A new fallback provider** the local-only drill should probe.
4. **An operator-transition phase change** with unhandled subsystem dependency.
5. **A KB contradiction the probe missed** — extend the negation lexicon.
