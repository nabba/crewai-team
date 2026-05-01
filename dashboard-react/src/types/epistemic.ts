// Type definitions for the Epistemic Integrity Layer API.
// Mirrors app/epistemic/{ledger.py,biases.py,api.py} — keep in sync.

export type VerificationStatus =
  | 'verified'
  | 'inferred'
  | 'assumed'
  | 'contradicted';

export type Register = 'declarative' | 'hedged' | 'unverified' | 'internal';

export type Severity = 'low' | 'medium' | 'high' | 'critical';

export type DetectorPhase = 'realtime' | 'posthoc';

export type EvidenceKind =
  | 'tool_call'
  | 'memory_lookup'
  | 'user_assertion'
  | 'prior_claim'
  | 'model_inference';

export interface EvidenceDTO {
  kind: EvidenceKind;
  source_ref: string;
  excerpt: string;
  confidence: number;
}

export interface VerifyingActionDTO {
  tool: string;
  args: Record<string, unknown>;
  expected_signal: string;
  estimated_seconds: number;
  safety: 'read_only';
}

export interface ClaimDTO {
  claim_id: string;
  task_id: string;
  span_id: number | null;
  agent_role: string;
  statement: string;
  status: VerificationStatus;
  register: Register;
  evidence: EvidenceDTO[];
  verifying_action: VerifyingActionDTO | null;
  load_bearing: boolean;
  tags: string[];
  superseded_by: string | null;
  created_at: string;
}

export interface BiasMatchDTO {
  bias_id: string;
  matched_claim_ids: string[];
  severity: Severity;
  detail: Record<string, unknown>;
}

export interface BiasFeedEntry {
  id: number;
  task_id: string;
  claim_id: string;
  bias_id: string;
  severity: Severity;
  matched_claim_ids: string[];
  detail: Record<string, unknown>;
  detected_at: string;
}

export interface BiasDefinitionDTO {
  id: string;
  name: string;
  description: string;
  severity: Severity;
  phase: DetectorPhase;
  corrective_action: string | null;
  blocking: boolean;
}

export interface VerifierShapeDTO {
  id: string;
  tool: string;
  expected_signal: string;
  estimated_seconds: number;
  tags_any: string[];
}

// ── /epistemic/now ─────────────────────────────────────────────────
export interface CalibrationSnapshot {
  factual_grounding: number | null;   // [0, 1] or null when affect not wired
  valence: number | null;
  arousal: number | null;
  attractor: string | null;
}

export interface EpistemicNowReport {
  task_id: string | null;
  ledger: ClaimDTO[] | null;
  load_bearing_count: number;
  unverified_load_bearing_count: number;
  bias_match_count: number;
  calibration: CalibrationSnapshot;
}

// ── /epistemic/feed ────────────────────────────────────────────────
export interface BiasFeedReport {
  window_minutes: number;
  count: number;
  matches: BiasFeedEntry[];
}

// ── /epistemic/biases ──────────────────────────────────────────────
export interface BiasLibraryReport {
  biases: BiasDefinitionDTO[];
}

// ── /epistemic/verifiers ───────────────────────────────────────────
export interface VerifierRegistryReport {
  verifiers: VerifierShapeDTO[];
}

// ── /epistemic/pushback/* ──────────────────────────────────────────
export type PushbackOutcome = 'reverified' | 'falsified' | 'unverifiable';
export type PushbackDetector = 'regex' | 'llm';

export interface PushbackEventDTO {
  id: number;
  task_id: string;
  contradicted_claim_id: string;
  user_evidence: string;
  confidence: number;
  detector: PushbackDetector;
  outcome: PushbackOutcome;
  new_evidence_excerpt: string;
  invalidated_claim_ids: string[];
  duration_seconds: number;
  detected_at: string;
}

export interface PushbackStatsReport {
  window_minutes: number;
  total: number;
  reverified: number;
  falsified: number;
  unverifiable: number;
  mean_seconds_to_recheck: number;
}

export interface PushbackRecentReport {
  window_minutes: number;
  count: number;
  events: PushbackEventDTO[];
}

// ── /epistemic/incidents ───────────────────────────────────────────

export interface TimelineEntryDTO {
  at: string;
  kind: string;             // "claim_emit" | "claim_supersede" | "bias_match" | "pushback"
  summary: string;
  claim_id: string | null;
  bias_id: string | null;
  severity: Severity | null;
}

export interface BehavioralChangeDTO {
  kind: string;             // "verifier_registry_addition" | "feedback_memory_entry" | "ledger_pattern_warning"
  target: string;
  body: string;
  proposed_by: string;
}

export interface IncidentSummaryDTO {
  incident_id: string;
  task_id: string;
  root_cause_bias_id: string;
  severity: Severity;
  self_improver_emitted: boolean;
  created_at: string;
}

export interface IncidentDetailDTO {
  incident_id: string;
  task_id: string;
  timeline: TimelineEntryDTO[];
  root_cause: BiasMatchDTO;
  enabling_factors: BiasMatchDTO[];
  missed_signals: string[];
  behavioral_changes: BehavioralChangeDTO[];
  cost: Record<string, unknown>;
  severity: Severity;
  created_at: string;
  self_improver_emitted: boolean;
}

export interface IncidentListReport {
  count: number;
  incidents: IncidentSummaryDTO[];
}

// ── /epistemic/peer-reviews ────────────────────────────────────────

export type PeerReviewDecision = 'allow' | 'revise' | 'veto';

export interface PeerReviewDTO {
  id: number;
  task_id: string;
  triggering_claim_id: string | null;
  proposal_excerpt: string;
  decision: PeerReviewDecision;
  rationale: string;
  suggested_revision: string | null;
  reviewers: string[];
  duration_seconds: number;
  requested_at: string;
}

export interface PeerReviewStatsReport {
  window_minutes: number;
  total: number;
  allow: number;
  revise: number;
  veto: number;
  mean_seconds: number;
}

export interface PeerReviewsRecentReport {
  window_minutes: number;
  count: number;
  reviews: PeerReviewDTO[];
}

// ── /epistemic/overrides ───────────────────────────────────────────

export type OverrideUserAction =
  | 'force_proceed'
  | 'use_revision'
  | 'abandon';

export type OverrideBlockedAction = 'block' | 'revise';

export interface OverrideDTO {
  override_id: string;
  task_id: string;
  peer_review_id: number | null;
  blocked_action: OverrideBlockedAction;
  user_action: OverrideUserAction;
  user_reasoning: string;
  overridden_at: string;
}

export interface OverrideStatsReport {
  window_minutes: number;
  total: number;
  force_proceed: number;
  use_revision: number;
  abandon: number;
}

export interface OverridesRecentReport {
  window_minutes: number;
  count: number;
  overrides: OverrideDTO[];
}
