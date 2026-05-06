// Change-request types — match the Python pydantic / dataclass output
// from app/change_requests/models.py + app/control_plane/changes_api.py
// (the API serializer adds the three is_* booleans).

export type ChangeStatus =
  | 'pending'
  | 'approved'
  | 'rejected'
  | 'applied'
  | 'apply_failed'
  | 'rolled_back'
  | 'tier_immutable_refused'
  | 'timeout';

export type DecisionSource =
  | 'signal-thumbs-up'
  | 'signal-thumbs-down'
  | 'react-approve'
  | 'react-reject'
  | 'timeout';

export interface ChangeRequest {
  // identity / provenance
  id: string;
  created_at: string;          // ISO-8601 UTC
  requestor: string;           // agent_id
  // proposed change
  path: string;
  new_content: string;
  old_content: string;
  reason: string;
  diff: string;
  // lifecycle
  status: ChangeStatus;
  decided_at?: string | null;
  decided_by?: DecisionSource | null;
  decision_reason?: string | null;
  // application
  git_branch?: string | null;
  git_commit_sha?: string | null;
  pr_url?: string | null;
  applied_at?: string | null;
  apply_error?: string | null;
  // rollback
  rollback_commit_sha?: string | null;
  rolled_back_at?: string | null;
  rolled_back_by?: string | null;
  rollback_pr_url?: string | null;
  // signal correlation
  signal_message_ts?: number | null;
  // server-derived (changes_api.py _serialize)
  is_terminal: boolean;
  is_rollbackable: boolean;
  is_protected: boolean;
}

export interface ChangeListResponse {
  count: number;
  changes: ChangeRequest[];
}

// POST /approve and /retry-apply share this shape
export interface ApplyResultPayload {
  ok: boolean;
  git_branch: string | null;
  git_commit_sha: string | null;
  pr_url: string | null;
  module_reload_ok?: boolean | null;
  module_reload_note?: string | null;
  error: string | null;
}

export interface ApproveResponse {
  ok: boolean;
  change: ChangeRequest;
  apply_result: ApplyResultPayload;
}

export interface RejectResponse {
  ok: boolean;
  change: ChangeRequest;
}

export interface RollbackResultPayload {
  ok: boolean;
  revert_branch: string | null;
  revert_commit_sha: string | null;
  revert_pr_url: string | null;
  module_reload_ok?: boolean | null;
  module_reload_note?: string | null;
  error: string | null;
}

export interface RollbackResponse {
  ok: boolean;
  change: ChangeRequest;
  rollback_result: RollbackResultPayload;
}
