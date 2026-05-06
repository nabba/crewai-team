// Coding-session types — match the Python dataclass output from
// app/coding_session/models.py + the API serializer in
// app/control_plane/coding_sessions_api.py (which adds the two
// is_* booleans).

export type CodingSessionStatus =
  | 'active'
  | 'submitted'
  | 'discarded'
  | 'expired'
  | 'failed';

export interface SubmitResult {
  path: string;
  change_request_id: string | null;
  status: string;
  refusal_reason?: string | null;
}

export interface CodingSession {
  // identity
  id: string;
  agent_id: string;
  purpose: string;
  created_at: string;
  // base + worktree
  base: string;
  base_sha: string;
  worktree_path: string;
  // lifecycle
  expires_at: string;
  last_activity_at: string;
  status: CodingSessionStatus;
  // tracking
  files_touched: string[];
  run_count: number;
  bytes_written: number;
  // terminal metadata
  terminated_at?: string | null;
  terminated_reason?: string | null;
  submit_results?: SubmitResult[] | null;
  // server-derived (coding_sessions_api.py _serialize)
  is_active: boolean;
  is_terminal: boolean;
}

export interface CodingSessionListResponse {
  count: number;
  sessions: CodingSession[];
}
