// Architecture-request types — match app/architecture_requests/models.py +
// app/control_plane/architecture_requests_api.py serializer (which adds
// the three is_* / package_is_protected helper booleans).

export type ArchStatus =
  | 'proposed'
  | 'approved'
  | 'scaffolded'
  | 'implementing'
  | 'completed'
  | 'rejected'
  | 'tier_immutable_refused'
  | 'timeout'
  | 'abandoned';

export type ArchDecisionSource =
  | 'signal-thumbs-up'
  | 'signal-thumbs-down'
  | 'react-approve'
  | 'react-reject'
  | 'timeout';

export interface FileSpec {
  path: string;
  purpose: string;
  initial_stub: string;
}

export interface IntegrationPoint {
  kind: string;
  target_module: string;
  detail: Record<string, unknown>;
}

export interface ArchitectureRequest {
  id: string;
  created_at: string;
  requestor: string;
  intent: string;
  motivation: string;
  package_path: string;
  file_layout: FileSpec[];
  integration_points: IntegrationPoint[];
  env_switches: Record<string, string>;
  test_plan: string;
  status: ArchStatus;
  decided_at?: string | null;
  decided_by?: ArchDecisionSource | null;
  decision_reason?: string | null;
  scaffolded_at?: string | null;
  scaffold_dir?: string | null;
  child_change_request_ids: string[];
  completed_at?: string | null;
  abandoned_at?: string | null;
  abandon_reason?: string | null;
  signal_message_ts?: number | null;
  // server-derived helpers
  is_terminal: boolean;
  is_decided: boolean;
  package_is_protected: boolean;
}

export interface ArchListResponse {
  count: number;
  architecture_requests: ArchitectureRequest[];
}

export interface ArchActionResponse {
  ok: boolean;
  architecture_request: ArchitectureRequest;
  scaffold_dir?: string;
  note?: string;
}

export interface ArchAuditEntry {
  event: string;
  request_id: string;
  status: string;
  package_path: string;
  requestor: string;
  decided_by?: string | null;
  ts: string;
}

export interface ArchAuditResponse {
  count: number;
  entries: ArchAuditEntry[];
}

export interface ArchManifestResponse {
  request_id: string;
  manifest_path: string;
  manifest_text: string;
}
