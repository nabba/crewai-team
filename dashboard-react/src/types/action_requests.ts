// Action-request types — mirror app/action_requests/models.py

export type ActionStatus =
  | 'pending'
  | 'approved'
  | 'applied'
  | 'apply_failed'
  | 'rejected'
  | 'invalid'
  | 'timeout';

export type ActionType = 'email_draft';
// Future: 'calendar_invite' | 'slack_message' | ...

export type ActionDecisionSource =
  | 'signal-thumbs-up'
  | 'signal-thumbs-down'
  | 'react-approve'
  | 'react-reject'
  | 'timeout';

export interface ActionRequest {
  id: string;
  created_at: string;
  requestor: string;
  action_type: ActionType;
  summary: string;
  data: Record<string, unknown>;
  reason: string;
  status: ActionStatus;
  decided_at?: string | null;
  decided_by?: ActionDecisionSource | null;
  decision_reason?: string | null;
  applied_at?: string | null;
  apply_error?: string | null;
  apply_artifact: Record<string, unknown>;
  invalid_reason?: string | null;
  signal_message_ts?: number | null;
  is_terminal: boolean;
  is_decided: boolean;
}

export interface ActionListResponse {
  count: number;
  action_requests: ActionRequest[];
}

export interface ActionTypesResponse {
  types: ActionType[];
}

export interface ActionResponse {
  ok: boolean;
  action_request: ActionRequest;
}
