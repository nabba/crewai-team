export interface Project {
  id: string;
  name: string;
  description?: string;
  mission?: string;
  is_active: boolean;
  config_json?: Record<string, unknown>;
  created_at: string;
  updated_at?: string;
}

export interface Ticket {
  id: string;
  project_id: string;
  title: string;
  description?: string;
  status: 'todo' | 'in_progress' | 'review' | 'done' | 'failed' | 'blocked';
  priority: number;
  assigned_agent?: string;
  assigned_crew?: string;
  source?: string;
  difficulty?: number;
  cost_usd: number;
  tokens_used: number;
  result_summary?: string;
  started_at?: string;
  completed_at?: string;
  created_at: string;
  updated_at?: string;
  comments?: TicketComment[];
}

export interface TicketComment {
  id: string;
  ticket_id: string;
  author: string;
  content: string;
  metadata_json?: Record<string, unknown>;
  created_at: string;
}

export interface Budget {
  agent_role: string;
  period: string;
  limit_usd: number;
  spent_usd: number;
  limit_tokens?: number;
  spent_tokens?: number;
  is_paused: boolean;
  pct_used?: number;
  warning_pct?: number;
  project_name?: string;
}

export interface AuditEntry {
  id: number;
  timestamp: string;
  project_id?: string;
  actor: string;
  action: string;
  resource_type?: string;
  resource_id?: string;
  detail_json?: Record<string, unknown>;
  cost_usd?: number;
  tokens?: number;
}

export interface GovernanceRequest {
  id: string;
  project_id?: string;
  request_type: string;
  requested_by: string;
  title: string;
  detail_json?: Record<string, unknown>;
  status: 'pending' | 'approved' | 'rejected' | 'expired';
  reviewed_by?: string;
  reviewed_at?: string;
  expires_at?: string;
  created_at: string;
}

export interface OrgChartAgent {
  agent_role: string;
  display_name: string;
  reports_to?: string;
  job_description?: string;
  soul_file?: string;
  default_model?: string;
  sort_order: number;
}

export interface HealthStatus {
  status: string;
  tickets_total: number;
  audit_entries: number;
  governance_pending: number;
}

export interface CostEntry {
  day: string;
  total_cost: number;
  total_tokens: number;
  call_count: number;
}

export interface AgentCost {
  actor: string;
  calls: number;
  total_cost: number;
  total_tokens: number;
}

export interface KanbanBoard {
  board: {
    todo: Ticket[];
    in_progress: Ticket[];
    review: Ticket[];
    done: Ticket[];
    failed: Ticket[];
    blocked: Ticket[];
  };
  counts: Record<string, number>;
  total: number;
}

// ── Consciousness Workspaces ──────────────────────────────────────────────

export interface WorkspaceItem {
  item_id: string;
  content: string;
  salience: number;
  source_agent: string;
  source_channel: string;
  cycles: number;
  consumed: boolean;
}

export interface WorkspaceSnapshot {
  project_id: string;
  cycle: number;
  capacity: number;
  active_count: number;
  peripheral_count: number;
  active_items: WorkspaceItem[];
}

export interface WorkspaceList {
  workspaces: WorkspaceSnapshot[];
  count: number;
}

export interface WorkspaceItems {
  project_id: string;
  active: WorkspaceItem[];
  peripheral: WorkspaceItem[];
  capacity: number;
  cycle: number;
}

export interface MetaWorkspace {
  meta_workspace: Record<string, unknown>;
  by_project: Record<string, { content: string; salience: number }[]>;
  project_count: number;
}
