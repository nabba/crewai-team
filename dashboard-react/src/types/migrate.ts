// Cloud migration types — match the Python dataclass outputs from
// app/substrate/cloud_doctor.py, cloud_cost.py, migration_runner.py
// served via app/control_plane/migrate_api.py.

export type CloudTarget = 'gcp' | 'aws';
export type Tier = 'cheapest' | 'prod';

export type AccountType = 'user' | 'service_account';

export interface CloudAccount {
  account: string;
  type: AccountType;
  active: 'yes' | 'no';
}

export interface AccountsResponse {
  accounts: CloudAccount[];
}

export type ProbeStatus = 'OK' | 'MISSING' | 'STALE' | 'UNKNOWN';
export type OverallStatus = 'OK' | 'MISSING' | 'STALE' | 'DEGRADED' | 'UNKNOWN';

export interface PreflightProbe {
  name: string;
  status: ProbeStatus;
  detail: string;
  required: boolean;
}

export interface PreflightResponse {
  target: string;
  timestamp: string;
  overall: OverallStatus;
  probes: PreflightProbe[];
}

export interface CostLineItem {
  category: string;   // control_plane | compute | storage | network | monitoring | secrets
  resource: string;
  monthly_usd: number;
  note: string;
}

export interface CostResponse {
  target: string;
  tier: string;
  region: string;
  enable_monitoring: boolean;
  has_domain: boolean;
  line_items: CostLineItem[];
  by_category: Record<string, number>;
  total_monthly_usd: number;
  total_annual_usd: number;
}

export interface CostRequest {
  target: CloudTarget;
  tier: Tier;
  region?: string;
  enable_monitoring: boolean;
  has_domain: boolean;
}

export interface StartRequest {
  target: CloudTarget;
  tier: Tier;
  region?: string;
  project_id: string;
  active_account: string;
  confirm_phrase: string;
  budget_cap_usd: number;
}

export interface StartResponse {
  run_id: string;
  status: RunStatus;
  execute_subprocess: boolean;
  dry_shell_mode: boolean;
}

export type RunStatus =
  | 'queued'
  | 'preparing'
  | 'preflight_failed'
  | 'running'
  | 'succeeded'
  | 'failed'
  | 'cancelled';

export interface RunRecord {
  run_id: string;
  status: RunStatus;
  target: string;
  tier: string;
  region: string;
  project_id: string;
  active_account: string;
  started_at: string;
  updated_at: string;
  completed_at: string;
  current_step: string;
  progress_pct: number;
  detail: string;
  report_path: string;
  error: string;
}

export interface RunsResponse {
  active_run_id: string | null;
  runs: RunRecord[];
}

export interface CancelResponse {
  cancel_requested: boolean;
  active_run_id: string | null;
}
