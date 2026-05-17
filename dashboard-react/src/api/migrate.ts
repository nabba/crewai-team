// Cloud-migration API hooks — mirrors changes.ts / coding_sessions.ts.
// Backend is exhaustive (7 endpoints — see app/control_plane/migrate_api.py).
// Polling cadence is deliberately tight on the in-flight run (2 s) and
// moderate on the runs list (10 s); everything else is on-demand only.

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api } from './client';
import type {
  AccountsResponse,
  CancelResponse,
  CloudTarget,
  CostRequest,
  CostResponse,
  PreflightResponse,
  RunRecord,
  RunsResponse,
  StartRequest,
  StartResponse,
} from '../types/migrate';

const M = '/api/cp/migrate';

export const migrateEndpoints = {
  accounts: () => `${M}/accounts`,
  preflight: (target: CloudTarget = 'gcp') =>
    `${M}/preflight?target=${encodeURIComponent(target)}`,
  cost: () => `${M}/cost`,
  start: () => `${M}/start`,
  runs: (limit = 20) => `${M}/runs?limit=${limit}`,
  run: (runId: string) => `${M}/runs/${encodeURIComponent(runId)}`,
  cancel: (runId: string) =>
    `${M}/runs/${encodeURIComponent(runId)}/cancel`,
  hardeningPreview: (profile: string, binauthzMode: string) =>
    `${M}/hardening-preview?profile=${encodeURIComponent(profile)}&binauthz_mode=${encodeURIComponent(binauthzMode)}`,
  bootstrapProject: () => `${M}/bootstrap-project`,
};

export const migrateKeys = {
  accounts: ['migrate', 'accounts'] as const,
  preflight: (target: CloudTarget) => ['migrate', 'preflight', target] as const,
  runs: (limit: number) => ['migrate', 'runs', limit] as const,
  run: (runId: string) => ['migrate', 'run', runId] as const,
  hardeningPreview: (profile: string, binauthzMode: string) =>
    ['migrate', 'hardening-preview', profile, binauthzMode] as const,
};

// Helper used by mutations to nudge the runs list + a specific run.
function invalidateMigrate(qc: ReturnType<typeof useQueryClient>) {
  qc.invalidateQueries({ queryKey: ['migrate', 'runs'] });
  qc.invalidateQueries({ queryKey: ['migrate', 'run'] });
}

// Terminal statuses don't need polling — return false from refetchInterval
// to stop the timer once the run is done.
const TERMINAL_STATUSES = new Set([
  'succeeded',
  'failed',
  'preflight_failed',
  'cancelled',
]);

export function useMigrateAccountsQuery() {
  return useQuery({
    queryKey: migrateKeys.accounts,
    queryFn: () => api<AccountsResponse>(migrateEndpoints.accounts()),
    // Accounts don't change without operator action in another tab —
    // refetch on window focus is enough.
    staleTime: 60_000,
  });
}

export function useMigratePreflightQuery(target: CloudTarget = 'gcp') {
  return useQuery({
    queryKey: migrateKeys.preflight(target),
    queryFn: () =>
      api<PreflightResponse>(migrateEndpoints.preflight(target)),
    // Operator can fix a probe (e.g. install gcloud) then click Recheck —
    // we don't auto-poll because each call shells out to gcloud/aws.
    staleTime: 30_000,
    refetchOnWindowFocus: false,
  });
}

export function useMigrateCostMutation() {
  // Cost is a POST that takes a body — modeled as a mutation so React
  // can call it imperatively on form change without flooding the
  // gateway with stale-query re-renders.
  return useMutation({
    mutationFn: (body: CostRequest) =>
      api<CostResponse>(migrateEndpoints.cost(), {
        method: 'POST',
        body: JSON.stringify(body),
      }),
  });
}

export function useMigrateStartMutation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: StartRequest) =>
      api<StartResponse>(migrateEndpoints.start(), {
        method: 'POST',
        body: JSON.stringify(body),
      }),
    onSuccess: () => invalidateMigrate(qc),
  });
}

export function useMigrateRunsQuery(limit = 10, interval = 10_000) {
  return useQuery({
    queryKey: migrateKeys.runs(limit),
    queryFn: () => api<RunsResponse>(migrateEndpoints.runs(limit)),
    refetchInterval: interval,
  });
}

// Polls one run every 2 s while it's active; backs off automatically on
// terminal statuses so the page is quiet after completion.
export function useMigrateRunQuery(runId: string | undefined) {
  return useQuery({
    queryKey: migrateKeys.run(runId ?? ''),
    queryFn: () => api<RunRecord>(migrateEndpoints.run(runId as string)),
    enabled: Boolean(runId),
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      if (status && TERMINAL_STATUSES.has(status)) return false;
      return 2_000;
    },
  });
}

export function useMigrateCancelMutation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ runId }: { runId: string }) =>
      api<CancelResponse>(migrateEndpoints.cancel(runId), {
        method: 'POST',
      }),
    onSuccess: () => invalidateMigrate(qc),
  });
}

// ── Hardening (2026-05-17) ────────────────────────────────────────

export interface AllowedCidr {
  cidr_block: string;
  display_name: string;
}

export interface HardeningPreview {
  profile: string;
  tailnet_reachable: boolean;
  tailnet_cidr: string | null;
  laptop_public_ip: string | null;
  recommended_cidrs: AllowedCidr[];
  org_id: string | null;
  binauthz_mode: string;
  notes: string[];
}

export function useHardeningPreviewQuery(
  profile: string = 'strict',
  binauthzMode: string = 'AUDIT',
) {
  return useQuery({
    queryKey: migrateKeys.hardeningPreview(profile, binauthzMode),
    queryFn: () =>
      api<HardeningPreview>(migrateEndpoints.hardeningPreview(profile, binauthzMode)),
    staleTime: 30_000,
    refetchOnWindowFocus: false,
  });
}

export interface BootstrapRequest {
  project_id: string;
  billing_account: string;
  org_id?: string | null;
  project_name?: string | null;
  confirm_phrase: string;
  dry_run?: boolean;
}

export interface BootstrapResponse {
  rc: number;
  stdout: string;
  stderr: string;
  dry_run: boolean;
  ok: boolean;
}

export function useBootstrapProjectMutation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: BootstrapRequest) =>
      api<BootstrapResponse>(migrateEndpoints.bootstrapProject(), {
        method: 'POST',
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['migrate', 'preflight'] });
    },
  });
}
