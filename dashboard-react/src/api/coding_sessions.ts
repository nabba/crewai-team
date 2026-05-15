// Coding-session API — read-only by design. No mutations are
// exposed; lifecycle is owned by the agent + reconciler. The
// operator's actionable surface is the change-request UI (#55),
// not this view.

import { useQuery } from '@tanstack/react-query';
import { api } from './client';
import type {
  CodingSession,
  CodingSessionListResponse,
  CodingSessionStatus,
} from '../types/coding_sessions';

const C = '/api/cp/coding-sessions';

export const codingSessionsEndpoints = {
  list: (status?: CodingSessionStatus, agentId?: string, limit = 200) => {
    const params = new URLSearchParams();
    if (status) params.set('status', status);
    if (agentId) params.set('agent_id', agentId);
    params.set('limit', String(limit));
    return `${C}?${params.toString()}`;
  },
  detail: (id: string) => `${C}/${encodeURIComponent(id)}`,
  evolutionRuns: (id: string, limit = 50) =>
    `${C}/${encodeURIComponent(id)}/evolution_runs?limit=${limit}`,
};

export const codingSessionsKeys = {
  list: (status?: CodingSessionStatus, agentId?: string) =>
    ['coding-sessions', 'list', status ?? 'all', agentId ?? 'all'] as const,
  detail: (id: string) => ['coding-sessions', 'detail', id] as const,
  evolutionRuns: (id: string) =>
    ['coding-sessions', 'evolution-runs', id] as const,
};

// Q7.4 — per-coding-session ShinkaEvolve audit response shape.
export type EvolutionRun = {
  ts: string;
  session_id: string;
  agent_id: string;
  initial_path: string;
  evaluate_path: string;
  num_generations: number;
  num_islands: number;
  max_cost_usd: number;
  status:
    | 'improved'
    | 'no_improvement'
    | 'refused'
    | 'disabled'
    | 'shinka_unavailable'
    | 'error';
  baseline_score: number;
  best_score: number;
  delta: number;
  generations_run: number;
  variants_evaluated: number;
  duration_seconds: number;
  diff_sha256: string;
  diff_length: number;
  error: string;
  refusal_reason: string;
};

export type EvolutionRunsResponse = {
  session_id: string;
  summary: {
    n_runs: number;
    by_status: Record<string, number>;
    best_delta?: number;
    total_max_cost_usd?: number;
    total_duration_seconds?: number;
    last_run_at?: string | null;
  };
  runs: EvolutionRun[];
};

export function useCodingSessionsListQuery(
  status?: CodingSessionStatus,
  agentId?: string,
) {
  return useQuery({
    queryKey: codingSessionsKeys.list(status, agentId),
    queryFn: () =>
      api<CodingSessionListResponse>(
        codingSessionsEndpoints.list(status, agentId),
      ),
    refetchInterval: 8_000,
  });
}

export function useCodingSessionDetailQuery(id: string | undefined) {
  return useQuery({
    queryKey: codingSessionsKeys.detail(id ?? ''),
    queryFn: () =>
      api<CodingSession>(codingSessionsEndpoints.detail(id as string)),
    enabled: Boolean(id),
    refetchInterval: 5_000,
  });
}

// Q7.4 — fetch the per-session evolution-runs audit. Enabled only
// when an id is known (the drawer is open).
export function useCodingSessionEvolutionRunsQuery(id: string | undefined) {
  return useQuery({
    queryKey: codingSessionsKeys.evolutionRuns(id ?? ''),
    queryFn: () =>
      api<EvolutionRunsResponse>(
        codingSessionsEndpoints.evolutionRuns(id as string),
      ),
    enabled: Boolean(id),
    refetchInterval: 10_000,
  });
}
