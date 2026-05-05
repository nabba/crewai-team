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
};

export const codingSessionsKeys = {
  list: (status?: CodingSessionStatus, agentId?: string) =>
    ['coding-sessions', 'list', status ?? 'all', agentId ?? 'all'] as const,
  detail: (id: string) => ['coding-sessions', 'detail', id] as const,
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
