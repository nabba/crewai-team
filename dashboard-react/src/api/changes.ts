// Change-request API — kept separate from queries.ts/endpoints.ts so the
// diff stays localized. Mirrors the forge.ts pattern.

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api } from './client';
import type {
  ApproveResponse,
  ChangeListResponse,
  ChangeRequest,
  ChangeStatus,
  RejectResponse,
  RollbackResponse,
} from '../types/changes';

const C = '/api/cp/changes';

export const changesEndpoints = {
  list: (status?: ChangeStatus, limit = 100) =>
    status
      ? `${C}?status=${encodeURIComponent(status)}&limit=${limit}`
      : `${C}?limit=${limit}`,
  detail: (id: string) => `${C}/${encodeURIComponent(id)}`,
  approve: (id: string) => `${C}/${encodeURIComponent(id)}/approve`,
  reject: (id: string) => `${C}/${encodeURIComponent(id)}/reject`,
  rollback: (id: string) => `${C}/${encodeURIComponent(id)}/rollback`,
  retryApply: (id: string) => `${C}/${encodeURIComponent(id)}/retry-apply`,
};

export const changesKeys = {
  list: (status?: ChangeStatus) =>
    ['changes', 'list', status ?? 'all'] as const,
  detail: (id: string) => ['changes', 'detail', id] as const,
};

export function useChangesListQuery(status?: ChangeStatus) {
  return useQuery({
    queryKey: changesKeys.list(status),
    queryFn: () => api<ChangeListResponse>(changesEndpoints.list(status)),
    refetchInterval: 8_000,
  });
}

export function useChangeDetailQuery(id: string | undefined) {
  return useQuery({
    queryKey: changesKeys.detail(id ?? ''),
    queryFn: () => api<ChangeRequest>(changesEndpoints.detail(id as string)),
    enabled: Boolean(id),
    refetchInterval: 5_000,
  });
}

// All four mutations invalidate the list and the detail cache for that id —
// the operator surface should reflect server state immediately even though
// status transitions also fan out via the polling interval.

function invalidateChange(qc: ReturnType<typeof useQueryClient>, id: string) {
  qc.invalidateQueries({ queryKey: ['changes', 'list'] });
  qc.invalidateQueries({ queryKey: changesKeys.detail(id) });
}

export function useApproveChangeMutation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({
      id,
      operator,
      reason,
    }: {
      id: string;
      operator?: string;
      reason?: string;
    }) =>
      api<ApproveResponse>(changesEndpoints.approve(id), {
        method: 'POST',
        body: JSON.stringify({
          operator: operator ?? 'react-operator',
          reason: reason ?? null,
        }),
      }),
    onSuccess: (_, { id }) => invalidateChange(qc, id),
  });
}

export function useRejectChangeMutation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({
      id,
      operator,
      reason,
    }: {
      id: string;
      operator?: string;
      reason?: string;
    }) =>
      api<RejectResponse>(changesEndpoints.reject(id), {
        method: 'POST',
        body: JSON.stringify({
          operator: operator ?? 'react-operator',
          reason: reason ?? null,
        }),
      }),
    onSuccess: (_, { id }) => invalidateChange(qc, id),
  });
}

export function useRollbackChangeMutation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, operator }: { id: string; operator?: string }) =>
      api<RollbackResponse>(changesEndpoints.rollback(id), {
        method: 'POST',
        body: JSON.stringify({ operator: operator ?? 'react-operator' }),
      }),
    onSuccess: (_, { id }) => invalidateChange(qc, id),
  });
}

export function useRetryApplyMutation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id }: { id: string }) =>
      api<ApproveResponse>(changesEndpoints.retryApply(id), {
        method: 'POST',
      }),
    onSuccess: (_, { id }) => invalidateChange(qc, id),
  });
}
