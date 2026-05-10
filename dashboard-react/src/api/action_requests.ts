// Action-requests API hooks. Mirrors api/changes.ts pattern.

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api } from './client';
import type {
  ActionListResponse,
  ActionRequest,
  ActionResponse,
  ActionStatus,
  ActionType,
  ActionTypesResponse,
} from '../types/action_requests';

const A = '/api/cp/action-requests';

export const actionEndpoints = {
  list: (status?: ActionStatus, action_type?: ActionType, limit = 100) => {
    const params = new URLSearchParams();
    if (status) params.set('status', status);
    if (action_type) params.set('action_type', action_type);
    params.set('limit', String(limit));
    return `${A}?${params.toString()}`;
  },
  detail: (id: string) => `${A}/${encodeURIComponent(id)}`,
  approve: (id: string) => `${A}/${encodeURIComponent(id)}/approve`,
  reject: (id: string) => `${A}/${encodeURIComponent(id)}/reject`,
  retryApply: (id: string) => `${A}/${encodeURIComponent(id)}/retry-apply`,
  types: () => `${A}/types`,
};

export const actionKeys = {
  list: (status?: ActionStatus, t?: ActionType) =>
    ['action-requests', 'list', status ?? 'all', t ?? 'all'] as const,
  detail: (id: string) => ['action-requests', 'detail', id] as const,
  types: () => ['action-requests', 'types'] as const,
};

export function useActionListQuery(
  status?: ActionStatus, action_type?: ActionType,
) {
  return useQuery({
    queryKey: actionKeys.list(status, action_type),
    queryFn: () =>
      api<ActionListResponse>(actionEndpoints.list(status, action_type)),
    refetchInterval: 8_000,
  });
}

export function useActionDetailQuery(id: string | undefined) {
  return useQuery({
    queryKey: actionKeys.detail(id ?? ''),
    queryFn: () =>
      api<ActionRequest>(actionEndpoints.detail(id as string)),
    enabled: Boolean(id),
    refetchInterval: 5_000,
  });
}

export function useActionTypesQuery() {
  return useQuery({
    queryKey: actionKeys.types(),
    queryFn: () => api<ActionTypesResponse>(actionEndpoints.types()),
  });
}

function invalidate(qc: ReturnType<typeof useQueryClient>, id: string) {
  qc.invalidateQueries({ queryKey: ['action-requests', 'list'] });
  qc.invalidateQueries({ queryKey: actionKeys.detail(id) });
}

export function useActionApproveMutation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, reason }: { id: string; reason?: string }) =>
      api<ActionResponse>(actionEndpoints.approve(id), {
        method: 'POST',
        body: JSON.stringify({
          operator: 'react-operator',
          reason: reason ?? null,
        }),
      }),
    onSuccess: (_, { id }) => invalidate(qc, id),
  });
}

export function useActionRejectMutation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, reason }: { id: string; reason?: string }) =>
      api<ActionResponse>(actionEndpoints.reject(id), {
        method: 'POST',
        body: JSON.stringify({
          operator: 'react-operator',
          reason: reason ?? null,
        }),
      }),
    onSuccess: (_, { id }) => invalidate(qc, id),
  });
}

export function useActionRetryApplyMutation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id }: { id: string }) =>
      api<ActionResponse>(actionEndpoints.retryApply(id), {
        method: 'POST',
      }),
    onSuccess: (_, { id }) => invalidate(qc, id),
  });
}
