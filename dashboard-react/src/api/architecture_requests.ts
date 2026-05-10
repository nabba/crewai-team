// Architecture-requests API hooks. Mirrors api/changes.ts.

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api } from './client';
import type {
  ArchActionResponse,
  ArchAuditResponse,
  ArchListResponse,
  ArchManifestResponse,
  ArchStatus,
  ArchitectureRequest,
} from '../types/architecture_requests';

const A = '/api/cp/architecture-requests';

export const archEndpoints = {
  list: (status?: ArchStatus, limit = 100) =>
    status
      ? `${A}?status=${encodeURIComponent(status)}&limit=${limit}`
      : `${A}?limit=${limit}`,
  detail: (id: string) => `${A}/${encodeURIComponent(id)}`,
  audit: (id: string) => `${A}/${encodeURIComponent(id)}/audit`,
  manifest: (id: string) => `${A}/${encodeURIComponent(id)}/scaffold/manifest`,
  approve: (id: string) => `${A}/${encodeURIComponent(id)}/approve`,
  reject: (id: string) => `${A}/${encodeURIComponent(id)}/reject`,
  scaffold: (id: string) => `${A}/${encodeURIComponent(id)}/scaffold`,
  abandon: (id: string) => `${A}/${encodeURIComponent(id)}/abandon`,
  recordChild: (id: string) => `${A}/${encodeURIComponent(id)}/record-child-cr`,
  markComplete: (id: string) => `${A}/${encodeURIComponent(id)}/mark-complete`,
};

export const archKeys = {
  list: (status?: ArchStatus) => ['arch-requests', 'list', status ?? 'all'] as const,
  detail: (id: string) => ['arch-requests', 'detail', id] as const,
  audit: (id: string) => ['arch-requests', 'audit', id] as const,
  manifest: (id: string) => ['arch-requests', 'manifest', id] as const,
};

export function useArchListQuery(status?: ArchStatus) {
  return useQuery({
    queryKey: archKeys.list(status),
    queryFn: () => api<ArchListResponse>(archEndpoints.list(status)),
    refetchInterval: 10_000,
  });
}

export function useArchDetailQuery(id: string | undefined) {
  return useQuery({
    queryKey: archKeys.detail(id ?? ''),
    queryFn: () => api<ArchitectureRequest>(archEndpoints.detail(id as string)),
    enabled: Boolean(id),
    refetchInterval: 6_000,
  });
}

export function useArchAuditQuery(id: string | undefined) {
  return useQuery({
    queryKey: archKeys.audit(id ?? ''),
    queryFn: () => api<ArchAuditResponse>(archEndpoints.audit(id as string)),
    enabled: Boolean(id),
    refetchInterval: 10_000,
  });
}

export function useArchManifestQuery(id: string | undefined, enabled: boolean) {
  return useQuery({
    queryKey: archKeys.manifest(id ?? ''),
    queryFn: () =>
      api<ArchManifestResponse>(archEndpoints.manifest(id as string)),
    enabled: Boolean(id) && enabled,
  });
}

function invalidate(qc: ReturnType<typeof useQueryClient>, id: string) {
  qc.invalidateQueries({ queryKey: ['arch-requests', 'list'] });
  qc.invalidateQueries({ queryKey: archKeys.detail(id) });
  qc.invalidateQueries({ queryKey: archKeys.audit(id) });
  qc.invalidateQueries({ queryKey: archKeys.manifest(id) });
}

export function useArchApproveMutation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({
      id,
      reason,
    }: {
      id: string;
      reason?: string;
    }) =>
      api<ArchActionResponse>(archEndpoints.approve(id), {
        method: 'POST',
        body: JSON.stringify({
          operator: 'react-operator',
          reason: reason ?? null,
        }),
      }),
    onSuccess: (_, { id }) => invalidate(qc, id),
  });
}

export function useArchRejectMutation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({
      id,
      reason,
    }: {
      id: string;
      reason?: string;
    }) =>
      api<ArchActionResponse>(archEndpoints.reject(id), {
        method: 'POST',
        body: JSON.stringify({
          operator: 'react-operator',
          reason: reason ?? null,
        }),
      }),
    onSuccess: (_, { id }) => invalidate(qc, id),
  });
}

export function useArchScaffoldMutation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id }: { id: string }) =>
      api<ArchActionResponse>(archEndpoints.scaffold(id), {
        method: 'POST',
      }),
    onSuccess: (_, { id }) => invalidate(qc, id),
  });
}

export function useArchAbandonMutation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, reason }: { id: string; reason: string }) =>
      api<ArchActionResponse>(archEndpoints.abandon(id), {
        method: 'POST',
        body: JSON.stringify({
          operator: 'react-operator',
          reason,
        }),
      }),
    onSuccess: (_, { id }) => invalidate(qc, id),
  });
}

export function useArchMarkCompleteMutation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id }: { id: string }) =>
      api<ArchActionResponse>(archEndpoints.markComplete(id), {
        method: 'POST',
      }),
    onSuccess: (_, { id }) => invalidate(qc, id),
  });
}
