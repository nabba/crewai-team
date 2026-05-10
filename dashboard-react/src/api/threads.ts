// Threads API hooks.

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api } from './client';
import type {
  Thread,
  ThreadListResponse,
  ThreadResponse,
  ThreadStatus,
} from '../types/threads';

const T = '/api/cp/threads';

export const threadsEndpoints = {
  list: (open_only = true, limit = 100) =>
    `${T}?open_only=${open_only}&limit=${limit}`,
  detail: (id: string) => `${T}/${encodeURIComponent(id)}`,
  create: () => T,
  addSubQuestion: (id: string) => `${T}/${encodeURIComponent(id)}/sub-question`,
  resolveSq: (id: string) => `${T}/${encodeURIComponent(id)}/resolve-sq`,
  blocker: (id: string) => `${T}/${encodeURIComponent(id)}/blocker`,
  clearBlockers: (id: string) => `${T}/${encodeURIComponent(id)}/clear-blockers`,
  note: (id: string) => `${T}/${encodeURIComponent(id)}/note`,
  transition: (id: string) => `${T}/${encodeURIComponent(id)}/transition`,
};

export const threadsKeys = {
  list: (open: boolean) => ['threads', 'list', open] as const,
  detail: (id: string) => ['threads', 'detail', id] as const,
};

export function useThreadsListQuery(open_only = true) {
  return useQuery({
    queryKey: threadsKeys.list(open_only),
    queryFn: () =>
      api<ThreadListResponse>(threadsEndpoints.list(open_only)),
    refetchInterval: 10_000,
  });
}

export function useThreadDetailQuery(id: string | undefined) {
  return useQuery({
    queryKey: threadsKeys.detail(id ?? ''),
    queryFn: () => api<Thread>(threadsEndpoints.detail(id as string)),
    enabled: Boolean(id),
    refetchInterval: 6_000,
  });
}

function invalidate(qc: ReturnType<typeof useQueryClient>, id?: string) {
  qc.invalidateQueries({ queryKey: ['threads', 'list'] });
  if (id) qc.invalidateQueries({ queryKey: threadsKeys.detail(id) });
}

export function useCreateThreadMutation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ title, description }: { title: string; description?: string }) =>
      api<ThreadResponse>(threadsEndpoints.create(), {
        method: 'POST',
        body: JSON.stringify({ title, description: description ?? '' }),
      }),
    onSuccess: () => invalidate(qc),
  });
}

export function useAddSubQuestionMutation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, text }: { id: string; text: string }) =>
      api<ThreadResponse>(threadsEndpoints.addSubQuestion(id), {
        method: 'POST',
        body: JSON.stringify({ text }),
      }),
    onSuccess: (_, { id }) => invalidate(qc, id),
  });
}

export function useResolveSqMutation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({
      id, subquestion_id, resolution,
    }: { id: string; subquestion_id: string; resolution?: string }) =>
      api<ThreadResponse>(threadsEndpoints.resolveSq(id), {
        method: 'POST',
        body: JSON.stringify({ subquestion_id, resolution: resolution ?? '' }),
      }),
    onSuccess: (_, { id }) => invalidate(qc, id),
  });
}

export function useAddBlockerMutation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, text }: { id: string; text: string }) =>
      api<ThreadResponse>(threadsEndpoints.blocker(id), {
        method: 'POST',
        body: JSON.stringify({ text }),
      }),
    onSuccess: (_, { id }) => invalidate(qc, id),
  });
}

export function useClearBlockersMutation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id }: { id: string }) =>
      api<ThreadResponse>(threadsEndpoints.clearBlockers(id), {
        method: 'POST',
      }),
    onSuccess: (_, { id }) => invalidate(qc, id),
  });
}

export function useAddNoteMutation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, text }: { id: string; text: string }) =>
      api<ThreadResponse>(threadsEndpoints.note(id), {
        method: 'POST',
        body: JSON.stringify({ text }),
      }),
    onSuccess: (_, { id }) => invalidate(qc, id),
  });
}

export function useTransitionThreadMutation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({
      id, transition, blocker, summary, reason,
    }: {
      id: string;
      transition: ThreadStatus;
      blocker?: string;
      summary?: string;
      reason?: string;
    }) =>
      api<ThreadResponse>(threadsEndpoints.transition(id), {
        method: 'POST',
        body: JSON.stringify({
          transition,
          blocker: blocker ?? null,
          summary: summary ?? null,
          reason: reason ?? null,
        }),
      }),
    onSuccess: (_, { id }) => invalidate(qc, id),
  });
}
