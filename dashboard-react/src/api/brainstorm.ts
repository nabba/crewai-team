// Brainstorm API hooks — wraps /api/cp/brainstorm/* endpoints.
//
// Mirrors the patterns established in src/api/companion.ts: query-key
// factory, useQuery for reads, useMutation for writes with onSuccess
// invalidations.

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api } from './client';
import type {
  BrainstormSessionPayload,
  SessionResponse,
  TechniqueInfo,
} from '../types/brainstorm';

const BASE = '/api/cp/brainstorm';

// ── Query keys ────────────────────────────────────────────────────────────

const keys = {
  techniques: ['brainstorm', 'techniques'] as const,
  sessions: (sender?: string) =>
    ['brainstorm', 'sessions', sender ?? 'default'] as const,
  active: (sender?: string) =>
    ['brainstorm', 'active', sender ?? 'default'] as const,
  session: (id: string) => ['brainstorm', 'session', id] as const,
};

// Polling intervals — multi-agent sessions can take many seconds, so we
// keep a slow but consistent refresh on the active-session view.
const POLL_NORMAL = 15_000;

// ── Reads ─────────────────────────────────────────────────────────────────

export function useTechniquesQuery() {
  return useQuery({
    queryKey: keys.techniques,
    queryFn: () => api<TechniqueInfo[]>(`${BASE}/techniques`),
    staleTime: Infinity, // Catalog is effectively static.
  });
}

export function useSessionsQuery(sender?: string) {
  const qs = sender ? `?sender=${encodeURIComponent(sender)}` : '';
  return useQuery({
    queryKey: keys.sessions(sender),
    queryFn: () => api<BrainstormSessionPayload[]>(`${BASE}/sessions${qs}`),
    refetchInterval: POLL_NORMAL,
  });
}

export function useActiveSessionQuery(sender?: string) {
  const qs = sender ? `?sender=${encodeURIComponent(sender)}` : '';
  return useQuery({
    queryKey: keys.active(sender),
    queryFn: () =>
      api<{ session: BrainstormSessionPayload | null }>(
        `${BASE}/sessions/active${qs}`,
      ),
    refetchInterval: POLL_NORMAL,
  });
}

export function useSessionQuery(sessionId: string | null) {
  return useQuery({
    queryKey: keys.session(sessionId ?? ''),
    queryFn: () =>
      api<BrainstormSessionPayload>(
        `${BASE}/sessions/${encodeURIComponent(sessionId!)}`,
      ),
    enabled: !!sessionId,
    refetchInterval: POLL_NORMAL,
  });
}

// ── Writes ────────────────────────────────────────────────────────────────

function _invalidateAll(qc: ReturnType<typeof useQueryClient>) {
  qc.invalidateQueries({ queryKey: ['brainstorm'] });
}

export function useStartSession() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (vars: {
      technique: string;
      topic: string;
      withAgents: number;
      sender?: string;
    }) => {
      const qs = vars.sender ? `?sender=${encodeURIComponent(vars.sender)}` : '';
      return api<SessionResponse>(`${BASE}/sessions${qs}`, {
        method: 'POST',
        body: JSON.stringify({
          technique: vars.technique,
          topic: vars.topic,
          with_agents: vars.withAgents,
        }),
      });
    },
    onSuccess: () => _invalidateAll(qc),
  });
}

export function useRespondMutation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (vars: {
      sessionId: string;
      message: string;
      sender?: string;
    }) => {
      const qs = vars.sender ? `?sender=${encodeURIComponent(vars.sender)}` : '';
      return api<SessionResponse>(
        `${BASE}/sessions/${encodeURIComponent(vars.sessionId)}/respond${qs}`,
        {
          method: 'POST',
          body: JSON.stringify({ message: vars.message }),
        },
      );
    },
    onSuccess: () => _invalidateAll(qc),
  });
}

export function useSkipMutation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (vars: { sessionId: string; sender?: string }) => {
      const qs = vars.sender ? `?sender=${encodeURIComponent(vars.sender)}` : '';
      return api<SessionResponse>(
        `${BASE}/sessions/${encodeURIComponent(vars.sessionId)}/skip${qs}`,
        { method: 'POST' },
      );
    },
    onSuccess: () => _invalidateAll(qc),
  });
}

export function usePauseMutation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (vars: { sessionId: string; sender?: string }) => {
      const qs = vars.sender ? `?sender=${encodeURIComponent(vars.sender)}` : '';
      return api<BrainstormSessionPayload>(
        `${BASE}/sessions/${encodeURIComponent(vars.sessionId)}/pause${qs}`,
        { method: 'POST' },
      );
    },
    onSuccess: () => _invalidateAll(qc),
  });
}

export function useResumeMutation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (vars: { sessionId: string; sender?: string }) => {
      const qs = vars.sender ? `?sender=${encodeURIComponent(vars.sender)}` : '';
      return api<SessionResponse>(
        `${BASE}/sessions/${encodeURIComponent(vars.sessionId)}/resume${qs}`,
        { method: 'POST' },
      );
    },
    onSuccess: () => _invalidateAll(qc),
  });
}

export function useCancelMutation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (vars: { sessionId: string; sender?: string }) => {
      const qs = vars.sender ? `?sender=${encodeURIComponent(vars.sender)}` : '';
      return api<BrainstormSessionPayload>(
        `${BASE}/sessions/${encodeURIComponent(vars.sessionId)}/cancel${qs}`,
        { method: 'POST' },
      );
    },
    onSuccess: () => _invalidateAll(qc),
  });
}

export function useFinishMutation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (vars: {
      sessionId: string;
      sender?: string;
      generateReport?: boolean;
    }) => {
      const params = new URLSearchParams();
      if (vars.sender) params.set('sender', vars.sender);
      if (vars.generateReport === false) params.set('generate_report', 'false');
      const qs = params.toString() ? `?${params.toString()}` : '';
      return api<BrainstormSessionPayload>(
        `${BASE}/sessions/${encodeURIComponent(vars.sessionId)}/finish${qs}`,
        { method: 'POST' },
      );
    },
    onSuccess: () => _invalidateAll(qc),
  });
}

export function useDeleteMutation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (vars: { sessionId: string }) => {
      return api<{ deleted: string }>(
        `${BASE}/sessions/${encodeURIComponent(vars.sessionId)}`,
        { method: 'DELETE' },
      );
    },
    onSuccess: () => _invalidateAll(qc),
  });
}
