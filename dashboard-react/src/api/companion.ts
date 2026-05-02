// Companion Layer API — feedback, ideas, sources, config, promote,
// document, wiki, grand-task, xworkspace inbox.
//
// Backend endpoints under /api/cp/companion/* (registered via
// app.api.companion_api). Workspace IDs are CP project ids.

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api } from './client';

// ── Types ──────────────────────────────────────────────────────────────────

export interface CompanionConfig {
  enabled: boolean;
  seed_prompt: string | null;
  daily_budget_usd: number;
  surface_threshold: number;
  novelty_threshold: number;
  transferability_threshold: number;
  panel_threshold: number;
  quiet_hours_start: number;
  quiet_hours_end: number;
  sources: unknown[];
}

export type CompanionConfigPatch = Partial<{
  enabled: boolean;
  seed_prompt: string | null;
  daily_budget_usd: number;
  surface_threshold: number;
  novelty_threshold: number;
  transferability_threshold: number;
  panel_threshold: number;
  quiet_hours_start: number;
  quiet_hours_end: number;
}>;

export interface IdeaSummary {
  idea_id: string;
  cycle_id: string;
  text: string;
  role: string;
  created_state: string;
  current_state: string | null;
  lineage_parents: string[];
  novelty: number;
  quality: number;
  transferability: number;
  panel_score?: number;
  created_at: number;
}

export interface CompanionSource {
  source_id: string;
  type: string;
  config: Record<string, unknown>;
  enabled: boolean;
  added_at: number;
  last_ingested_at: number;
  last_ingest_status: string;
}

export interface SourceSuggestion {
  type: string;
  config: Record<string, unknown>;
  reason: string;
}

export interface DocumentArtifacts {
  formats: Record<string, string>;
}

export interface WikiPage {
  idea_id: string;
  filename: string;
  path: string;
  title: string;
}

export interface GrandTaskProposal {
  proposal_id: string;
  text: string;
  rationale: string;
  superseded_seed: string | null;
  ts: number;
}

export interface XWorkspaceProposal {
  kernel_id: string;
  source_workspace_id: string;
  source_idea_id: string;
  text: string;
  relevance_score: number;
  ts: number;
}

// ── Query keys ─────────────────────────────────────────────────────────────

const keys = {
  config: (ws: string) => ['companion', 'config', ws] as const,
  ideas: (ws: string) => ['companion', 'ideas', ws] as const,
  sources: (ws: string) => ['companion', 'sources', ws] as const,
  suggestions: (ws: string) => ['companion', 'suggestions', ws] as const,
  document: (ws: string, idea: string) =>
    ['companion', 'document', ws, idea] as const,
  wikiPages: (ws: string) => ['companion', 'wiki', ws] as const,
  proposals: (ws: string) => ['companion', 'proposals', ws] as const,
  inbox: (ws: string) => ['companion', 'inbox', ws] as const,
};

// Polling intervals
const POLL_FAST = 5_000; // 5 s — live monitor
const POLL_NORMAL = 20_000; // 20 s — ideas / inbox
const POLL_SLOW = 60_000; // 60 s — sources / proposals

// ── Config (Phase 6.5) ─────────────────────────────────────────────────────

export function useCompanionConfig(workspaceId: string) {
  return useQuery({
    queryKey: keys.config(workspaceId),
    queryFn: () =>
      api<{ workspace_id: string; config: CompanionConfig }>(
        `/api/cp/companion/config/${encodeURIComponent(workspaceId)}`,
      ),
    enabled: !!workspaceId,
    refetchInterval: POLL_SLOW,
  });
}

export function useUpdateCompanionConfig() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (vars: {
      workspaceId: string;
      patch: CompanionConfigPatch;
    }) =>
      api<{ ok: boolean; config: CompanionConfig }>(
        `/api/cp/companion/config/${encodeURIComponent(vars.workspaceId)}`,
        { method: 'POST', body: JSON.stringify(vars.patch) },
      ),
    onSuccess: (_d, vars) =>
      qc.invalidateQueries({ queryKey: keys.config(vars.workspaceId) }),
  });
}

// ── Ideas (Phase 4) ────────────────────────────────────────────────────────

export function useCompanionIdeas(workspaceId: string, limit = 50) {
  return useQuery({
    queryKey: keys.ideas(workspaceId),
    queryFn: () =>
      api<{ workspace_id: string; count: number; ideas: IdeaSummary[] }>(
        `/api/cp/companion/ideas/${encodeURIComponent(workspaceId)}?limit=${limit}`,
      ),
    enabled: !!workspaceId,
    refetchInterval: POLL_NORMAL,
  });
}

export function useThumbsFeedback() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (vars: {
      workspaceId: string;
      ideaId: string;
      polarity: 'up' | 'down';
      comment?: string;
    }) =>
      api<{ event_id: string; ok: boolean }>(
        '/api/cp/companion/feedback',
        {
          method: 'POST',
          body: JSON.stringify({
            idea_id: vars.ideaId,
            workspace_id: vars.workspaceId,
            polarity: vars.polarity,
            comment: vars.comment ?? '',
          }),
        },
      ),
    onSuccess: (_d, vars) =>
      qc.invalidateQueries({ queryKey: keys.ideas(vars.workspaceId) }),
  });
}

// ── Sources (Phase 6) ──────────────────────────────────────────────────────

export function useCompanionSources(workspaceId: string) {
  return useQuery({
    queryKey: keys.sources(workspaceId),
    queryFn: () =>
      api<{ workspace_id: string; count: number; sources: CompanionSource[] }>(
        `/api/cp/companion/sources/${encodeURIComponent(workspaceId)}`,
      ),
    enabled: !!workspaceId,
    refetchInterval: POLL_SLOW,
  });
}

export function useSourceSuggestions(workspaceId: string, enabled = false) {
  return useQuery({
    queryKey: keys.suggestions(workspaceId),
    queryFn: () =>
      api<{ workspace_id: string; count: number; suggestions: SourceSuggestion[] }>(
        `/api/cp/companion/sources/${encodeURIComponent(workspaceId)}/suggestions`,
      ),
    enabled: enabled && !!workspaceId,
    staleTime: Infinity, // suggestions fetched on demand
    refetchInterval: false,
  });
}

export function useAddSource() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (vars: {
      workspaceId: string;
      type: string;
      config: Record<string, unknown>;
      enabled?: boolean;
    }) =>
      api<{ source_id: string; ok: boolean }>(
        `/api/cp/companion/sources/${encodeURIComponent(vars.workspaceId)}`,
        {
          method: 'POST',
          body: JSON.stringify({
            type: vars.type,
            config: vars.config,
            enabled: vars.enabled ?? true,
          }),
        },
      ),
    onSuccess: (_d, vars) =>
      qc.invalidateQueries({ queryKey: keys.sources(vars.workspaceId) }),
  });
}

export function useRemoveSource() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (vars: { workspaceId: string; sourceId: string }) =>
      api<{ ok: boolean }>(
        `/api/cp/companion/sources/${encodeURIComponent(vars.workspaceId)}/${encodeURIComponent(vars.sourceId)}`,
        { method: 'DELETE' },
      ),
    onSuccess: (_d, vars) =>
      qc.invalidateQueries({ queryKey: keys.sources(vars.workspaceId) }),
  });
}

// ── Documents + Wiki (Phases 8 + 9) ────────────────────────────────────────

export function useDocumentArtifacts(workspaceId: string, ideaId: string) {
  return useQuery({
    queryKey: keys.document(workspaceId, ideaId),
    queryFn: () =>
      api<{ workspace_id: string; idea_id: string; formats: Record<string, string> }>(
        `/api/cp/companion/document/${encodeURIComponent(workspaceId)}/${encodeURIComponent(ideaId)}`,
      ),
    enabled: !!(workspaceId && ideaId),
  });
}

export function usePromoteIdea() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (vars: {
      workspaceId: string;
      ideaId: string;
      formats?: string[];
    }) =>
      api<{ ok: boolean; formats: Record<string, string> }>(
        `/api/cp/companion/promote/${encodeURIComponent(vars.workspaceId)}/${encodeURIComponent(vars.ideaId)}`,
        {
          method: 'POST',
          body: JSON.stringify({ formats: vars.formats ?? ['md'] }),
        },
      ),
    onSuccess: (_d, vars) => {
      qc.invalidateQueries({ queryKey: keys.document(vars.workspaceId, vars.ideaId) });
      qc.invalidateQueries({ queryKey: keys.ideas(vars.workspaceId) });
      qc.invalidateQueries({ queryKey: keys.wikiPages(vars.workspaceId) });
    },
  });
}

export function useWikiPages(workspaceId: string) {
  return useQuery({
    queryKey: keys.wikiPages(workspaceId),
    queryFn: () =>
      api<{ workspace_id: string; count: number; pages: WikiPage[] }>(
        `/api/cp/companion/wiki/${encodeURIComponent(workspaceId)}`,
      ),
    enabled: !!workspaceId,
    refetchInterval: POLL_SLOW,
  });
}

export async function fetchWikiPageBody(
  workspaceId: string,
  ideaId: string,
): Promise<string> {
  const res = await fetch(
    `/api/cp/companion/wiki/${encodeURIComponent(workspaceId)}/${encodeURIComponent(ideaId)}`,
    {
      headers: { Accept: 'text/markdown' },
    },
  );
  if (!res.ok) throw new Error(`wiki page fetch ${res.status}`);
  return res.text();
}

// ── Grand task (Phase 11) ──────────────────────────────────────────────────

export function useGrandTaskProposals(workspaceId: string) {
  return useQuery({
    queryKey: keys.proposals(workspaceId),
    queryFn: () =>
      api<{
        workspace_id: string;
        count: number;
        proposals: GrandTaskProposal[];
      }>(`/api/cp/companion/grand-task/${encodeURIComponent(workspaceId)}/proposals`),
    enabled: !!workspaceId,
    refetchInterval: POLL_SLOW,
  });
}

export function useAcceptGrandTask() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (vars: { workspaceId: string; proposalId: string }) =>
      api<{ ok: boolean }>(
        `/api/cp/companion/grand-task/${encodeURIComponent(vars.workspaceId)}/${encodeURIComponent(vars.proposalId)}/accept`,
        { method: 'POST' },
      ),
    onSuccess: (_d, vars) => {
      qc.invalidateQueries({ queryKey: keys.proposals(vars.workspaceId) });
      qc.invalidateQueries({ queryKey: keys.config(vars.workspaceId) });
    },
  });
}

export function useRejectGrandTask() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (vars: {
      workspaceId: string;
      proposalId: string;
      reason?: string;
    }) =>
      api<{ ok: boolean }>(
        `/api/cp/companion/grand-task/${encodeURIComponent(vars.workspaceId)}/${encodeURIComponent(vars.proposalId)}/reject`,
        {
          method: 'POST',
          body: JSON.stringify({ reason: vars.reason ?? '' }),
        },
      ),
    onSuccess: (_d, vars) =>
      qc.invalidateQueries({ queryKey: keys.proposals(vars.workspaceId) }),
  });
}

// ── Cross-workspace inbox (Phase 13) ──────────────────────────────────────

export function useXWorkspaceInbox(workspaceId: string) {
  return useQuery({
    queryKey: keys.inbox(workspaceId),
    queryFn: () =>
      api<{
        workspace_id: string;
        count: number;
        proposals: XWorkspaceProposal[];
      }>(`/api/cp/companion/xworkspace/${encodeURIComponent(workspaceId)}/inbox`),
    enabled: !!workspaceId,
    refetchInterval: POLL_NORMAL,
  });
}

export function useAcceptXWorkspaceKernel() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (vars: { workspaceId: string; kernelId: string }) =>
      api<{ ok: boolean }>(
        `/api/cp/companion/xworkspace/${encodeURIComponent(vars.workspaceId)}/inbox/${encodeURIComponent(vars.kernelId)}/accept`,
        { method: 'POST' },
      ),
    onSuccess: (_d, vars) =>
      qc.invalidateQueries({ queryKey: keys.inbox(vars.workspaceId) }),
  });
}

export function useDismissXWorkspaceKernel() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (vars: {
      workspaceId: string;
      kernelId: string;
      reason?: string;
    }) =>
      api<{ ok: boolean }>(
        `/api/cp/companion/xworkspace/${encodeURIComponent(vars.workspaceId)}/inbox/${encodeURIComponent(vars.kernelId)}/dismiss`,
        {
          method: 'POST',
          body: JSON.stringify({ reason: vars.reason ?? '' }),
        },
      ),
    onSuccess: (_d, vars) =>
      qc.invalidateQueries({ queryKey: keys.inbox(vars.workspaceId) }),
  });
}

export { POLL_FAST, POLL_NORMAL, POLL_SLOW };
