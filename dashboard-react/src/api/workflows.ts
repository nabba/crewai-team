// Workflows API hooks — PROGRAM §46.3 (Q8.3).
//
// Operators see saved workflow_templates and recent runs at /workflows.
// The bulk surface is async: POST /run returns a run_id; the page
// polls /runs/<id> until terminal.

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api } from './client';

const W = '/api/cp/workflows';

export type WorkflowRunStatus =
  | 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled';

export interface WorkflowNode {
  id: string;
  tool_name: string;
  args: Record<string, unknown>;
  depends_on: string[];
  description: string;
}

export interface WorkflowTemplate {
  id: string;
  name: string;
  description: string;
  nodes: WorkflowNode[];
  inputs: string[];
  created_at: string;
  last_run_at: string | null;
  run_count: number;
  success_count: number;
}

export interface WorkflowRun {
  id: string;
  template_id: string;
  status: WorkflowRunStatus;
  started_at: string;
  finished_at: string | null;
  inputs: Record<string, unknown>;
  node_outputs: Record<string, unknown>;
  node_statuses: Record<string, string>;
  error: string;
  error_node: string;
  is_terminal: boolean;
}

export const workflowsEndpoints = {
  list: (limit = 200) => `${W}?limit=${limit}`,
  detail: (id: string) => `${W}/${encodeURIComponent(id)}`,
  create: () => W,
  remove: (id: string) => `${W}/${encodeURIComponent(id)}`,
  run: (id: string) => `${W}/${encodeURIComponent(id)}/run`,
  listRuns: (templateId?: string, limit = 50) => {
    const params = new URLSearchParams();
    if (templateId) params.set('template_id', templateId);
    params.set('limit', String(limit));
    return `${W}/runs?${params.toString()}`;
  },
  runStatus: (runId: string) => `${W}/runs/${encodeURIComponent(runId)}`,
  cancelRun: (runId: string) =>
    `${W}/runs/${encodeURIComponent(runId)}/cancel`,
};

export const workflowsKeys = {
  list: () => ['workflows', 'list'] as const,
  detail: (id: string) => ['workflows', 'detail', id] as const,
  runs: (templateId?: string) =>
    ['workflows', 'runs', templateId ?? 'all'] as const,
  run: (runId: string) => ['workflows', 'run', runId] as const,
};

export function useWorkflowsListQuery() {
  return useQuery({
    queryKey: workflowsKeys.list(),
    queryFn: () =>
      api<{ count: number; templates: WorkflowTemplate[] }>(
        workflowsEndpoints.list(),
      ),
    refetchInterval: 15_000,
  });
}

export function useWorkflowRunsQuery(templateId?: string) {
  return useQuery({
    queryKey: workflowsKeys.runs(templateId),
    queryFn: () =>
      api<{ count: number; runs: WorkflowRun[] }>(
        workflowsEndpoints.listRuns(templateId),
      ),
    refetchInterval: 5_000,
  });
}

export function useWorkflowRunStatusQuery(runId: string | undefined) {
  return useQuery({
    queryKey: workflowsKeys.run(runId ?? ''),
    queryFn: () =>
      api<WorkflowRun>(workflowsEndpoints.runStatus(runId as string)),
    enabled: Boolean(runId),
    refetchInterval: 3_000,
  });
}

export function useStartWorkflowMutation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({
      templateId, inputs,
    }: { templateId: string; inputs: Record<string, unknown> }) =>
      api<{ ok: boolean; run: WorkflowRun }>(
        workflowsEndpoints.run(templateId),
        {
          method: 'POST',
          body: JSON.stringify({ inputs }),
        },
      ),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['workflows', 'runs'] });
    },
  });
}

export function useCancelWorkflowRunMutation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ runId }: { runId: string }) =>
      api<{ ok: boolean; cancelled: string }>(
        workflowsEndpoints.cancelRun(runId),
        { method: 'POST' },
      ),
    onSuccess: (_, { runId }) => {
      qc.invalidateQueries({ queryKey: workflowsKeys.run(runId) });
      qc.invalidateQueries({ queryKey: ['workflows', 'runs'] });
    },
  });
}
