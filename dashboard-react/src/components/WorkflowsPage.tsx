// Workflow templates page — PROGRAM §46.3 (Q8.3).
//
// Read-mostly view: list templates, list recent runs, kick off a run
// with input substitution. New templates are created via the REST POST
// endpoint or operator-edited JSON — the React surface is intentionally
// minimal for v1 (no DAG editor), and the JSON payload makes hand-
// authoring + version control natural.

import { useState } from 'react';
import {
  useStartWorkflowMutation,
  useWorkflowRunStatusQuery,
  useWorkflowRunsQuery,
  useWorkflowsListQuery,
  type WorkflowRun,
  type WorkflowRunStatus,
  type WorkflowTemplate,
} from '../api/workflows';
import { Skeleton } from './ui/Skeleton';

const RUN_STATUS_BADGE: Record<
  WorkflowRunStatus,
  { bg: string; fg: string; label: string }
> = {
  queued: { bg: 'bg-[#7a8599]/15', fg: 'text-[#7a8599]', label: 'QUEUED' },
  running: { bg: 'bg-[#22d3ee]/15', fg: 'text-[#22d3ee]', label: 'RUNNING' },
  succeeded: { bg: 'bg-[#34d399]/15', fg: 'text-[#34d399]', label: 'SUCCEEDED' },
  failed: { bg: 'bg-[#f87171]/15', fg: 'text-[#f87171]', label: 'FAILED' },
  cancelled: { bg: 'bg-[#7a8599]/15', fg: 'text-[#7a8599]', label: 'CANCELLED' },
};

function StatusBadge({ status }: { status: WorkflowRunStatus }) {
  const s = RUN_STATUS_BADGE[status];
  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-medium ${s.bg} ${s.fg}`}
    >
      {s.label}
    </span>
  );
}

function formatRelative(iso: string | null | undefined): string {
  if (!iso) return '—';
  const d = new Date(iso);
  if (isNaN(d.getTime())) return iso;
  const delta = (Date.now() - d.getTime()) / 1000;
  if (delta < 60) return 'just now';
  if (delta < 3600) return `${Math.floor(delta / 60)}m ago`;
  if (delta < 86400) return `${Math.floor(delta / 3600)}h ago`;
  return d.toLocaleDateString();
}

function TemplateCard({
  template,
  onRun,
}: {
  template: WorkflowTemplate;
  onRun: (t: WorkflowTemplate) => void;
}) {
  const successRate =
    template.run_count > 0
      ? Math.round((template.success_count / template.run_count) * 100)
      : null;
  return (
    <div className="p-3 border border-[#1e2738] rounded-lg bg-[#111820]">
      <div className="flex items-center justify-between gap-2">
        <h3 className="text-sm font-semibold text-[#e2e8f0] truncate">
          {template.name}
        </h3>
        <button
          onClick={() => onRun(template)}
          className="text-xs px-2 py-0.5 rounded-full border border-[#60a5fa]/30 bg-[#60a5fa]/15 text-[#60a5fa] hover:bg-[#60a5fa]/25"
        >
          Run
        </button>
      </div>
      {template.description && (
        <p className="text-xs text-[#7a8599] mt-1 line-clamp-2">
          {template.description}
        </p>
      )}
      <div className="text-[10px] text-[#7a8599] mt-2 flex flex-wrap gap-x-3 gap-y-1">
        <span>{template.nodes.length} nodes</span>
        {template.inputs.length > 0 && (
          <span>inputs: {template.inputs.join(', ')}</span>
        )}
        <span>{template.run_count} run(s)</span>
        {successRate !== null && (
          <span className="text-[#34d399]">{successRate}% success</span>
        )}
        <span>last: {formatRelative(template.last_run_at)}</span>
      </div>
    </div>
  );
}

function RunRow({ run }: { run: WorkflowRun }) {
  return (
    <div className="text-xs p-2.5 bg-[#111820] border border-[#1e2738] rounded">
      <div className="flex items-center justify-between gap-2">
        <StatusBadge status={run.status} />
        <code className="font-mono text-[10px] text-[#7a8599]">
          {run.id.slice(0, 8)}
        </code>
        <span className="text-[#7a8599] tabular-nums">
          {formatRelative(run.started_at)}
        </span>
      </div>
      <div className="mt-1 text-[#7a8599]">
        template: <span className="text-[#cbd5e1]">{run.template_id.slice(0, 12)}…</span>
      </div>
      {run.status === 'failed' && (
        <div className="mt-1 text-[#f87171] truncate">
          {run.error_node ? `[${run.error_node}] ` : ''}
          {run.error}
        </div>
      )}
      {Object.keys(run.node_statuses).length > 0 && (
        <div className="mt-1 flex flex-wrap gap-1">
          {Object.entries(run.node_statuses).map(([nid, st]) => (
            <span
              key={nid}
              className={`text-[9px] px-1.5 py-0.5 rounded ${
                st === 'succeeded'
                  ? 'bg-[#34d399]/10 text-[#34d399]'
                  : st === 'failed'
                  ? 'bg-[#f87171]/10 text-[#f87171]'
                  : st === 'running'
                  ? 'bg-[#22d3ee]/10 text-[#22d3ee]'
                  : 'bg-[#7a8599]/10 text-[#7a8599]'
              }`}
            >
              {nid}:{st}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function RunDialog({
  template,
  onClose,
}: {
  template: WorkflowTemplate;
  onClose: () => void;
}) {
  const [inputs, setInputs] = useState<Record<string, string>>({});
  const [pendingRunId, setPendingRunId] = useState<string | null>(null);
  const start = useStartWorkflowMutation();
  const runQ = useWorkflowRunStatusQuery(pendingRunId ?? undefined);

  const onStart = async () => {
    const r = await start.mutateAsync({
      templateId: template.id,
      inputs,
    });
    setPendingRunId(r.run.id);
  };

  return (
    <div
      className="fixed inset-0 z-40 flex justify-end bg-black/60"
      onClick={onClose}
    >
      <div
        className="w-full max-w-lg bg-[#0a0e14] border-l border-[#1e2738] flex flex-col h-full"
        onClick={(e) => e.stopPropagation()}
      >
        <header className="px-5 py-4 border-b border-[#1e2738]">
          <h2 className="text-sm font-semibold text-[#e2e8f0]">
            Run: {template.name}
          </h2>
          <p className="text-xs text-[#7a8599] mt-1">
            {template.nodes.length} nodes ·{' '}
            {template.inputs.length} input(s)
          </p>
        </header>
        <div className="flex-1 overflow-y-auto p-5 space-y-3">
          {template.inputs.length === 0 ? (
            <p className="text-xs text-[#7a8599]">
              No declared inputs. Click "Start" to enqueue.
            </p>
          ) : (
            template.inputs.map((name) => (
              <div key={name}>
                <label className="text-xs text-[#7a8599] mb-0.5 block">
                  {name}
                </label>
                <input
                  type="text"
                  value={inputs[name] ?? ''}
                  onChange={(e) =>
                    setInputs({ ...inputs, [name]: e.target.value })
                  }
                  className="w-full px-2 py-1 text-xs bg-[#111820] border border-[#1e2738] rounded text-[#e2e8f0] font-mono"
                  placeholder={`{${name}}`}
                />
              </div>
            ))
          )}

          {pendingRunId && (
            <div className="mt-4 pt-4 border-t border-[#1e2738]">
              <div className="text-xs text-[#7a8599] mb-2">
                Run id:{' '}
                <code className="text-[#cbd5e1] font-mono">
                  {pendingRunId.slice(0, 12)}…
                </code>
              </div>
              {runQ.data ? (
                <RunRow run={runQ.data} />
              ) : (
                <Skeleton className="h-16" />
              )}
            </div>
          )}
        </div>
        <footer className="px-5 py-3 border-t border-[#1e2738] flex gap-2 justify-end">
          <button
            onClick={onClose}
            className="text-xs px-3 py-1.5 rounded text-[#7a8599] hover:text-[#e2e8f0] hover:bg-[#1e2738]"
          >
            Close
          </button>
          <button
            disabled={start.isPending}
            onClick={onStart}
            className="text-xs px-3 py-1.5 rounded bg-[#60a5fa]/15 text-[#60a5fa] border border-[#60a5fa]/30 hover:bg-[#60a5fa]/25 disabled:opacity-50"
          >
            {start.isPending ? 'Starting…' : 'Start run'}
          </button>
        </footer>
      </div>
    </div>
  );
}

export function WorkflowsPage() {
  const listQ = useWorkflowsListQuery();
  const runsQ = useWorkflowRunsQuery();
  const [runDialog, setRunDialog] = useState<WorkflowTemplate | null>(null);

  const templates = listQ.data?.templates ?? [];
  const runs = runsQ.data?.runs ?? [];

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-xl font-semibold text-[#e2e8f0]">Workflows</h1>
        <p className="text-sm text-[#7a8599] mt-1">
          Compose registered tools into a JSON DAG. Sits between the
          skills registry (one template, one Commander call) and the
          Forge (synthesise new tools). PROGRAM §46.3 (Q8.3).
        </p>
      </div>

      {/* Templates */}
      <section>
        <h2 className="text-xs font-semibold text-[#7a8599] uppercase tracking-wider mb-2">
          Templates ({templates.length})
        </h2>
        {listQ.isLoading ? (
          <div className="space-y-2">
            <Skeleton className="h-20" />
            <Skeleton className="h-20" />
          </div>
        ) : templates.length === 0 ? (
          <div className="p-6 text-center text-xs text-[#7a8599] border border-[#1e2738] rounded-lg bg-[#111820]">
            No saved workflow templates yet. POST a JSON spec to{' '}
            <code className="font-mono text-[#fbbf24]">/api/cp/workflows</code>
            {' '}to create one.
          </div>
        ) : (
          <div className="space-y-2">
            {templates.map((t) => (
              <TemplateCard
                key={t.id}
                template={t}
                onRun={setRunDialog}
              />
            ))}
          </div>
        )}
      </section>

      {/* Recent runs */}
      <section>
        <h2 className="text-xs font-semibold text-[#7a8599] uppercase tracking-wider mb-2">
          Recent runs ({runs.length})
        </h2>
        {runsQ.isLoading ? (
          <div className="space-y-2">
            <Skeleton className="h-16" />
          </div>
        ) : runs.length === 0 ? (
          <div className="p-6 text-center text-xs text-[#7a8599] border border-[#1e2738] rounded-lg bg-[#111820]">
            No runs yet.
          </div>
        ) : (
          <div className="space-y-1.5">
            {runs.slice(0, 20).map((r) => (
              <RunRow key={r.id} run={r} />
            ))}
          </div>
        )}
      </section>

      {runDialog && (
        <RunDialog
          template={runDialog}
          onClose={() => setRunDialog(null)}
        />
      )}
    </div>
  );
}
