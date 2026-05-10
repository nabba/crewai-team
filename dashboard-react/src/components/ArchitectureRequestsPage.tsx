// Architecture-requests operator surface.
// Lists agent-proposed subsystem additions with status filter; opens a
// drawer for the full proposal (motivation, file_layout, integration
// points, env switches, test plan) and per-status action buttons
// (approve / reject / scaffold / abandon / mark-complete).
//
// Backend lives at /api/cp/architecture-requests — see
// app/control_plane/architecture_requests_api.py.

import { useState } from 'react';
import { Skeleton } from './ui/Skeleton';
import {
  useArchAbandonMutation,
  useArchApproveMutation,
  useArchAuditQuery,
  useArchDetailQuery,
  useArchListQuery,
  useArchManifestQuery,
  useArchMarkCompleteMutation,
  useArchRejectMutation,
  useArchScaffoldMutation,
} from '../api/architecture_requests';
import type {
  ArchitectureRequest,
  ArchStatus,
} from '../types/architecture_requests';

const STATUS_FILTERS: (ArchStatus | 'all')[] = [
  'all',
  'proposed',
  'approved',
  'scaffolded',
  'implementing',
  'completed',
  'rejected',
  'tier_immutable_refused',
  'timeout',
  'abandoned',
];

const STATUS_BADGE: Record<
  ArchStatus,
  { bg: string; fg: string; border: string; label: string }
> = {
  proposed: {
    bg: 'bg-[#fbbf24]/15',
    fg: 'text-[#fbbf24]',
    border: 'border-[#fbbf24]/30',
    label: 'PROPOSED',
  },
  approved: {
    bg: 'bg-[#60a5fa]/15',
    fg: 'text-[#60a5fa]',
    border: 'border-[#60a5fa]/30',
    label: 'APPROVED',
  },
  scaffolded: {
    bg: 'bg-[#a78bfa]/15',
    fg: 'text-[#a78bfa]',
    border: 'border-[#a78bfa]/30',
    label: 'SCAFFOLDED',
  },
  implementing: {
    bg: 'bg-[#22d3ee]/15',
    fg: 'text-[#22d3ee]',
    border: 'border-[#22d3ee]/30',
    label: 'IMPLEMENTING',
  },
  completed: {
    bg: 'bg-[#34d399]/15',
    fg: 'text-[#34d399]',
    border: 'border-[#34d399]/30',
    label: 'COMPLETED',
  },
  rejected: {
    bg: 'bg-[#7a8599]/15',
    fg: 'text-[#7a8599]',
    border: 'border-[#7a8599]/30',
    label: 'REJECTED',
  },
  tier_immutable_refused: {
    bg: 'bg-[#f87171]/15',
    fg: 'text-[#f87171]',
    border: 'border-[#f87171]/40',
    label: 'TIER_IMMUTABLE',
  },
  timeout: {
    bg: 'bg-[#7a8599]/15',
    fg: 'text-[#7a8599]',
    border: 'border-[#7a8599]/30',
    label: 'TIMEOUT',
  },
  abandoned: {
    bg: 'bg-[#7a8599]/15',
    fg: 'text-[#7a8599]',
    border: 'border-[#7a8599]/30',
    label: 'ABANDONED',
  },
};

function StatusBadge({ status }: { status: ArchStatus }) {
  const s = STATUS_BADGE[status];
  return (
    <span
      className={`inline-block rounded border px-2 py-[2px] text-[10px] font-mono uppercase tracking-wide ${s.bg} ${s.fg} ${s.border}`}
    >
      {s.label}
    </span>
  );
}

function relTime(iso: string): string {
  const ms = Date.now() - new Date(iso).getTime();
  const s = Math.max(0, Math.floor(ms / 1000));
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

export function ArchitectureRequestsPage() {
  const [filter, setFilter] = useState<ArchStatus | 'all'>('all');
  const [openId, setOpenId] = useState<string | null>(null);

  const list = useArchListQuery(filter === 'all' ? undefined : filter);

  return (
    <div className="flex h-full flex-col gap-4 p-6">
      <header className="flex flex-wrap items-baseline gap-4">
        <h1 className="text-xl font-semibold tracking-tight text-[#e6edf7]">
          Architecture Requests
        </h1>
        <p className="text-sm text-[#7a8599]">
          Agent-proposed subsystem additions. Approve the design at
          package granularity; per-file landing flows through the
          standard <a href="/changes" className="underline hover:text-[#e6edf7]">change-request</a> gate.
        </p>
      </header>

      <div className="flex flex-wrap gap-2">
        {STATUS_FILTERS.map((s) => (
          <button
            type="button"
            key={s}
            onClick={() => setFilter(s)}
            className={`rounded border px-2 py-1 text-xs font-mono uppercase tracking-wide transition ${
              filter === s
                ? 'border-[#60a5fa] bg-[#60a5fa]/10 text-[#e6edf7]'
                : 'border-[#1f2937] text-[#7a8599] hover:border-[#374151] hover:text-[#9ca3af]'
            }`}
          >
            {s}
          </button>
        ))}
      </div>

      <div className="flex-1 overflow-hidden rounded border border-[#1f2937] bg-[#0c1019]">
        {list.isLoading ? (
          <div className="space-y-2 p-4">
            <Skeleton className="h-12" />
            <Skeleton className="h-12" />
            <Skeleton className="h-12" />
          </div>
        ) : list.error ? (
          <div className="p-4 text-sm text-[#f87171]">
            Failed to load: {(list.error as Error).message}
          </div>
        ) : !list.data || list.data.count === 0 ? (
          <div className="flex h-full items-center justify-center p-8 text-center text-sm text-[#7a8599]">
            No architecture requests
            {filter !== 'all' ? ` with status ${filter}` : ''} yet. Agents
            propose new subsystems via the{' '}
            <code className="mx-1 rounded bg-[#1f2937] px-1 py-[1px] text-xs">
              propose_architecture
            </code>{' '}
            tool; this page lists them as they arrive.
          </div>
        ) : (
          <ul className="divide-y divide-[#1f2937] overflow-y-auto">
            {list.data.architecture_requests.map((r) => (
              <RequestRow
                key={r.id}
                request={r}
                onClick={() => setOpenId(r.id)}
              />
            ))}
          </ul>
        )}
      </div>

      {openId && (
        <RequestDrawer
          requestId={openId}
          onClose={() => setOpenId(null)}
        />
      )}
    </div>
  );
}

function RequestRow({
  request,
  onClick,
}: {
  request: ArchitectureRequest;
  onClick: () => void;
}) {
  return (
    <li
      onClick={onClick}
      className="cursor-pointer px-4 py-3 transition hover:bg-[#111827]"
    >
      <div className="flex flex-wrap items-baseline gap-3">
        <StatusBadge status={request.status} />
        <code className="font-mono text-xs text-[#9ca3af]">
          {request.package_path}
        </code>
        <span className="text-sm text-[#e6edf7]">{request.intent}</span>
        <span className="ml-auto text-xs text-[#7a8599]">
          {relTime(request.created_at)} · {request.requestor}
        </span>
      </div>
      <div className="mt-1 flex flex-wrap gap-3 text-xs text-[#7a8599]">
        <span>{request.file_layout.length} file(s)</span>
        <span>·</span>
        <span>{request.integration_points.length} integration(s)</span>
        <span>·</span>
        <span>{Object.keys(request.env_switches).length} env switch(es)</span>
        {request.child_change_request_ids.length > 0 && (
          <>
            <span>·</span>
            <span>{request.child_change_request_ids.length} child CR(s)</span>
          </>
        )}
      </div>
    </li>
  );
}

function RequestDrawer({
  requestId,
  onClose,
}: {
  requestId: string;
  onClose: () => void;
}) {
  const detail = useArchDetailQuery(requestId);
  const audit = useArchAuditQuery(requestId);
  const r = detail.data;
  const showManifest =
    r?.scaffold_dir !== undefined &&
    r?.scaffold_dir !== null &&
    r.scaffold_dir.length > 0;
  const manifest = useArchManifestQuery(requestId, Boolean(showManifest));

  const approve = useArchApproveMutation();
  const reject = useArchRejectMutation();
  const scaffold = useArchScaffoldMutation();
  const abandon = useArchAbandonMutation();
  const markComplete = useArchMarkCompleteMutation();
  const [rejectReason, setRejectReason] = useState('');
  const [abandonReason, setAbandonReason] = useState('');

  return (
    <div
      className="fixed inset-0 z-30 flex justify-end bg-black/40"
      onClick={onClose}
    >
      <aside
        className="flex h-full w-full max-w-3xl flex-col gap-3 overflow-y-auto border-l border-[#1f2937] bg-[#0c1019] p-6 text-sm text-[#e6edf7]"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-baseline gap-3">
          {r ? (
            <>
              <StatusBadge status={r.status} />
              <h2 className="text-lg font-semibold">{r.intent}</h2>
            </>
          ) : (
            <Skeleton className="h-6 w-3/4" />
          )}
          <button
            type="button"
            className="ml-auto text-[#7a8599] hover:text-[#e6edf7]"
            onClick={onClose}
          >
            close
          </button>
        </div>

        {r && (
          <>
            <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-[#7a8599]">
              <span>id: <code className="text-[#9ca3af]">{r.id}</code></span>
              <span>requestor: <code className="text-[#9ca3af]">{r.requestor}</code></span>
              <span>package: <code className="text-[#9ca3af]">{r.package_path}</code></span>
              <span>created: {new Date(r.created_at).toLocaleString()}</span>
              {r.decided_by && <span>decided_by: {r.decided_by}</span>}
            </div>

            <section>
              <h3 className="mb-1 text-xs font-mono uppercase tracking-wide text-[#7a8599]">
                Motivation
              </h3>
              <p className="whitespace-pre-wrap text-[#cbd5e1]">{r.motivation}</p>
            </section>

            <section>
              <h3 className="mb-1 text-xs font-mono uppercase tracking-wide text-[#7a8599]">
                File layout ({r.file_layout.length})
              </h3>
              <ul className="space-y-1">
                {r.file_layout.map((f) => (
                  <li key={f.path} className="font-mono text-xs">
                    <span className="text-[#60a5fa]">{f.path}</span>
                    <span className="text-[#7a8599]"> — {f.purpose}</span>
                  </li>
                ))}
              </ul>
            </section>

            <section>
              <h3 className="mb-1 text-xs font-mono uppercase tracking-wide text-[#7a8599]">
                Integration points ({r.integration_points.length})
              </h3>
              <ul className="space-y-1">
                {r.integration_points.map((ip, i) => (
                  <li key={`${ip.kind}-${ip.target_module}-${i}`} className="text-xs">
                    <code className="text-[#a78bfa]">{ip.kind}</code>{' '}
                    →{' '}
                    <code className="text-[#9ca3af]">{ip.target_module}</code>
                  </li>
                ))}
              </ul>
            </section>

            <section>
              <h3 className="mb-1 text-xs font-mono uppercase tracking-wide text-[#7a8599]">
                Env switches
              </h3>
              {Object.keys(r.env_switches).length === 0 ? (
                <p className="text-xs text-[#7a8599]">(none)</p>
              ) : (
                <ul className="space-y-1 font-mono text-xs">
                  {Object.entries(r.env_switches).map(([k, v]) => (
                    <li key={k}>
                      <span className="text-[#fbbf24]">{k}</span>{' '}
                      <span className="text-[#7a8599]">= {v}</span>
                    </li>
                  ))}
                </ul>
              )}
            </section>

            <section>
              <h3 className="mb-1 text-xs font-mono uppercase tracking-wide text-[#7a8599]">
                Test plan
              </h3>
              <p className="whitespace-pre-wrap text-[#cbd5e1]">{r.test_plan}</p>
            </section>

            {r.scaffold_dir && (
              <section>
                <h3 className="mb-1 text-xs font-mono uppercase tracking-wide text-[#7a8599]">
                  Scaffold staged at
                </h3>
                <code className="text-xs text-[#9ca3af]">{r.scaffold_dir}</code>
                {manifest.data && (
                  <pre className="mt-2 max-h-72 overflow-auto rounded bg-[#000] p-3 text-[11px] text-[#cbd5e1]">
                    {manifest.data.manifest_text}
                  </pre>
                )}
              </section>
            )}

            {r.child_change_request_ids.length > 0 && (
              <section>
                <h3 className="mb-1 text-xs font-mono uppercase tracking-wide text-[#7a8599]">
                  Child change-requests
                </h3>
                <ul className="space-y-1 text-xs">
                  {r.child_change_request_ids.map((cid) => (
                    <li key={cid}>
                      <a
                        href={`/changes`}
                        className="text-[#60a5fa] underline hover:text-[#e6edf7]"
                      >
                        {cid}
                      </a>
                    </li>
                  ))}
                </ul>
              </section>
            )}

            {audit.data && audit.data.entries.length > 0 && (
              <section>
                <h3 className="mb-1 text-xs font-mono uppercase tracking-wide text-[#7a8599]">
                  Audit trail
                </h3>
                <ul className="space-y-1 text-xs">
                  {audit.data.entries.map((e, i) => (
                    <li key={`${e.event}-${i}`} className="font-mono">
                      <span className="text-[#7a8599]">
                        {new Date(e.ts).toLocaleString()}
                      </span>{' '}
                      <span className="text-[#fbbf24]">{e.event}</span>{' '}
                      <span className="text-[#9ca3af]">→ {e.status}</span>
                      {e.decided_by && (
                        <span className="text-[#7a8599]"> ({e.decided_by})</span>
                      )}
                    </li>
                  ))}
                </ul>
              </section>
            )}

            <section className="mt-2 border-t border-[#1f2937] pt-3">
              <h3 className="mb-2 text-xs font-mono uppercase tracking-wide text-[#7a8599]">
                Actions
              </h3>
              <div className="flex flex-wrap gap-2">
                {r.status === 'proposed' && (
                  <>
                    <button
                      type="button"
                      onClick={() =>
                        approve.mutate({ id: r.id, reason: 'approved via React' })
                      }
                      disabled={approve.isPending}
                      className="rounded border border-[#34d399]/40 bg-[#34d399]/10 px-3 py-1 text-xs text-[#34d399] hover:bg-[#34d399]/20 disabled:opacity-50"
                    >
                      {approve.isPending ? 'approving...' : 'Approve'}
                    </button>
                    <input
                      type="text"
                      placeholder="rejection reason (optional)"
                      value={rejectReason}
                      onChange={(e) => setRejectReason(e.target.value)}
                      className="rounded border border-[#1f2937] bg-[#0c1019] px-2 py-1 text-xs"
                    />
                    <button
                      type="button"
                      onClick={() =>
                        reject.mutate({ id: r.id, reason: rejectReason || undefined })
                      }
                      disabled={reject.isPending}
                      className="rounded border border-[#f87171]/40 bg-[#f87171]/10 px-3 py-1 text-xs text-[#f87171] hover:bg-[#f87171]/20 disabled:opacity-50"
                    >
                      {reject.isPending ? 'rejecting...' : 'Reject'}
                    </button>
                  </>
                )}
                {r.status === 'approved' && (
                  <button
                    type="button"
                    onClick={() => scaffold.mutate({ id: r.id })}
                    disabled={scaffold.isPending}
                    className="rounded border border-[#a78bfa]/40 bg-[#a78bfa]/10 px-3 py-1 text-xs text-[#a78bfa] hover:bg-[#a78bfa]/20 disabled:opacity-50"
                  >
                    {scaffold.isPending ? 'scaffolding...' : 'Scaffold'}
                  </button>
                )}
                {r.status === 'implementing' && (
                  <button
                    type="button"
                    onClick={() => markComplete.mutate({ id: r.id })}
                    disabled={markComplete.isPending}
                    className="rounded border border-[#34d399]/40 bg-[#34d399]/10 px-3 py-1 text-xs text-[#34d399] hover:bg-[#34d399]/20 disabled:opacity-50"
                  >
                    {markComplete.isPending ? 'completing...' : 'Mark complete'}
                  </button>
                )}
                {(r.status === 'scaffolded' || r.status === 'implementing') && (
                  <>
                    <input
                      type="text"
                      placeholder="abandon reason"
                      value={abandonReason}
                      onChange={(e) => setAbandonReason(e.target.value)}
                      className="rounded border border-[#1f2937] bg-[#0c1019] px-2 py-1 text-xs"
                    />
                    <button
                      type="button"
                      onClick={() =>
                        abandon.mutate({
                          id: r.id,
                          reason: abandonReason || 'abandoned via React',
                        })
                      }
                      disabled={abandon.isPending}
                      className="rounded border border-[#7a8599]/40 bg-[#7a8599]/10 px-3 py-1 text-xs text-[#7a8599] hover:bg-[#7a8599]/20 disabled:opacity-50"
                    >
                      {abandon.isPending ? 'abandoning...' : 'Abandon'}
                    </button>
                  </>
                )}
                {r.is_terminal && (
                  <span className="text-xs text-[#7a8599]">
                    (terminal — no further actions)
                  </span>
                )}
              </div>
              {(approve.error ||
                reject.error ||
                scaffold.error ||
                abandon.error ||
                markComplete.error) && (
                <p className="mt-2 text-xs text-[#f87171]">
                  {((approve.error ||
                    reject.error ||
                    scaffold.error ||
                    abandon.error ||
                    markComplete.error) as Error).message}
                </p>
              )}
            </section>
          </>
        )}
      </aside>
    </div>
  );
}
