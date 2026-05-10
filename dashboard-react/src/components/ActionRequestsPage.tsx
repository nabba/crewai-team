// Action-requests operator surface.
// Lists agent-proposed external actions (emails, calendar invites,
// etc) with status filter; opens a drawer for the full payload and
// per-status action buttons (approve / reject / retry-apply).

import { useState } from 'react';
import { Skeleton } from './ui/Skeleton';
import {
  useActionApproveMutation,
  useActionDetailQuery,
  useActionListQuery,
  useActionRejectMutation,
  useActionRetryApplyMutation,
} from '../api/action_requests';
import type { ActionRequest, ActionStatus } from '../types/action_requests';

const STATUS_FILTERS: (ActionStatus | 'all')[] = [
  'all',
  'pending',
  'approved',
  'applied',
  'apply_failed',
  'rejected',
  'invalid',
  'timeout',
];

const STATUS_BADGE: Record<
  ActionStatus,
  { bg: string; fg: string; border: string; label: string }
> = {
  pending: {
    bg: 'bg-[#fbbf24]/15', fg: 'text-[#fbbf24]',
    border: 'border-[#fbbf24]/30', label: 'PENDING',
  },
  approved: {
    bg: 'bg-[#60a5fa]/15', fg: 'text-[#60a5fa]',
    border: 'border-[#60a5fa]/30', label: 'APPROVED',
  },
  applied: {
    bg: 'bg-[#34d399]/15', fg: 'text-[#34d399]',
    border: 'border-[#34d399]/30', label: 'APPLIED',
  },
  apply_failed: {
    bg: 'bg-[#f87171]/15', fg: 'text-[#f87171]',
    border: 'border-[#f87171]/30', label: 'APPLY FAILED',
  },
  rejected: {
    bg: 'bg-[#7a8599]/15', fg: 'text-[#7a8599]',
    border: 'border-[#7a8599]/30', label: 'REJECTED',
  },
  invalid: {
    bg: 'bg-[#f87171]/15', fg: 'text-[#f87171]',
    border: 'border-[#f87171]/40', label: 'INVALID',
  },
  timeout: {
    bg: 'bg-[#7a8599]/15', fg: 'text-[#7a8599]',
    border: 'border-[#7a8599]/30', label: 'TIMEOUT',
  },
};

function StatusBadge({ status }: { status: ActionStatus }) {
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

export function ActionRequestsPage() {
  const [filter, setFilter] = useState<ActionStatus | 'all'>('all');
  const [openId, setOpenId] = useState<string | null>(null);
  const list = useActionListQuery(filter === 'all' ? undefined : filter);

  return (
    <div className="flex h-full flex-col gap-4 p-6">
      <header className="flex flex-wrap items-baseline gap-4">
        <h1 className="text-xl font-semibold tracking-tight text-[#e6edf7]">
          Action Requests
        </h1>
        <p className="text-sm text-[#7a8599]">
          Operator-gated NON-CODE actions: email drafts, calendar invites,
          and the like. Approve via 👍 in Signal or here. Each action type
          has its own validator + applier; the gate is universal.
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
            No action requests
            {filter !== 'all' ? ` with status ${filter}` : ''} yet.
            Companion proposes actions; this page lists them as they arrive.
          </div>
        ) : (
          <ul className="divide-y divide-[#1f2937] overflow-y-auto">
            {list.data.action_requests.map((r) => (
              <Row key={r.id} request={r} onClick={() => setOpenId(r.id)} />
            ))}
          </ul>
        )}
      </div>

      {openId && (
        <Drawer requestId={openId} onClose={() => setOpenId(null)} />
      )}
    </div>
  );
}

function Row({
  request,
  onClick,
}: {
  request: ActionRequest;
  onClick: () => void;
}) {
  return (
    <li
      onClick={onClick}
      className="cursor-pointer px-4 py-3 transition hover:bg-[#111827]"
    >
      <div className="flex flex-wrap items-baseline gap-3">
        <StatusBadge status={request.status} />
        <code className="font-mono text-xs text-[#a78bfa]">
          {request.action_type}
        </code>
        <span className="text-sm text-[#e6edf7]">{request.summary}</span>
        <span className="ml-auto text-xs text-[#7a8599]">
          {relTime(request.created_at)} · {request.requestor}
        </span>
      </div>
      {request.invalid_reason && (
        <p className="mt-1 truncate text-xs text-[#f87171]">
          INVALID: {request.invalid_reason}
        </p>
      )}
      {request.apply_error && (
        <p className="mt-1 truncate text-xs text-[#f87171]">
          APPLY ERROR: {request.apply_error}
        </p>
      )}
    </li>
  );
}

function Drawer({
  requestId,
  onClose,
}: {
  requestId: string;
  onClose: () => void;
}) {
  const detail = useActionDetailQuery(requestId);
  const r = detail.data;
  const approve = useActionApproveMutation();
  const reject = useActionRejectMutation();
  const retry = useActionRetryApplyMutation();
  const [rejectReason, setRejectReason] = useState('');

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
              <h2 className="text-lg font-semibold">{r.summary}</h2>
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
              <span>action: <code className="text-[#a78bfa]">{r.action_type}</code></span>
              <span>requestor: <code className="text-[#9ca3af]">{r.requestor}</code></span>
              <span>created: {new Date(r.created_at).toLocaleString()}</span>
              {r.decided_by && <span>decided_by: {r.decided_by}</span>}
            </div>

            <section>
              <h3 className="mb-1 text-xs font-mono uppercase tracking-wide text-[#7a8599]">
                Reason
              </h3>
              <p className="whitespace-pre-wrap text-[#cbd5e1]">{r.reason}</p>
            </section>

            <section>
              <h3 className="mb-1 text-xs font-mono uppercase tracking-wide text-[#7a8599]">
                Payload
              </h3>
              <pre className="overflow-x-auto rounded bg-[#000] p-3 text-[11px] text-[#cbd5e1]">
                {JSON.stringify(r.data, null, 2)}
              </pre>
            </section>

            {r.invalid_reason && (
              <section>
                <h3 className="mb-1 text-xs font-mono uppercase tracking-wide text-[#f87171]">
                  Validator rejection
                </h3>
                <p className="text-[#f87171]">{r.invalid_reason}</p>
              </section>
            )}

            {r.apply_error && (
              <section>
                <h3 className="mb-1 text-xs font-mono uppercase tracking-wide text-[#f87171]">
                  Apply error
                </h3>
                <p className="text-[#f87171]">{r.apply_error}</p>
              </section>
            )}

            {Object.keys(r.apply_artifact).length > 0 && (
              <section>
                <h3 className="mb-1 text-xs font-mono uppercase tracking-wide text-[#7a8599]">
                  Apply artifact
                </h3>
                <pre className="overflow-x-auto rounded bg-[#000] p-3 text-[11px] text-[#cbd5e1]">
                  {JSON.stringify(r.apply_artifact, null, 2)}
                </pre>
              </section>
            )}

            <section className="mt-2 border-t border-[#1f2937] pt-3">
              <h3 className="mb-2 text-xs font-mono uppercase tracking-wide text-[#7a8599]">
                Actions
              </h3>
              <div className="flex flex-wrap gap-2">
                {r.status === 'pending' && (
                  <>
                    <button
                      type="button"
                      onClick={() =>
                        approve.mutate({ id: r.id, reason: 'approved via React' })
                      }
                      disabled={approve.isPending}
                      className="rounded border border-[#34d399]/40 bg-[#34d399]/10 px-3 py-1 text-xs text-[#34d399] hover:bg-[#34d399]/20 disabled:opacity-50"
                    >
                      {approve.isPending ? 'applying...' : 'Approve + apply'}
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
                        reject.mutate({
                          id: r.id,
                          reason: rejectReason || undefined,
                        })
                      }
                      disabled={reject.isPending}
                      className="rounded border border-[#f87171]/40 bg-[#f87171]/10 px-3 py-1 text-xs text-[#f87171] hover:bg-[#f87171]/20 disabled:opacity-50"
                    >
                      {reject.isPending ? 'rejecting...' : 'Reject'}
                    </button>
                  </>
                )}
                {r.status === 'apply_failed' && (
                  <button
                    type="button"
                    onClick={() => retry.mutate({ id: r.id })}
                    disabled={retry.isPending}
                    className="rounded border border-[#fbbf24]/40 bg-[#fbbf24]/10 px-3 py-1 text-xs text-[#fbbf24] hover:bg-[#fbbf24]/20 disabled:opacity-50"
                  >
                    {retry.isPending ? 'retrying...' : 'Retry apply'}
                  </button>
                )}
                {r.is_terminal && r.status !== 'apply_failed' && (
                  <span className="text-xs text-[#7a8599]">
                    (terminal — no further actions)
                  </span>
                )}
              </div>
              {(approve.error || reject.error || retry.error) && (
                <p className="mt-2 text-xs text-[#f87171]">
                  {((approve.error || reject.error || retry.error) as Error).message}
                </p>
              )}
            </section>
          </>
        )}
      </aside>
    </div>
  );
}
