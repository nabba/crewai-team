// Phase 5.3b — operator React surface for the change-request system.
// Lists agent-proposed code changes with status filter, opens a drawer
// for the diff and per-status action buttons (approve / reject /
// rollback / retry-apply).
//
// Backend lives at /api/cp/changes — see app/control_plane/changes_api.py
// and docs/CHANGE_REQUESTS.md.

import { useState } from 'react';
import { Skeleton } from './ui/Skeleton';
import {
  useApproveChangeMutation,
  useChangeDetailQuery,
  useChangesListQuery,
  useRejectChangeMutation,
  useRetryApplyMutation,
  useRollbackChangeMutation,
} from '../api/changes';
import type { ChangeRequest, ChangeStatus } from '../types/changes';

const STATUS_FILTERS: (ChangeStatus | 'all')[] = [
  'all',
  'pending',
  'approved',
  'applied',
  'apply_failed',
  'rejected',
  'rolled_back',
  'tier_immutable_refused',
  'timeout',
];

const STATUS_BADGE: Record<
  ChangeStatus,
  { bg: string; fg: string; border: string; label: string }
> = {
  pending: {
    bg: 'bg-[#fbbf24]/15',
    fg: 'text-[#fbbf24]',
    border: 'border-[#fbbf24]/30',
    label: 'PENDING',
  },
  approved: {
    bg: 'bg-[#60a5fa]/15',
    fg: 'text-[#60a5fa]',
    border: 'border-[#60a5fa]/30',
    label: 'APPROVED',
  },
  applied: {
    bg: 'bg-[#34d399]/15',
    fg: 'text-[#34d399]',
    border: 'border-[#34d399]/30',
    label: 'APPLIED',
  },
  apply_failed: {
    bg: 'bg-[#f87171]/15',
    fg: 'text-[#f87171]',
    border: 'border-[#f87171]/30',
    label: 'APPLY FAILED',
  },
  rejected: {
    bg: 'bg-[#7a8599]/15',
    fg: 'text-[#7a8599]',
    border: 'border-[#7a8599]/30',
    label: 'REJECTED',
  },
  rolled_back: {
    bg: 'bg-[#a78bfa]/15',
    fg: 'text-[#a78bfa]',
    border: 'border-[#a78bfa]/30',
    label: 'ROLLED BACK',
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
};

function StatusBadge({ status }: { status: ChangeStatus }) {
  const s = STATUS_BADGE[status];
  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-medium border ${s.bg} ${s.fg} ${s.border}`}
    >
      {s.label}
    </span>
  );
}

function ProtectedBadge({ isProtected }: { isProtected: boolean }) {
  if (!isProtected) return null;
  return (
    <span
      className="inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-medium bg-[#f87171]/15 text-[#f87171] border border-[#f87171]/40"
      title="Path is in TIER_IMMUTABLE — agent path can never modify it, even with operator approval."
    >
      🛑 PROTECTED
    </span>
  );
}

// Render a unified diff with line-level coloring. Lines starting with
// `+` are green; `-` are red; `@@` are blue (hunk headers); everything
// else is muted. Long diffs are truncated to 800 lines with a footer.
function DiffView({ diff }: { diff: string }) {
  if (!diff) {
    return (
      <div className="text-xs text-[#7a8599] italic">
        (no diff — likely a new-file creation with empty old_content)
      </div>
    );
  }
  const allLines = diff.split('\n');
  const MAX = 800;
  const truncated = allLines.length > MAX;
  const lines = truncated ? allLines.slice(0, MAX) : allLines;

  return (
    <div className="rounded-md border border-[#1e2738] bg-[#0a0e14] overflow-x-auto font-mono text-[11px] leading-snug">
      <pre className="px-3 py-2 whitespace-pre">
        {lines.map((line, i) => {
          let cls = 'text-[#cbd5e1]';
          if (line.startsWith('+++') || line.startsWith('---'))
            cls = 'text-[#7a8599]';
          else if (line.startsWith('@@')) cls = 'text-[#60a5fa]';
          else if (line.startsWith('+')) cls = 'text-[#34d399]';
          else if (line.startsWith('-')) cls = 'text-[#f87171]';
          else cls = 'text-[#94a3b8]';
          return (
            <div key={i} className={cls}>
              {line || ' '}
            </div>
          );
        })}
        {truncated && (
          <div className="text-[#fbbf24] mt-2">
            … diff truncated at {MAX} lines (full diff is {allLines.length}{' '}
            lines)
          </div>
        )}
      </pre>
    </div>
  );
}

function ChangeRow({
  change,
  onClick,
  isActive,
}: {
  change: ChangeRequest;
  onClick: () => void;
  isActive: boolean;
}) {
  return (
    <button
      onClick={onClick}
      className={`w-full text-left p-4 rounded-lg border transition-colors ${
        isActive
          ? 'border-[#60a5fa]/40 bg-[#60a5fa]/5'
          : 'border-[#1e2738] bg-[#111820] hover:border-[#60a5fa]/30 hover:bg-[#1e2738]/50'
      }`}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 flex-wrap">
            <code className="text-sm font-semibold text-[#e2e8f0] truncate">
              {change.path}
            </code>
            <StatusBadge status={change.status} />
            <ProtectedBadge isProtected={change.is_protected} />
            <span className="text-[10px] font-mono text-[#7a8599]">
              by {change.requestor}
            </span>
          </div>
          {change.reason && (
            <div className="text-xs text-[#cbd5e1] mt-1 line-clamp-2">
              {change.reason}
            </div>
          )}
          {change.apply_error && (
            <div className="text-xs text-[#f87171] mt-1 truncate">
              error: {change.apply_error}
            </div>
          )}
        </div>
        <div className="text-[10px] text-[#7a8599] font-mono flex-shrink-0 text-right">
          <div>{new Date(change.created_at).toLocaleString()}</div>
          <div className="opacity-60 mt-0.5">{change.id.slice(0, 8)}…</div>
        </div>
      </div>
    </button>
  );
}

function ActionButtons({
  change,
  onClose,
}: {
  change: ChangeRequest;
  onClose: () => void;
}) {
  const [confirmRollback, setConfirmRollback] = useState(false);
  const approve = useApproveChangeMutation();
  const reject = useRejectChangeMutation();
  const rollback = useRollbackChangeMutation();
  const retry = useRetryApplyMutation();

  // TIER_IMMUTABLE_REFUSED has no action — the rule is absolute.
  if (change.status === 'tier_immutable_refused') {
    return (
      <div className="text-xs text-[#f87171] bg-[#f87171]/10 border border-[#f87171]/30 rounded-md p-3">
        <strong className="block mb-1">🛑 TIER_IMMUTABLE</strong>
        This path is in the absolute-no-modify list. No human-override
        path can authorize an agent write. Operator must edit directly
        via PR (gateway redeploy required).
      </div>
    );
  }

  const pending = (
    approve.isPending ||
    reject.isPending ||
    rollback.isPending ||
    retry.isPending
  );

  return (
    <div className="space-y-3">
      {change.status === 'pending' && (
        <div className="flex flex-wrap gap-2">
          <button
            disabled={pending}
            onClick={() => approve.mutate({ id: change.id })}
            className="px-4 py-2 rounded-md text-sm font-medium bg-[#34d399]/15 text-[#34d399] hover:bg-[#34d399]/25 border border-[#34d399]/30 disabled:opacity-50 transition-colors"
          >
            ✓ Approve + apply
          </button>
          <button
            disabled={pending}
            onClick={() =>
              reject.mutate({ id: change.id, reason: 'rejected via React' })
            }
            className="px-4 py-2 rounded-md text-sm font-medium bg-[#f87171]/15 text-[#f87171] hover:bg-[#f87171]/25 border border-[#f87171]/30 disabled:opacity-50 transition-colors"
          >
            ✗ Reject
          </button>
        </div>
      )}

      {change.status === 'apply_failed' && (
        <div className="space-y-2">
          <div className="text-xs text-[#f87171]">
            Apply failed:{' '}
            <code className="font-mono">{change.apply_error ?? 'unknown error'}</code>
          </div>
          <div className="flex flex-wrap gap-2">
            <button
              disabled={pending}
              onClick={() => retry.mutate({ id: change.id })}
              className="px-4 py-2 rounded-md text-sm font-medium bg-[#fbbf24]/15 text-[#fbbf24] hover:bg-[#fbbf24]/25 border border-[#fbbf24]/30 disabled:opacity-50 transition-colors"
            >
              ↻ Retry apply
            </button>
          </div>
        </div>
      )}

      {change.is_rollbackable && (
        <div className="space-y-2">
          {!confirmRollback ? (
            <button
              disabled={pending}
              onClick={() => setConfirmRollback(true)}
              className="px-4 py-2 rounded-md text-sm font-medium bg-[#a78bfa]/15 text-[#a78bfa] hover:bg-[#a78bfa]/25 border border-[#a78bfa]/30 disabled:opacity-50 transition-colors"
            >
              ⤺ Roll back…
            </button>
          ) : (
            <div className="flex flex-wrap items-center gap-2 p-2 border border-[#a78bfa]/30 rounded-md bg-[#a78bfa]/5">
              <span className="text-xs text-[#cbd5e1]">
                Revert {change.git_commit_sha?.slice(0, 8) ?? '?'} on{' '}
                <code className="font-mono">{change.path}</code>?
              </span>
              <button
                disabled={pending}
                onClick={() => {
                  rollback.mutate(
                    { id: change.id },
                    { onSettled: () => setConfirmRollback(false) },
                  );
                }}
                className="px-3 py-1 rounded text-xs font-medium bg-[#a78bfa]/25 text-[#a78bfa] hover:bg-[#a78bfa]/40 border border-[#a78bfa]/40 disabled:opacity-50"
              >
                Confirm rollback
              </button>
              <button
                disabled={pending}
                onClick={() => setConfirmRollback(false)}
                className="px-3 py-1 rounded text-xs text-[#7a8599] hover:text-[#cbd5e1]"
              >
                Cancel
              </button>
            </div>
          )}
        </div>
      )}

      {(approve.error || reject.error || rollback.error || retry.error) && (
        <div className="text-xs text-[#f87171] bg-[#f87171]/10 border border-[#f87171]/30 rounded-md p-2">
          {(approve.error || reject.error || rollback.error || retry.error)?.message}
        </div>
      )}

      {change.is_terminal && (
        <button
          onClick={onClose}
          className="px-3 py-1 rounded text-xs text-[#7a8599] hover:text-[#cbd5e1]"
        >
          Close
        </button>
      )}
    </div>
  );
}

function ChangeDetailDrawer({
  changeId,
  onClose,
}: {
  changeId: string | null;
  onClose: () => void;
}) {
  // Hooks must run on every render — early return goes after.
  const q = useChangeDetailQuery(changeId ?? undefined);
  const change = q.data;

  if (!changeId) return null;

  return (
    <div
      className="fixed inset-0 z-40 flex justify-end bg-black/60"
      onClick={onClose}
    >
      <div
        className="w-full max-w-3xl bg-[#0a0e14] border-l border-[#1e2738] flex flex-col h-full"
        onClick={(e) => e.stopPropagation()}
      >
        <header className="flex items-center justify-between px-5 py-4 border-b border-[#1e2738] flex-shrink-0">
          <div className="min-w-0">
            <div className="text-xs text-[#7a8599] font-mono truncate">
              {changeId}
            </div>
            <div className="text-sm font-semibold text-[#e2e8f0] truncate">
              {change?.path ?? '…'}
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded text-[#7a8599] hover:text-[#e2e8f0] hover:bg-[#1e2738]"
            aria-label="Close drawer"
          >
            <svg
              className="w-5 h-5"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        </header>

        <div className="flex-1 overflow-y-auto p-5 space-y-5">
          {q.isLoading && !change ? (
            <div className="space-y-3">
              <Skeleton className="h-6 w-1/3" />
              <Skeleton className="h-32" />
            </div>
          ) : !change ? (
            <div className="text-sm text-[#f87171]">
              Failed to load change request.
            </div>
          ) : (
            <>
              {/* Status row */}
              <div className="flex items-center gap-2 flex-wrap">
                <StatusBadge status={change.status} />
                <ProtectedBadge isProtected={change.is_protected} />
                <span className="text-xs text-[#7a8599] font-mono">
                  by {change.requestor}
                </span>
                <span className="text-xs text-[#7a8599] font-mono">
                  · {new Date(change.created_at).toLocaleString()}
                </span>
                {change.decided_by && (
                  <span className="text-xs text-[#7a8599] font-mono">
                    · decided by {change.decided_by}
                  </span>
                )}
              </div>

              {/* Reason */}
              <section>
                <h3 className="text-xs font-semibold text-[#7a8599] uppercase tracking-wider mb-2">
                  Reason
                </h3>
                <div className="text-sm text-[#cbd5e1] whitespace-pre-wrap">
                  {change.reason || '(no reason)'}
                </div>
              </section>

              {/* Decision metadata, if any */}
              {(change.decision_reason || change.decided_at) && (
                <section>
                  <h3 className="text-xs font-semibold text-[#7a8599] uppercase tracking-wider mb-2">
                    Decision
                  </h3>
                  <div className="text-xs text-[#cbd5e1] space-y-1">
                    {change.decided_at && (
                      <div>
                        <span className="text-[#7a8599]">at:</span>{' '}
                        <span className="font-mono">
                          {new Date(change.decided_at).toLocaleString()}
                        </span>
                      </div>
                    )}
                    {change.decision_reason && (
                      <div>
                        <span className="text-[#7a8599]">note:</span>{' '}
                        {change.decision_reason}
                      </div>
                    )}
                  </div>
                </section>
              )}

              {/* Application metadata, if any */}
              {(change.git_branch ||
                change.git_commit_sha ||
                change.pr_url ||
                change.applied_at) && (
                <section>
                  <h3 className="text-xs font-semibold text-[#7a8599] uppercase tracking-wider mb-2">
                    Apply
                  </h3>
                  <div className="text-xs space-y-1">
                    {change.applied_at && (
                      <div>
                        <span className="text-[#7a8599]">applied at:</span>{' '}
                        <span className="font-mono text-[#cbd5e1]">
                          {new Date(change.applied_at).toLocaleString()}
                        </span>
                      </div>
                    )}
                    {change.git_branch && (
                      <div>
                        <span className="text-[#7a8599]">branch:</span>{' '}
                        <code className="font-mono text-[#cbd5e1]">
                          {change.git_branch}
                        </code>
                      </div>
                    )}
                    {change.git_commit_sha && (
                      <div>
                        <span className="text-[#7a8599]">commit:</span>{' '}
                        <code className="font-mono text-[#cbd5e1]">
                          {change.git_commit_sha.slice(0, 12)}
                        </code>
                      </div>
                    )}
                    {change.pr_url && (
                      <div>
                        <span className="text-[#7a8599]">PR:</span>{' '}
                        <a
                          href={change.pr_url}
                          target="_blank"
                          rel="noreferrer"
                          className="text-[#60a5fa] hover:underline font-mono break-all"
                        >
                          {change.pr_url}
                        </a>
                      </div>
                    )}
                  </div>
                </section>
              )}

              {/* Rollback metadata, if any */}
              {change.rolled_back_at && (
                <section>
                  <h3 className="text-xs font-semibold text-[#a78bfa] uppercase tracking-wider mb-2">
                    Rolled back
                  </h3>
                  <div className="text-xs space-y-1">
                    <div>
                      <span className="text-[#7a8599]">at:</span>{' '}
                      <span className="font-mono text-[#cbd5e1]">
                        {new Date(change.rolled_back_at).toLocaleString()}
                      </span>
                    </div>
                    {change.rolled_back_by && (
                      <div>
                        <span className="text-[#7a8599]">by:</span>{' '}
                        <span className="font-mono text-[#cbd5e1]">
                          {change.rolled_back_by}
                        </span>
                      </div>
                    )}
                    {change.rollback_pr_url && (
                      <div>
                        <span className="text-[#7a8599]">revert PR:</span>{' '}
                        <a
                          href={change.rollback_pr_url}
                          target="_blank"
                          rel="noreferrer"
                          className="text-[#a78bfa] hover:underline font-mono break-all"
                        >
                          {change.rollback_pr_url}
                        </a>
                      </div>
                    )}
                  </div>
                </section>
              )}

              {/* Diff */}
              <section>
                <h3 className="text-xs font-semibold text-[#7a8599] uppercase tracking-wider mb-2">
                  Diff
                </h3>
                <DiffView diff={change.diff} />
              </section>

              {/* Actions */}
              <section className="pt-2 border-t border-[#1e2738]">
                <ActionButtons change={change} onClose={onClose} />
              </section>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export function ChangesPage() {
  const [statusFilter, setStatusFilter] = useState<ChangeStatus | 'all'>('all');
  const [activeId, setActiveId] = useState<string | null>(null);
  const listQ = useChangesListQuery(
    statusFilter === 'all' ? undefined : statusFilter,
  );

  const changes = listQ.data?.changes ?? [];

  // Counts per status for the row of stat cards. Server already filters
  // when `statusFilter !== 'all'`, so for accurate counts we'd want a
  // separate aggregated endpoint — for v1 just show counts of the current
  // (possibly filtered) view.
  const counts: Partial<Record<ChangeStatus, number>> = {};
  for (const c of changes) {
    counts[c.status] = (counts[c.status] ?? 0) + 1;
  }

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-xl font-semibold text-[#e2e8f0]">Change Requests</h1>
        <p className="text-sm text-[#7a8599] mt-1">
          Agent-proposed code modifications to restricted paths. Approving
          here hot-applies the change and opens an auto-PR against{' '}
          <code className="font-mono">main</code>. Operator merge is gate 2.
          TIER_IMMUTABLE files are refused at request time and cannot be
          overridden.
        </p>
      </div>

      {/* Filter tabs */}
      <div className="flex flex-wrap gap-1.5">
        {STATUS_FILTERS.map((s) => (
          <button
            key={s}
            onClick={() => setStatusFilter(s)}
            className={`px-3 py-1 rounded-full text-xs font-medium transition-colors ${
              statusFilter === s
                ? 'bg-[#60a5fa]/15 text-[#60a5fa] border border-[#60a5fa]/30'
                : 'bg-[#111820] text-[#7a8599] border border-[#1e2738] hover:bg-[#1e2738]'
            }`}
          >
            {s}
          </button>
        ))}
      </div>

      {/* List */}
      {listQ.isLoading ? (
        <div className="space-y-2">
          <Skeleton className="h-16" />
          <Skeleton className="h-16" />
          <Skeleton className="h-16" />
        </div>
      ) : changes.length === 0 ? (
        <div className="p-8 text-center text-sm text-[#7a8599] border border-[#1e2738] rounded-lg bg-[#111820]">
          {statusFilter === 'all' ? (
            <>
              No change requests yet. Agents that have the{' '}
              <code className="font-mono text-[#fbbf24]">
                request_restricted_write
              </code>{' '}
              tool can propose code changes — they'll appear here when
              submitted.
            </>
          ) : (
            <>
              No change requests with status{' '}
              <code className="font-mono text-[#fbbf24]">{statusFilter}</code>.
            </>
          )}
        </div>
      ) : (
        <div className="space-y-2">
          {changes.map((c) => (
            <ChangeRow
              key={c.id}
              change={c}
              isActive={activeId === c.id}
              onClick={() => setActiveId(c.id)}
            />
          ))}
        </div>
      )}

      <ChangeDetailDrawer
        changeId={activeId}
        onClose={() => setActiveId(null)}
      />
    </div>
  );
}
