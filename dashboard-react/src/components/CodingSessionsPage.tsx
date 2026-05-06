// Phase 5.4-f — read-only operator view of agent coding sessions.
//
// Lists agent worktrees with status filter; opens a drawer with
// per-session detail (purpose, base sha, files touched, terminal
// metadata, submit_results if any). Polling at 8 s; no actions —
// session lifecycle is owned by the agent + reconciler.
//
// Operator decisions about a session's diff happen in the change-
// request UI (`/cp/changes`) — that's where the actionable surface
// lives.

import { useState } from 'react';
import { Link } from 'react-router-dom';
import { Skeleton } from './ui/Skeleton';
import {
  useCodingSessionDetailQuery,
  useCodingSessionsListQuery,
} from '../api/coding_sessions';
import type {
  CodingSession,
  CodingSessionStatus,
} from '../types/coding_sessions';

const STATUS_FILTERS: (CodingSessionStatus | 'all')[] = [
  'all',
  'active',
  'submitted',
  'discarded',
  'expired',
  'failed',
];

const STATUS_BADGE: Record<
  CodingSessionStatus,
  { bg: string; fg: string; border: string; label: string }
> = {
  active: {
    bg: 'bg-[#22d3ee]/15',
    fg: 'text-[#22d3ee]',
    border: 'border-[#22d3ee]/30',
    label: 'ACTIVE',
  },
  submitted: {
    bg: 'bg-[#34d399]/15',
    fg: 'text-[#34d399]',
    border: 'border-[#34d399]/30',
    label: 'SUBMITTED',
  },
  discarded: {
    bg: 'bg-[#7a8599]/15',
    fg: 'text-[#7a8599]',
    border: 'border-[#7a8599]/30',
    label: 'DISCARDED',
  },
  expired: {
    bg: 'bg-[#fbbf24]/15',
    fg: 'text-[#fbbf24]',
    border: 'border-[#fbbf24]/30',
    label: 'EXPIRED',
  },
  failed: {
    bg: 'bg-[#f87171]/15',
    fg: 'text-[#f87171]',
    border: 'border-[#f87171]/40',
    label: 'FAILED',
  },
};

function StatusBadge({ status }: { status: CodingSessionStatus }) {
  const s = STATUS_BADGE[status];
  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-medium border ${s.bg} ${s.fg} ${s.border}`}
    >
      {s.label}
    </span>
  );
}

function formatRelative(iso: string): string {
  if (!iso) return '—';
  const t = Date.parse(iso);
  if (isNaN(t)) return iso;
  const delta = (Date.now() - t) / 1000;
  if (delta < 60) return `${Math.floor(delta)}s ago`;
  if (delta < 3600) return `${Math.floor(delta / 60)}m ago`;
  if (delta < 86400) return `${Math.floor(delta / 3600)}h ago`;
  return new Date(t).toLocaleString();
}

function formatBytes(b: number): string {
  if (b < 1024) return `${b} B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KiB`;
  return `${(b / 1024 / 1024).toFixed(1)} MiB`;
}

function CodingSessionRow({
  session,
  isActive,
  onClick,
}: {
  session: CodingSession;
  isActive: boolean;
  onClick: () => void;
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
            <StatusBadge status={session.status} />
            <span className="text-[10px] font-mono text-[#7a8599]">
              {session.agent_id}
            </span>
            <span className="text-[10px] font-mono text-[#7a8599]">
              base: {session.base} ({session.base_sha.slice(0, 8)})
            </span>
          </div>
          {session.purpose && (
            <div className="text-sm text-[#cbd5e1] mt-1 line-clamp-2">
              {session.purpose}
            </div>
          )}
          <div className="text-[10px] text-[#7a8599] font-mono mt-2">
            {session.files_touched.length} files · {session.run_count} runs ·{' '}
            {formatBytes(session.bytes_written)}
          </div>
          {session.terminated_reason && (
            <div className="text-[10px] text-[#fbbf24] mt-1">
              {session.terminated_reason}
            </div>
          )}
        </div>
        <div className="text-[10px] text-[#7a8599] font-mono flex-shrink-0 text-right">
          <div>created {formatRelative(session.created_at)}</div>
          <div className="opacity-60 mt-0.5">
            activity {formatRelative(session.last_activity_at)}
          </div>
          <div className="opacity-60 mt-0.5">{session.id.slice(0, 8)}…</div>
        </div>
      </div>
    </button>
  );
}

function SubmitResultRow({
  path,
  change_request_id,
  status,
  refusal_reason,
}: {
  path: string;
  change_request_id: string | null;
  status: string;
  refusal_reason?: string | null;
}) {
  const refused = refusal_reason !== null && refusal_reason !== undefined;
  return (
    <div className="flex items-start gap-3 p-2 rounded-md border border-[#1e2738] bg-[#0a0e14]">
      <code className="text-xs text-[#cbd5e1] flex-1 truncate">{path}</code>
      <div className="flex flex-col items-end text-[10px] font-mono flex-shrink-0">
        <span className={refused ? 'text-[#f87171]' : 'text-[#34d399]'}>
          {status}
        </span>
        {change_request_id && (
          <Link
            to={`/changes`}
            className="text-[#60a5fa] hover:underline"
            title={`Change request ${change_request_id}`}
          >
            {change_request_id.slice(0, 12)}
          </Link>
        )}
        {refused && refusal_reason && (
          <span
            className="text-[#7a8599] truncate max-w-xs"
            title={refusal_reason}
          >
            {refusal_reason.slice(0, 60)}
            {refusal_reason.length > 60 && '…'}
          </span>
        )}
      </div>
    </div>
  );
}

function CodingSessionDetailDrawer({
  sessionId,
  onClose,
}: {
  sessionId: string | null;
  onClose: () => void;
}) {
  const q = useCodingSessionDetailQuery(sessionId ?? undefined);
  const session = q.data;

  if (!sessionId) return null;

  return (
    <div
      className="fixed inset-0 z-40 flex justify-end bg-black/60"
      onClick={onClose}
    >
      <div
        className="w-full max-w-2xl bg-[#0a0e14] border-l border-[#1e2738] flex flex-col h-full"
        onClick={(e) => e.stopPropagation()}
      >
        <header className="flex items-center justify-between px-5 py-4 border-b border-[#1e2738] flex-shrink-0">
          <div className="min-w-0">
            <div className="text-xs text-[#7a8599] font-mono truncate">
              {sessionId}
            </div>
            <div className="text-sm font-semibold text-[#e2e8f0] truncate">
              {session?.purpose ?? '…'}
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
          {q.isLoading && !session ? (
            <div className="space-y-3">
              <Skeleton className="h-6 w-1/3" />
              <Skeleton className="h-32" />
            </div>
          ) : !session ? (
            <div className="text-sm text-[#f87171]">
              Failed to load session.
            </div>
          ) : (
            <>
              {/* Status row */}
              <div className="flex items-center gap-2 flex-wrap">
                <StatusBadge status={session.status} />
                <span className="text-xs text-[#7a8599] font-mono">
                  {session.agent_id}
                </span>
                <span className="text-xs text-[#7a8599] font-mono">
                  · created {formatRelative(session.created_at)}
                </span>
                <span className="text-xs text-[#7a8599] font-mono">
                  · last activity {formatRelative(session.last_activity_at)}
                </span>
              </div>

              {/* Base + worktree */}
              <section>
                <h3 className="text-xs font-semibold text-[#7a8599] uppercase tracking-wider mb-2">
                  Worktree
                </h3>
                <div className="text-xs space-y-1 font-mono">
                  <div>
                    <span className="text-[#7a8599]">base:</span>{' '}
                    <span className="text-[#cbd5e1]">{session.base}</span>{' '}
                    <span className="text-[#7a8599]">
                      ({session.base_sha.slice(0, 12)})
                    </span>
                  </div>
                  <div>
                    <span className="text-[#7a8599]">path:</span>{' '}
                    <code className="text-[#cbd5e1] break-all">
                      {session.worktree_path}
                    </code>
                  </div>
                  <div>
                    <span className="text-[#7a8599]">expires:</span>{' '}
                    <span className="text-[#cbd5e1]">{session.expires_at}</span>
                  </div>
                </div>
              </section>

              {/* Activity counters */}
              <section>
                <h3 className="text-xs font-semibold text-[#7a8599] uppercase tracking-wider mb-2">
                  Activity
                </h3>
                <div className="text-xs grid grid-cols-3 gap-2">
                  <div>
                    <div className="text-[#7a8599] text-[10px]">FILES</div>
                    <div className="text-[#e2e8f0] text-lg font-semibold">
                      {session.files_touched.length}
                    </div>
                  </div>
                  <div>
                    <div className="text-[#7a8599] text-[10px]">RUNS</div>
                    <div className="text-[#e2e8f0] text-lg font-semibold">
                      {session.run_count}
                    </div>
                  </div>
                  <div>
                    <div className="text-[#7a8599] text-[10px]">BYTES</div>
                    <div className="text-[#e2e8f0] text-lg font-semibold">
                      {formatBytes(session.bytes_written)}
                    </div>
                  </div>
                </div>
              </section>

              {/* Files touched */}
              {session.files_touched.length > 0 && (
                <section>
                  <h3 className="text-xs font-semibold text-[#7a8599] uppercase tracking-wider mb-2">
                    Files touched
                  </h3>
                  <ul className="text-xs space-y-1 font-mono">
                    {session.files_touched.map((p) => (
                      <li key={p} className="text-[#cbd5e1] truncate">
                        {p}
                      </li>
                    ))}
                  </ul>
                </section>
              )}

              {/* Terminal metadata */}
              {session.is_terminal && (
                <section>
                  <h3 className="text-xs font-semibold text-[#7a8599] uppercase tracking-wider mb-2">
                    Termination
                  </h3>
                  <div className="text-xs space-y-1">
                    <div>
                      <span className="text-[#7a8599]">at:</span>{' '}
                      <span className="font-mono text-[#cbd5e1]">
                        {session.terminated_at}
                      </span>
                    </div>
                    {session.terminated_reason && (
                      <div>
                        <span className="text-[#7a8599]">reason:</span>{' '}
                        <span className="text-[#cbd5e1]">
                          {session.terminated_reason}
                        </span>
                      </div>
                    )}
                  </div>
                </section>
              )}

              {/* Submit results, if any */}
              {session.submit_results && session.submit_results.length > 0 && (
                <section>
                  <h3 className="text-xs font-semibold text-[#7a8599] uppercase tracking-wider mb-2">
                    Submission ({session.submit_results.length} files)
                  </h3>
                  <div className="space-y-2">
                    {session.submit_results.map((r) => (
                      <SubmitResultRow key={r.path} {...r} />
                    ))}
                  </div>
                  <div className="text-xs text-[#7a8599] mt-3">
                    Operator approve / reject the resulting change requests at{' '}
                    <Link
                      to="/changes"
                      className="text-[#60a5fa] hover:underline"
                    >
                      /cp/changes
                    </Link>
                    .
                  </div>
                </section>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export function CodingSessionsPage() {
  const [statusFilter, setStatusFilter] = useState<
    CodingSessionStatus | 'all'
  >('all');
  const [activeId, setActiveId] = useState<string | null>(null);
  const listQ = useCodingSessionsListQuery(
    statusFilter === 'all' ? undefined : statusFilter,
  );

  const sessions = listQ.data?.sessions ?? [];

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-xl font-semibold text-[#e2e8f0]">
          Coding Sessions
        </h1>
        <p className="text-sm text-[#7a8599] mt-1">
          Agent worktrees — read-only view. Lifecycle is owned by the
          agent + reconciler; submission decisions happen in the{' '}
          <Link to="/changes" className="text-[#60a5fa] hover:underline">
            change-request UI
          </Link>
          .
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
          <Skeleton className="h-20" />
          <Skeleton className="h-20" />
        </div>
      ) : sessions.length === 0 ? (
        <div className="p-8 text-center text-sm text-[#7a8599] border border-[#1e2738] rounded-lg bg-[#111820]">
          {statusFilter === 'all' ? (
            <>
              No coding sessions yet. Agents that have the{' '}
              <code className="font-mono text-[#fbbf24]">
                coding_session_*
              </code>{' '}
              tools can start one — they'll appear here.
            </>
          ) : (
            <>
              No sessions with status{' '}
              <code className="font-mono text-[#fbbf24]">{statusFilter}</code>.
            </>
          )}
        </div>
      ) : (
        <div className="space-y-2">
          {sessions.map((s) => (
            <CodingSessionRow
              key={s.id}
              session={s}
              isActive={activeId === s.id}
              onClick={() => setActiveId(s.id)}
            />
          ))}
        </div>
      )}

      <CodingSessionDetailDrawer
        sessionId={activeId}
        onClose={() => setActiveId(null)}
      />
    </div>
  );
}
