// Long-horizon threads operator surface.
// Lists open threads (cross-day questions); drawer shows sub-questions,
// blockers, notes, related-CR/inquiry links, plus per-status actions.

import { useState } from 'react';
import { Skeleton } from './ui/Skeleton';
import {
  useAddBlockerMutation,
  useAddNoteMutation,
  useAddSubQuestionMutation,
  useClearBlockersMutation,
  useCreateThreadMutation,
  useResolveSqMutation,
  useThreadDetailQuery,
  useThreadsListQuery,
  useTransitionThreadMutation,
} from '../api/threads';
import type { Thread, ThreadStatus } from '../types/threads';

const STATUS_BADGE: Record<
  ThreadStatus,
  { bg: string; fg: string; border: string; label: string }
> = {
  open: {
    bg: 'bg-[#fbbf24]/15', fg: 'text-[#fbbf24]',
    border: 'border-[#fbbf24]/30', label: 'OPEN',
  },
  in_progress: {
    bg: 'bg-[#60a5fa]/15', fg: 'text-[#60a5fa]',
    border: 'border-[#60a5fa]/30', label: 'IN PROGRESS',
  },
  blocked: {
    bg: 'bg-[#f87171]/15', fg: 'text-[#f87171]',
    border: 'border-[#f87171]/30', label: 'BLOCKED',
  },
  resolved: {
    bg: 'bg-[#34d399]/15', fg: 'text-[#34d399]',
    border: 'border-[#34d399]/30', label: 'RESOLVED',
  },
  abandoned: {
    bg: 'bg-[#7a8599]/15', fg: 'text-[#7a8599]',
    border: 'border-[#7a8599]/30', label: 'ABANDONED',
  },
};

function StatusBadge({ status }: { status: ThreadStatus }) {
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

export function ThreadsPage() {
  const [openOnly, setOpenOnly] = useState(true);
  const [openId, setOpenId] = useState<string | null>(null);
  const [showCreate, setShowCreate] = useState(false);
  const list = useThreadsListQuery(openOnly);

  return (
    <div className="flex h-full flex-col gap-4 p-6">
      <header className="flex flex-wrap items-baseline gap-4">
        <h1 className="text-xl font-semibold tracking-tight text-[#e6edf7]">
          Threads
        </h1>
        <p className="text-sm text-[#7a8599]">
          Long-horizon questions. A thread carries cross-day, cross-crew
          context for one line of inquiry — what's resolved, what's
          blocking, what's being looked at next.
        </p>
        <button
          type="button"
          onClick={() => setShowCreate(true)}
          className="ml-auto rounded border border-[#34d399]/40 bg-[#34d399]/10 px-3 py-1 text-xs text-[#34d399] hover:bg-[#34d399]/20"
        >
          + New thread
        </button>
      </header>

      <div className="flex gap-2">
        <button
          type="button"
          onClick={() => setOpenOnly(true)}
          className={`rounded border px-2 py-1 text-xs font-mono uppercase tracking-wide transition ${
            openOnly
              ? 'border-[#60a5fa] bg-[#60a5fa]/10 text-[#e6edf7]'
              : 'border-[#1f2937] text-[#7a8599] hover:border-[#374151]'
          }`}
        >
          OPEN
        </button>
        <button
          type="button"
          onClick={() => setOpenOnly(false)}
          className={`rounded border px-2 py-1 text-xs font-mono uppercase tracking-wide transition ${
            !openOnly
              ? 'border-[#60a5fa] bg-[#60a5fa]/10 text-[#e6edf7]'
              : 'border-[#1f2937] text-[#7a8599] hover:border-[#374151]'
          }`}
        >
          ALL
        </button>
      </div>

      <div className="flex-1 overflow-hidden rounded border border-[#1f2937] bg-[#0c1019]">
        {list.isLoading ? (
          <div className="space-y-2 p-4">
            <Skeleton className="h-12" />
            <Skeleton className="h-12" />
          </div>
        ) : list.error ? (
          <div className="p-4 text-sm text-[#f87171]">
            Failed to load: {(list.error as Error).message}
          </div>
        ) : !list.data || list.data.count === 0 ? (
          <div className="flex h-full items-center justify-center p-8 text-center text-sm text-[#7a8599]">
            No {openOnly ? 'open' : ''} threads. Click &quot;+ New thread&quot; to start one.
          </div>
        ) : (
          <ul className="divide-y divide-[#1f2937] overflow-y-auto">
            {list.data.threads.map((t) => (
              <Row key={t.id} thread={t} onClick={() => setOpenId(t.id)} />
            ))}
          </ul>
        )}
      </div>

      {openId && (
        <Drawer threadId={openId} onClose={() => setOpenId(null)} />
      )}
      {showCreate && (
        <CreateModal onClose={() => setShowCreate(false)} />
      )}
    </div>
  );
}

function Row({ thread, onClick }: { thread: Thread; onClick: () => void }) {
  return (
    <li
      onClick={onClick}
      className="cursor-pointer px-4 py-3 transition hover:bg-[#111827]"
    >
      <div className="flex flex-wrap items-baseline gap-3">
        <StatusBadge status={thread.status} />
        <span className="text-sm font-medium text-[#e6edf7]">{thread.title}</span>
        <span className="ml-auto text-xs text-[#7a8599]">
          {relTime(thread.last_touched_at || thread.created_at)}
        </span>
      </div>
      <div className="mt-1 flex flex-wrap gap-3 text-xs text-[#7a8599]">
        <span>{thread.open_subquestion_count} open</span>
        <span>·</span>
        <span>{thread.resolved_subquestion_count} resolved</span>
        {thread.blockers.length > 0 && (
          <>
            <span>·</span>
            <span className="text-[#f87171]">
              {thread.blockers.length} blocker(s)
            </span>
          </>
        )}
      </div>
    </li>
  );
}

function CreateModal({ onClose }: { onClose: () => void }) {
  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const create = useCreateThreadMutation();

  return (
    <div
      className="fixed inset-0 z-30 flex items-center justify-center bg-black/40 p-4"
      onClick={onClose}
    >
      <div
        className="w-full max-w-lg rounded border border-[#1f2937] bg-[#0c1019] p-6"
        onClick={(e) => e.stopPropagation()}
      >
        <h2 className="mb-3 text-lg font-semibold text-[#e6edf7]">
          New thread
        </h2>
        <input
          type="text"
          placeholder="Title (one line)"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          className="mb-2 w-full rounded border border-[#1f2937] bg-[#000] px-3 py-2 text-sm text-[#e6edf7]"
        />
        <textarea
          placeholder="Description (optional)"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          rows={3}
          className="mb-3 w-full rounded border border-[#1f2937] bg-[#000] px-3 py-2 text-sm text-[#e6edf7]"
        />
        <div className="flex gap-2">
          <button
            type="button"
            disabled={!title.trim() || create.isPending}
            onClick={() =>
              create.mutate(
                { title, description },
                { onSuccess: onClose },
              )
            }
            className="rounded border border-[#34d399]/40 bg-[#34d399]/10 px-3 py-1 text-xs text-[#34d399] hover:bg-[#34d399]/20 disabled:opacity-50"
          >
            {create.isPending ? 'creating...' : 'Create'}
          </button>
          <button
            type="button"
            onClick={onClose}
            className="rounded border border-[#1f2937] px-3 py-1 text-xs text-[#7a8599] hover:text-[#e6edf7]"
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}

function Drawer({ threadId, onClose }: { threadId: string; onClose: () => void }) {
  const detail = useThreadDetailQuery(threadId);
  const t = detail.data;

  const addSq = useAddSubQuestionMutation();
  const resolveSq = useResolveSqMutation();
  const addBlocker = useAddBlockerMutation();
  const clearBlockers = useClearBlockersMutation();
  const addNote = useAddNoteMutation();
  const transition = useTransitionThreadMutation();

  const [sqText, setSqText] = useState('');
  const [blockerText, setBlockerText] = useState('');
  const [noteText, setNoteText] = useState('');
  const [resolveSummary, setResolveSummary] = useState('');
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
          {t ? (
            <>
              <StatusBadge status={t.status} />
              <h2 className="text-lg font-semibold">{t.title}</h2>
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

        {t && (
          <>
            {t.description && (
              <p className="whitespace-pre-wrap text-[#cbd5e1]">{t.description}</p>
            )}
            <div className="text-xs text-[#7a8599]">
              created: {new Date(t.created_at).toLocaleString()} · last touched: {relTime(t.last_touched_at || t.created_at)}
            </div>

            <section>
              <h3 className="mb-1 text-xs font-mono uppercase tracking-wide text-[#7a8599]">
                Sub-questions ({t.open_subquestion_count} open / {t.resolved_subquestion_count} resolved)
              </h3>
              {t.sub_questions.length === 0 ? (
                <p className="text-xs text-[#7a8599]">(none)</p>
              ) : (
                <ul className="space-y-1">
                  {t.sub_questions.map((sq) => (
                    <li
                      key={sq.id}
                      className="flex flex-wrap gap-2 text-xs"
                    >
                      <span className={sq.resolved ? 'text-[#34d399]' : 'text-[#fbbf24]'}>
                        {sq.resolved ? '✓' : '·'}
                      </span>
                      <span className="text-[#cbd5e1]">{sq.text}</span>
                      {sq.resolved && sq.resolution && (
                        <span className="text-[#7a8599]">— {sq.resolution}</span>
                      )}
                      {!sq.resolved && !t.is_terminal && (
                        <button
                          type="button"
                          onClick={() =>
                            resolveSq.mutate({
                              id: t.id, subquestion_id: sq.id,
                              resolution: prompt('Resolution (optional):') ?? '',
                            })
                          }
                          className="ml-auto text-[10px] text-[#34d399] hover:underline"
                        >
                          resolve
                        </button>
                      )}
                    </li>
                  ))}
                </ul>
              )}
              {!t.is_terminal && (
                <div className="mt-2 flex gap-2">
                  <input
                    type="text"
                    placeholder="add sub-question"
                    value={sqText}
                    onChange={(e) => setSqText(e.target.value)}
                    className="flex-1 rounded border border-[#1f2937] bg-[#000] px-2 py-1 text-xs"
                  />
                  <button
                    type="button"
                    disabled={!sqText.trim() || addSq.isPending}
                    onClick={() =>
                      addSq.mutate(
                        { id: t.id, text: sqText },
                        { onSuccess: () => setSqText('') },
                      )
                    }
                    className="rounded border border-[#60a5fa]/40 bg-[#60a5fa]/10 px-3 py-1 text-xs text-[#60a5fa]"
                  >
                    add
                  </button>
                </div>
              )}
            </section>

            <section>
              <h3 className="mb-1 text-xs font-mono uppercase tracking-wide text-[#7a8599]">
                Blockers
              </h3>
              {t.blockers.length === 0 ? (
                <p className="text-xs text-[#7a8599]">(none)</p>
              ) : (
                <ul className="space-y-1 text-xs text-[#f87171]">
                  {t.blockers.map((b, i) => (
                    <li key={i}>· {b}</li>
                  ))}
                </ul>
              )}
              {!t.is_terminal && (
                <div className="mt-2 flex gap-2">
                  <input
                    type="text"
                    placeholder="add blocker"
                    value={blockerText}
                    onChange={(e) => setBlockerText(e.target.value)}
                    className="flex-1 rounded border border-[#1f2937] bg-[#000] px-2 py-1 text-xs"
                  />
                  <button
                    type="button"
                    disabled={!blockerText.trim() || addBlocker.isPending}
                    onClick={() =>
                      addBlocker.mutate(
                        { id: t.id, text: blockerText },
                        { onSuccess: () => setBlockerText('') },
                      )
                    }
                    className="rounded border border-[#f87171]/40 bg-[#f87171]/10 px-3 py-1 text-xs text-[#f87171]"
                  >
                    add
                  </button>
                  {t.blockers.length > 0 && (
                    <button
                      type="button"
                      onClick={() => clearBlockers.mutate({ id: t.id })}
                      className="rounded border border-[#34d399]/40 bg-[#34d399]/10 px-3 py-1 text-xs text-[#34d399]"
                    >
                      clear all
                    </button>
                  )}
                </div>
              )}
            </section>

            <section>
              <h3 className="mb-1 text-xs font-mono uppercase tracking-wide text-[#7a8599]">
                Notes
              </h3>
              {t.notes.length === 0 ? (
                <p className="text-xs text-[#7a8599]">(none)</p>
              ) : (
                <ul className="space-y-1 text-xs">
                  {t.notes.map((n, i) => (
                    <li key={i} className="text-[#cbd5e1]">· {n}</li>
                  ))}
                </ul>
              )}
              {!t.is_terminal && (
                <div className="mt-2 flex gap-2">
                  <input
                    type="text"
                    placeholder="add note"
                    value={noteText}
                    onChange={(e) => setNoteText(e.target.value)}
                    className="flex-1 rounded border border-[#1f2937] bg-[#000] px-2 py-1 text-xs"
                  />
                  <button
                    type="button"
                    disabled={!noteText.trim() || addNote.isPending}
                    onClick={() =>
                      addNote.mutate(
                        { id: t.id, text: noteText },
                        { onSuccess: () => setNoteText('') },
                      )
                    }
                    className="rounded border border-[#60a5fa]/40 bg-[#60a5fa]/10 px-3 py-1 text-xs text-[#60a5fa]"
                  >
                    add
                  </button>
                </div>
              )}
            </section>

            {(t.related_crew_task_ids.length > 0 ||
              t.related_inquiry_slugs.length > 0) && (
              <section>
                <h3 className="mb-1 text-xs font-mono uppercase tracking-wide text-[#7a8599]">
                  Linked
                </h3>
                {t.related_crew_task_ids.length > 0 && (
                  <p className="text-xs">
                    Crew tasks: {t.related_crew_task_ids.join(', ')}
                  </p>
                )}
                {t.related_inquiry_slugs.length > 0 && (
                  <p className="text-xs">
                    Inquiries: {t.related_inquiry_slugs.join(', ')}
                  </p>
                )}
              </section>
            )}

            {!t.is_terminal && (
              <section className="mt-2 border-t border-[#1f2937] pt-3">
                <h3 className="mb-2 text-xs font-mono uppercase tracking-wide text-[#7a8599]">
                  Transitions
                </h3>
                <div className="flex flex-wrap gap-2">
                  <input
                    type="text"
                    placeholder="resolution summary"
                    value={resolveSummary}
                    onChange={(e) => setResolveSummary(e.target.value)}
                    className="rounded border border-[#1f2937] bg-[#000] px-2 py-1 text-xs"
                  />
                  <button
                    type="button"
                    onClick={() =>
                      transition.mutate({
                        id: t.id, transition: 'resolved',
                        summary: resolveSummary,
                      })
                    }
                    disabled={transition.isPending}
                    className="rounded border border-[#34d399]/40 bg-[#34d399]/10 px-3 py-1 text-xs text-[#34d399]"
                  >
                    Resolve
                  </button>
                  <input
                    type="text"
                    placeholder="abandon reason"
                    value={abandonReason}
                    onChange={(e) => setAbandonReason(e.target.value)}
                    className="rounded border border-[#1f2937] bg-[#000] px-2 py-1 text-xs"
                  />
                  <button
                    type="button"
                    disabled={!abandonReason.trim() || transition.isPending}
                    onClick={() =>
                      transition.mutate({
                        id: t.id, transition: 'abandoned',
                        reason: abandonReason,
                      })
                    }
                    className="rounded border border-[#7a8599]/40 bg-[#7a8599]/10 px-3 py-1 text-xs text-[#7a8599]"
                  >
                    Abandon
                  </button>
                </div>
              </section>
            )}
          </>
        )}
      </aside>
    </div>
  );
}
