// Inquiries operator surface — read-only.
//
// Lists the weekly philosophical inquiry essays produced by the
// app/subia/inquiry/ subsystem. Each essay is a markdown page in
// wiki/self/inquiries/; this surface exposes them through
// /api/cp/inquiries/. Also shows the operator-curated question
// list with answer state per question.
//
// New questions land via change_requests, not from this page —
// edits to wiki/self/inquiry_questions.md flow through the
// standard human-gated review path.

import { useState } from 'react';
import { Skeleton } from './ui/Skeleton';
import {
  useInquiriesListQuery,
  useInquiryDetailQuery,
  useInquiryQuestionsQuery,
} from '../api/inquiries';

function relTime(iso: string): string {
  if (!iso) return '';
  const ms = Date.now() - new Date(iso).getTime();
  const s = Math.max(0, Math.floor(ms / 1000));
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

export function InquiriesPage() {
  const [openFilename, setOpenFilename] = useState<string | null>(null);

  const list = useInquiriesListQuery();
  const questions = useInquiryQuestionsQuery();

  return (
    <div className="flex h-full flex-col gap-4 p-6">
      <header className="flex flex-wrap items-baseline gap-4">
        <h1 className="text-xl font-semibold tracking-tight text-[#e6edf7]">
          Inquiries
        </h1>
        <p className="text-sm text-[#7a8599]">
          Weekly philosophical inquiry essays from{' '}
          <code className="rounded bg-[#1f2937] px-1 py-[1px] text-xs">
            app/subia/inquiry/
          </code>
          . Observational; does not feed reward, fitness, evaluation,
          or <code className="rounded bg-[#1f2937] px-1 py-[1px] text-xs">current_goals</code>.
        </p>
      </header>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[2fr_1fr] flex-1 overflow-hidden">
        <section className="flex flex-col overflow-hidden rounded border border-[#1f2937] bg-[#0c1019]">
          <div className="border-b border-[#1f2937] px-4 py-2 text-xs font-mono uppercase tracking-wide text-[#7a8599]">
            Recent essays
          </div>
          {list.isLoading ? (
            <div className="space-y-2 p-4">
              <Skeleton className="h-16" />
              <Skeleton className="h-16" />
              <Skeleton className="h-16" />
            </div>
          ) : list.error ? (
            <div className="p-4 text-sm text-[#f87171]">
              Failed to load: {(list.error as Error).message}
            </div>
          ) : !list.data || list.data.count === 0 ? (
            <div className="flex h-full items-center justify-center p-8 text-center text-sm text-[#7a8599]">
              No inquiries written yet. The scheduler runs once per
              week (default — overridable via{' '}
              <code className="mx-1 rounded bg-[#1f2937] px-1 py-[1px] text-xs">
                INQUIRY_MIN_INTERVAL_DAYS
              </code>
              ).
            </div>
          ) : (
            <ul className="divide-y divide-[#1f2937] overflow-y-auto">
              {list.data.inquiries.map((entry) => (
                <li
                  key={entry.filename}
                  onClick={() => setOpenFilename(entry.filename)}
                  className="cursor-pointer px-4 py-3 transition hover:bg-[#111827]"
                >
                  <div className="flex flex-wrap items-baseline gap-3">
                    <span className="font-mono text-xs text-[#a78bfa]">
                      {entry.date ?? '—'}
                    </span>
                    <span className="text-sm font-medium text-[#e6edf7]">
                      {entry.question_text || entry.slug || entry.filename}
                    </span>
                    <span className="ml-auto text-xs text-[#7a8599]">
                      {relTime(entry.modified_at)} ·{' '}
                      {(entry.size_bytes / 1024).toFixed(1)} KB
                    </span>
                  </div>
                  {entry.preview && (
                    <p className="mt-1 truncate text-xs text-[#9ca3af]">
                      {entry.preview}
                    </p>
                  )}
                </li>
              ))}
            </ul>
          )}
        </section>

        <section className="flex flex-col overflow-hidden rounded border border-[#1f2937] bg-[#0c1019]">
          <div className="border-b border-[#1f2937] px-4 py-2 text-xs font-mono uppercase tracking-wide text-[#7a8599]">
            Curated questions
          </div>
          {questions.isLoading ? (
            <div className="space-y-2 p-4">
              <Skeleton className="h-12" />
              <Skeleton className="h-12" />
            </div>
          ) : questions.error ? (
            <div className="p-4 text-sm text-[#f87171]">
              {(questions.error as Error).message}
            </div>
          ) : !questions.data || questions.data.count === 0 ? (
            <div className="p-4 text-sm text-[#7a8599]">
              No questions yet. Curate the list at{' '}
              <code className="rounded bg-[#1f2937] px-1 py-[1px] text-xs">
                wiki/self/inquiry_questions.md
              </code>
              .
            </div>
          ) : (
            <ul className="divide-y divide-[#1f2937] overflow-y-auto">
              {questions.data.questions.map((q) => (
                <li key={q.slug} className="px-4 py-3">
                  <p className="text-sm text-[#e6edf7]">{q.text}</p>
                  <p className="mt-1 text-xs text-[#7a8599]">
                    last answered:{' '}
                    {q.most_recent_answer_date ?? (
                      <span className="text-[#fbbf24]">never</span>
                    )}
                  </p>
                </li>
              ))}
            </ul>
          )}
        </section>
      </div>

      {openFilename && (
        <InquiryDrawer
          filename={openFilename}
          onClose={() => setOpenFilename(null)}
        />
      )}
    </div>
  );
}

function InquiryDrawer({
  filename,
  onClose,
}: {
  filename: string;
  onClose: () => void;
}) {
  const detail = useInquiryDetailQuery(filename);

  return (
    <div
      className="fixed inset-0 z-30 flex justify-end bg-black/40"
      onClick={onClose}
    >
      <aside
        className="flex h-full w-full max-w-3xl flex-col gap-3 overflow-y-auto border-l border-[#1f2937] bg-[#0c1019] p-6"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-baseline gap-3">
          <h2 className="text-lg font-semibold text-[#e6edf7]">
            {detail.data?.question_text ?? filename}
          </h2>
          <button
            type="button"
            className="ml-auto text-[#7a8599] hover:text-[#e6edf7]"
            onClick={onClose}
          >
            close
          </button>
        </div>
        {detail.data?.date && (
          <div className="text-xs font-mono text-[#7a8599]">
            {detail.data.date} · {filename}
          </div>
        )}
        {detail.isLoading ? (
          <Skeleton className="h-96" />
        ) : detail.error ? (
          <p className="text-sm text-[#f87171]">
            {(detail.error as Error).message}
          </p>
        ) : (
          <pre className="overflow-x-auto whitespace-pre-wrap rounded bg-[#000] p-4 text-xs text-[#cbd5e1]">
            {detail.data?.body ?? ''}
          </pre>
        )}
      </aside>
    </div>
  );
}
