// Unified proposals dashboard.
// Aggregates the three producer outputs:
//   - capability_gap_analyzer (docs/proposed_capabilities/)
//   - library_radar          (docs/proposed_libraries/)
//   - recipe_consolidation   (workspace/training/recipe_retirement_proposals.jsonl)
// All three converge on operator review; this page is the unified surface.

import { useState } from 'react';
import { Skeleton } from './ui/Skeleton';
import {
  useProposalDetailQuery,
  useProposalsListQuery,
} from '../api/proposals';
import type { ProposalKind, ProposalSummary } from '../types/proposals';

const KIND_LABEL: Record<ProposalKind | 'all', string> = {
  all: 'ALL',
  capability: 'CAPABILITY GAPS',
  library: 'LIBRARY ADOPTION',
  recipe: 'RECIPE RETIREMENT',
};

const KIND_COLOR: Record<ProposalKind, string> = {
  capability: 'text-[#a78bfa]',
  library: 'text-[#60a5fa]',
  recipe: 'text-[#fbbf24]',
};

function relTime(iso: string): string {
  if (!iso) return '';
  const ms = Date.now() - new Date(iso).getTime();
  const s = Math.max(0, Math.floor(ms / 1000));
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

export function ProposalsPage() {
  const [filter, setFilter] = useState<ProposalKind | 'all'>('all');
  const [open, setOpen] = useState<{ kind: ProposalKind; name: string } | null>(null);
  const list = useProposalsListQuery(filter === 'all' ? undefined : filter);

  return (
    <div className="flex h-full flex-col gap-4 p-6">
      <header className="flex flex-wrap items-baseline gap-4">
        <h1 className="text-xl font-semibold tracking-tight text-[#e6edf7]">
          Proposals
        </h1>
        <p className="text-sm text-[#7a8599]">
          Drafts from the three producers (capability gaps, library
          adoption, recipe retirement). All converge on operator review;
          promote via /cp/architecture-requests, /cp/changes, or directly
          via the meta-agent ledger.
        </p>
      </header>

      <div className="flex flex-wrap gap-2">
        {(['all', 'capability', 'library', 'recipe'] as const).map((k) => (
          <button
            type="button"
            key={k}
            onClick={() => setFilter(k)}
            className={`rounded border px-2 py-1 text-xs font-mono uppercase tracking-wide transition ${
              filter === k
                ? 'border-[#60a5fa] bg-[#60a5fa]/10 text-[#e6edf7]'
                : 'border-[#1f2937] text-[#7a8599] hover:border-[#374151]'
            }`}
          >
            {KIND_LABEL[k]}
          </button>
        ))}
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
            <div>
              No proposals
              {filter !== 'all' ? ` of kind ${filter}` : ''} yet. The
              producers run on idle cadences (capability_gap_analyzer
              daily, library_radar daily, recipe_consolidation weekly)
              and write here as evidence accumulates.
            </div>
          </div>
        ) : (
          <ul className="divide-y divide-[#1f2937] overflow-y-auto">
            {list.data.proposals.map((p) => (
              <Row
                key={`${p.kind}/${p.name}`}
                proposal={p}
                onClick={() => setOpen({ kind: p.kind, name: p.name })}
              />
            ))}
          </ul>
        )}
      </div>

      {open && (
        <Drawer
          kind={open.kind}
          name={open.name}
          onClose={() => setOpen(null)}
        />
      )}
    </div>
  );
}

function Row({
  proposal,
  onClick,
}: {
  proposal: ProposalSummary;
  onClick: () => void;
}) {
  return (
    <li
      onClick={onClick}
      className="cursor-pointer px-4 py-3 transition hover:bg-[#111827]"
    >
      <div className="flex flex-wrap items-baseline gap-3">
        <span
          className={`font-mono text-xs uppercase tracking-wide ${KIND_COLOR[proposal.kind]}`}
        >
          {proposal.kind}
        </span>
        <span className="text-sm text-[#e6edf7]">{proposal.title}</span>
        <span className="ml-auto text-xs text-[#7a8599]">
          {relTime(proposal.modified_at)}
        </span>
      </div>
      <p className="mt-1 truncate text-xs text-[#7a8599]">{proposal.name}</p>
    </li>
  );
}

function Drawer({
  kind,
  name,
  onClose,
}: {
  kind: ProposalKind;
  name: string;
  onClose: () => void;
}) {
  const detail = useProposalDetailQuery(kind, name);

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
          <span className={`font-mono text-xs uppercase ${KIND_COLOR[kind]}`}>
            {kind}
          </span>
          <h2 className="text-lg font-semibold text-[#e6edf7]">
            {detail.data?.title ?? name}
          </h2>
          <button
            type="button"
            className="ml-auto text-[#7a8599] hover:text-[#e6edf7]"
            onClick={onClose}
          >
            close
          </button>
        </div>

        {detail.isLoading ? (
          <Skeleton className="h-96" />
        ) : detail.error ? (
          <p className="text-sm text-[#f87171]">
            {(detail.error as Error).message}
          </p>
        ) : detail.data?.body ? (
          <pre className="overflow-x-auto whitespace-pre-wrap rounded bg-[#000] p-4 text-xs text-[#cbd5e1]">
            {detail.data.body}
          </pre>
        ) : detail.data?.row ? (
          <pre className="overflow-x-auto rounded bg-[#000] p-4 text-xs text-[#cbd5e1]">
            {JSON.stringify(detail.data.row, null, 2)}
          </pre>
        ) : null}
      </aside>
    </div>
  );
}
