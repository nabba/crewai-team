// TuningProposalsPanel — autotune-generated proposals for the bias
// library and verifier registry.
//
// The autotuner analyses the bias-feed + override + peer-review
// + incident logs and emits TuningProposal records. The operator
// reviews each proposal here, accepts (with an optional note) or
// rejects, and then opens a CODEOWNERS PR with the YAML patch text.
//
// Per the safety boundary in SELF_REFLECTION.md §11: the layer
// PROPOSES; humans approve. Accepting here only marks the row in
// the DB — the YAML change still lands via PR review.
import { useState } from 'react';
import {
  useTuningAcceptMutation,
  useTuningProposalsQuery,
  useTuningRejectMutation,
  useTuningRunMutation,
} from '../../api/epistemic';
import type {
  TuningProposalDTO,
  TuningProposalKind,
  TuningProposalStatus,
} from '../../types/epistemic';

const KIND_TONE: Record<TuningProposalKind, string> = {
  severity_downgrade: 'text-[#fbbf24] border-[#fbbf24]/30 bg-[#1f1c0e]',
  severity_upgrade: 'text-[#fb923c] border-[#fb923c]/30 bg-[#1f1407]',
  retirement_candidate: 'text-[#7a8599] border-[#7a8599]/30 bg-[#11151c]',
  verifier_retirement: 'text-[#7a8599] border-[#7a8599]/30 bg-[#11151c]',
};

const STATUS_FILTERS: Array<{ label: string; value: string | null }> = [
  { label: 'Open', value: 'proposed' },
  { label: 'Accepted', value: 'accepted' },
  { label: 'Rejected', value: 'rejected' },
  { label: 'All', value: null },
];

export function TuningProposalsPanel() {
  const [statusFilter, setStatusFilter] =
    useState<string | null>('proposed');
  const proposalsQuery = useTuningProposalsQuery(statusFilter);
  const runMutation = useTuningRunMutation();

  return (
    <section className="rounded-lg bg-[#111820] border border-[#1e2738] p-4 space-y-4">
      <header className="flex items-baseline justify-between">
        <div>
          <h2 className="text-lg font-medium text-[#e2e8f0]">
            Autotune proposals
          </h2>
          <p className="text-xs text-[#7a8599]">
            The autotuner analyses bias-fire, override, peer-review,
            and incident patterns and proposes severity / retirement
            adjustments. Accepting a proposal here records the
            decision; the YAML change still requires a CODEOWNERS PR.
          </p>
        </div>
        <button
          onClick={() => runMutation.mutate(undefined)}
          disabled={runMutation.isPending}
          className="px-3 py-1 rounded text-xs border border-[#60a5fa]/40 text-[#60a5fa] hover:bg-[#60a5fa]/10 disabled:opacity-50 whitespace-nowrap"
        >
          {runMutation.isPending ? 'analyzing…' : 'Run analysis'}
        </button>
      </header>

      <FilterRow value={statusFilter} onChange={setStatusFilter} />

      {proposalsQuery.isLoading ? (
        <ListSkeleton />
      ) : proposalsQuery.isError ? (
        <ErrorRow>{String(proposalsQuery.error)}</ErrorRow>
      ) : proposalsQuery.data && proposalsQuery.data.proposals.length > 0 ? (
        <ProposalList proposals={proposalsQuery.data.proposals} />
      ) : (
        <EmptyState statusFilter={statusFilter} />
      )}
    </section>
  );
}

function FilterRow({
  value,
  onChange,
}: {
  value: string | null;
  onChange: (v: string | null) => void;
}) {
  return (
    <div className="flex items-center gap-1 text-xs">
      {STATUS_FILTERS.map((f) => (
        <button
          key={f.label}
          onClick={() => onChange(f.value)}
          className={
            f.value === value
              ? 'px-2 py-0.5 rounded bg-[#60a5fa]/20 text-[#60a5fa] border border-[#60a5fa]/40'
              : 'px-2 py-0.5 rounded text-[#7a8599] border border-[#1e2738] hover:text-[#e2e8f0]'
          }
        >
          {f.label}
        </button>
      ))}
    </div>
  );
}

function ProposalList({
  proposals,
}: {
  proposals: TuningProposalDTO[];
}) {
  return (
    <ul className="divide-y divide-[#1e2738]">
      {proposals.map((p) => (
        <ProposalRow key={p.proposal_id} proposal={p} />
      ))}
    </ul>
  );
}

function ProposalRow({ proposal }: { proposal: TuningProposalDTO }) {
  const [expanded, setExpanded] = useState(false);
  const [note, setNote] = useState('');
  const acceptMutation = useTuningAcceptMutation();
  const rejectMutation = useTuningRejectMutation();

  const isOpen = proposal.status === 'proposed';
  const kindLabel = proposal.kind.replace(/_/g, ' ');
  const confidencePct = Math.round(proposal.confidence * 100);

  return (
    <li className="py-3">
      <button
        onClick={() => setExpanded((v) => !v)}
        className="w-full text-left flex items-start gap-3 hover:bg-[#0a0e14]/50 -mx-2 px-2 rounded"
      >
        <span
          className={`px-2 py-0.5 rounded text-xs border whitespace-nowrap ${KIND_TONE[proposal.kind]}`}
        >
          {kindLabel}
        </span>
        <div className="flex-1 min-w-0">
          <p className="text-sm text-[#e2e8f0]">
            <code className="text-[#22d3ee]">
              {proposal.target_kind}/{proposal.target_id}
            </code>
            <span className="ml-2 text-xs text-[#7a8599]">
              confidence {confidencePct}%
            </span>
          </p>
          <p className="text-xs text-[#7a8599] mt-0.5 line-clamp-2">
            {proposal.rationale}
          </p>
        </div>
        <div className="flex flex-col items-end gap-0.5 text-xs whitespace-nowrap">
          <StatusBadge status={proposal.status} />
          <span className="text-[#7a8599]">{expanded ? '▾' : '▸'}</span>
        </div>
      </button>

      {expanded && (
        <div className="mt-3 ml-8 space-y-3">
          <Section title="Rationale">
            <p className="text-sm text-[#e2e8f0] whitespace-pre-wrap">
              {proposal.rationale}
            </p>
          </Section>

          <Section title="Metric evidence">
            <pre className="text-xs text-[#7a8599] whitespace-pre-wrap font-mono bg-[#0a0e14] p-2 rounded border border-[#1e2738]">
              {JSON.stringify(proposal.metric_evidence, null, 2)}
            </pre>
          </Section>

          <Section title="YAML patch">
            <pre className="text-xs text-[#e2e8f0] whitespace-pre-wrap font-mono bg-[#0a0e14] p-2 rounded border border-[#1e2738]">
              {proposal.yaml_patch}
            </pre>
          </Section>

          {proposal.operator_note && (
            <Section title="Operator note">
              <p className="text-sm text-[#a78bfa] italic">
                {proposal.operator_note}
              </p>
            </Section>
          )}

          {isOpen && (
            <div className="space-y-2 pt-2 border-t border-[#1e2738]">
              <textarea
                value={note}
                onChange={(e) => setNote(e.target.value)}
                placeholder="Optional note explaining the decision…"
                rows={2}
                className="w-full bg-[#0a0e14] border border-[#1e2738] rounded px-2 py-1 text-xs text-[#e2e8f0] placeholder-[#7a8599]/60"
              />
              <div className="flex gap-2 text-xs">
                <button
                  onClick={() =>
                    acceptMutation.mutate({
                      proposalId: proposal.proposal_id,
                      operatorNote: note,
                    })
                  }
                  disabled={
                    acceptMutation.isPending || rejectMutation.isPending
                  }
                  className="px-3 py-1 rounded border border-[#34d399]/40 text-[#34d399] hover:bg-[#34d399]/10 disabled:opacity-50"
                >
                  Accept
                </button>
                <button
                  onClick={() =>
                    rejectMutation.mutate({
                      proposalId: proposal.proposal_id,
                      operatorNote: note,
                    })
                  }
                  disabled={
                    acceptMutation.isPending || rejectMutation.isPending
                  }
                  className="px-3 py-1 rounded border border-[#f87171]/40 text-[#f87171] hover:bg-[#f87171]/10 disabled:opacity-50"
                >
                  Reject
                </button>
                <span className="ml-auto text-[#7a8599] self-center">
                  Acceptance records the decision; YAML change goes via PR.
                </span>
              </div>
            </div>
          )}

          <p className="text-xs text-[#7a8599]">
            <code>{proposal.proposal_id}</code>
            <span className="mx-1">·</span>
            content_hash <code>{proposal.content_hash}</code>
            <span className="mx-1">·</span>
            updated {new Date(proposal.updated_at).toLocaleString()}
          </p>
        </div>
      )}
    </li>
  );
}

function StatusBadge({ status }: { status: TuningProposalStatus }) {
  const tones: Record<TuningProposalStatus, string> = {
    proposed: 'text-[#60a5fa]',
    accepted: 'text-[#34d399]',
    rejected: 'text-[#f87171]',
    superseded: 'text-[#7a8599]',
  };
  return <span className={`text-xs ${tones[status]}`}>{status}</span>;
}

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section>
      <h4 className="text-xs uppercase tracking-wide text-[#7a8599]">
        {title}
      </h4>
      <div className="mt-1">{children}</div>
    </section>
  );
}

function EmptyState({ statusFilter }: { statusFilter: string | null }) {
  return (
    <div className="rounded border border-[#1e2738] bg-[#0a0e14] p-4 text-sm text-[#7a8599]">
      No {statusFilter ?? 'tuning'} proposals. Click <em>Run analysis</em>{' '}
      to compute fresh proposals from the last 7 days of bias / override
      / peer-review / incident data, or run from the CLI:{' '}
      <code>python -m app.epistemic</code>.
    </div>
  );
}

function ListSkeleton() {
  return (
    <div className="space-y-2">
      {Array.from({ length: 3 }).map((_, i) => (
        <div
          key={i}
          className="h-12 rounded border border-[#1e2738] bg-[#0a0e14] animate-pulse"
        />
      ))}
    </div>
  );
}

function ErrorRow({ children }: { children: React.ReactNode }) {
  return (
    <div className="rounded bg-[#1a0e0e] border border-[#f87171]/40 p-3 text-sm text-[#f87171]">
      {children}
    </div>
  );
}
