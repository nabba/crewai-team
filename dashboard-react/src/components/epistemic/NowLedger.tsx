import { useMemo, useState } from 'react';
import type { ClaimDTO, VerificationStatus } from '../../types/epistemic';

const STATUS_TONE: Record<VerificationStatus, string> = {
  verified: 'bg-[#0e1f15] text-[#34d399] border-[#34d399]/30',
  inferred: 'bg-[#1f1c0e] text-[#fbbf24] border-[#fbbf24]/30',
  assumed: 'bg-[#11151c] text-[#7a8599] border-[#7a8599]/30',
  contradicted: 'bg-[#1f0e0e] text-[#f87171] border-[#f87171]/30',
};

type Filter = 'all' | 'load_bearing' | 'unverified';

export function NowLedger({ claims }: { claims: ClaimDTO[] }) {
  const [filter, setFilter] = useState<Filter>('all');

  const visible = useMemo(
    () =>
      claims.filter((c) => {
        if (filter === 'load_bearing') return c.load_bearing;
        if (filter === 'unverified')
          return (
            c.load_bearing &&
            (c.status === 'inferred' || c.status === 'assumed')
          );
        return true;
      }),
    [claims, filter],
  );

  return (
    <section className="rounded-lg bg-[#111820] border border-[#1e2738] p-4">
      <header className="flex items-center justify-between mb-3">
        <div>
          <h2 className="text-lg font-medium text-[#e2e8f0]">Claim Ledger</h2>
          <p className="text-xs text-[#7a8599]">
            Provenance for every assertion in the active task
          </p>
        </div>
        <FilterChips value={filter} onChange={setFilter} />
      </header>

      {visible.length === 0 ? (
        <EmptyState filter={filter} totalCount={claims.length} />
      ) : (
        <ul className="divide-y divide-[#1e2738]">
          {visible.map((c) => (
            <ClaimRow key={c.claim_id} claim={c} />
          ))}
        </ul>
      )}
    </section>
  );
}

function FilterChips({
  value,
  onChange,
}: {
  value: Filter;
  onChange: (f: Filter) => void;
}) {
  const opts: { id: Filter; label: string }[] = [
    { id: 'all', label: 'All' },
    { id: 'load_bearing', label: 'Load-bearing' },
    { id: 'unverified', label: 'Unverified' },
  ];
  return (
    <div className="flex gap-1 text-xs">
      {opts.map((o) => (
        <button
          key={o.id}
          onClick={() => onChange(o.id)}
          className={`px-2 py-1 rounded border ${
            value === o.id
              ? 'bg-[#60a5fa]/20 border-[#60a5fa]/60 text-[#60a5fa]'
              : 'bg-transparent border-[#1e2738] text-[#7a8599] hover:text-[#e2e8f0]'
          }`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

function ClaimRow({ claim }: { claim: ClaimDTO }) {
  const [open, setOpen] = useState(false);
  return (
    <li className="py-3">
      <button
        onClick={() => setOpen((o) => !o)}
        className="w-full text-left flex items-start gap-3"
      >
        <StatusBadge status={claim.status} />
        <div className="flex-1 min-w-0">
          <p className="text-sm text-[#e2e8f0] truncate">{claim.statement}</p>
          <p className="text-xs text-[#7a8599] mt-0.5">
            <span className="text-[#a78bfa]">{claim.agent_role}</span>
            <span className="mx-1.5">·</span>
            <span>{claim.register}</span>
            {claim.load_bearing && (
              <>
                <span className="mx-1.5">·</span>
                <span className="text-[#60a5fa]">load-bearing</span>
              </>
            )}
            {claim.tags.length > 0 && (
              <>
                <span className="mx-1.5">·</span>
                <span>{claim.tags.join(', ')}</span>
              </>
            )}
          </p>
        </div>
        {claim.verifying_action && claim.status === 'inferred' && (
          <span className="text-xs text-[#fbbf24] self-center whitespace-nowrap">
            verifier available
          </span>
        )}
      </button>
      {open && <ClaimDetail claim={claim} />}
    </li>
  );
}

function StatusBadge({ status }: { status: VerificationStatus }) {
  return (
    <span
      className={`inline-block px-2 py-0.5 rounded text-xs border ${STATUS_TONE[status]}`}
    >
      {status}
    </span>
  );
}

function ClaimDetail({ claim }: { claim: ClaimDTO }) {
  return (
    <div className="mt-3 ml-8 space-y-3 text-xs">
      {claim.evidence.length > 0 && (
        <div>
          <div className="text-[#7a8599] uppercase tracking-wide text-[10px]">
            Evidence
          </div>
          <ul className="mt-1 space-y-1.5">
            {claim.evidence.map((e, i) => (
              <li
                key={i}
                className="text-[#e2e8f0] bg-[#0a0e14] rounded p-2 border border-[#1e2738]"
              >
                <div className="text-[#7a8599] text-[10px] mb-1">
                  [{e.kind}] confidence={e.confidence.toFixed(2)}
                </div>
                <pre className="whitespace-pre-wrap break-words text-[11px]">
                  {e.excerpt}
                </pre>
              </li>
            ))}
          </ul>
        </div>
      )}
      {claim.verifying_action && (
        <div>
          <div className="text-[#7a8599] uppercase tracking-wide text-[10px]">
            Verifying action
          </div>
          <div className="mt-1 bg-[#0a0e14] rounded p-2 border border-[#1e2738]">
            <code className="text-[#22d3ee]">{claim.verifying_action.tool}</code>
            <span className="text-[#7a8599] ml-2">
              ~{claim.verifying_action.estimated_seconds}s ·{' '}
              {claim.verifying_action.safety}
            </span>
            <p className="text-[#7a8599] mt-1">
              {claim.verifying_action.expected_signal}
            </p>
            {Object.keys(claim.verifying_action.args).length > 0 && (
              <pre className="text-[#e2e8f0] text-[11px] mt-1 whitespace-pre-wrap">
                args: {JSON.stringify(claim.verifying_action.args)}
              </pre>
            )}
          </div>
        </div>
      )}
      {claim.superseded_by && (
        <div className="text-[#f87171]">
          superseded by{' '}
          <code className="bg-[#0a0e14] px-1 rounded">{claim.superseded_by}</code>
        </div>
      )}
    </div>
  );
}

function EmptyState({
  filter,
  totalCount,
}: {
  filter: Filter;
  totalCount: number;
}) {
  if (totalCount === 0) {
    return (
      <div className="text-sm text-[#7a8599] py-6 text-center">
        No claims emitted yet for this task.
      </div>
    );
  }
  return (
    <div className="text-sm text-[#7a8599] py-6 text-center">
      No claims match the <span className="text-[#e2e8f0]">{filter}</span>{' '}
      filter.
    </div>
  );
}
