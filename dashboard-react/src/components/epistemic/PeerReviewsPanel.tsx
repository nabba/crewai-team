// PeerReviewsPanel — destructive-recommendation peer reviews.
//
// Reads /epistemic/peer-reviews/{stats,recent}. Mirrors the
// PushbackPanel structure: tile row with allow/revise/veto counts +
// mean duration, then an event list with decision-toned badges and
// proposal excerpts.
import { useState } from 'react';
import {
  usePeerReviewsRecentQuery,
  usePeerReviewStatsQuery,
} from '../../api/epistemic';
import type {
  PeerReviewDecision,
  PeerReviewDTO,
  PeerReviewStatsReport,
} from '../../types/epistemic';

const DECISION_TONE: Record<PeerReviewDecision, string> = {
  allow: 'text-[#34d399] border-[#34d399]/30 bg-[#0e1f15]',
  revise: 'text-[#fbbf24] border-[#fbbf24]/30 bg-[#1f1c0e]',
  veto: 'text-[#f87171] border-[#f87171]/30 bg-[#1f0e0e]',
};

const WINDOW_OPTIONS = [
  { label: '1h', minutes: 60 },
  { label: '24h', minutes: 1440 },
  { label: '7d', minutes: 7 * 1440 },
] as const;

export function PeerReviewsPanel() {
  const [windowMin, setWindowMin] = useState<number>(1440);
  const statsQuery = usePeerReviewStatsQuery(windowMin);
  const recentQuery = usePeerReviewsRecentQuery(windowMin, 50);

  return (
    <section className="rounded-lg bg-[#111820] border border-[#1e2738] p-4 space-y-4">
      <header className="flex items-baseline justify-between">
        <div>
          <h2 className="text-lg font-medium text-[#e2e8f0]">
            Peer review (destructive)
          </h2>
          <p className="text-xs text-[#7a8599]">
            Adversarial second-opinion when{' '}
            <code>destructive_without_recheck</code> fires CRITICAL.
            Phase 6 default executor: ledger-health heuristic (vetoes
            when load-bearing claims are unverified). LLM executor
            opt-in via <code>EPISTEMIC_PEER_REVIEW_LLM=true</code>.
          </p>
        </div>
        <WindowSelector value={windowMin} onChange={setWindowMin} />
      </header>

      {statsQuery.isLoading ? (
        <StatsSkeleton />
      ) : statsQuery.isError ? (
        <ErrorRow>{String(statsQuery.error)}</ErrorRow>
      ) : statsQuery.data ? (
        <StatsTiles stats={statsQuery.data} />
      ) : null}

      {recentQuery.isLoading ? (
        <ListSkeleton />
      ) : recentQuery.isError ? (
        <ErrorRow>{String(recentQuery.error)}</ErrorRow>
      ) : recentQuery.data && recentQuery.data.reviews.length > 0 ? (
        <ReviewList reviews={recentQuery.data.reviews} />
      ) : (
        <EmptyState />
      )}
    </section>
  );
}

function WindowSelector({
  value,
  onChange,
}: {
  value: number;
  onChange: (n: number) => void;
}) {
  return (
    <div className="flex items-center gap-1 text-xs">
      {WINDOW_OPTIONS.map((opt) => (
        <button
          key={opt.minutes}
          onClick={() => onChange(opt.minutes)}
          className={
            opt.minutes === value
              ? 'px-2 py-0.5 rounded bg-[#60a5fa]/20 text-[#60a5fa] border border-[#60a5fa]/40'
              : 'px-2 py-0.5 rounded text-[#7a8599] border border-[#1e2738] hover:text-[#e2e8f0]'
          }
        >
          {opt.label}
        </button>
      ))}
    </div>
  );
}

function StatsTiles({ stats }: { stats: PeerReviewStatsReport }) {
  return (
    <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
      <Tile label="Total" value={stats.total} tone="text-[#e2e8f0]" />
      <Tile label="Allow" value={stats.allow} tone="text-[#34d399]" />
      <Tile label="Revise" value={stats.revise} tone="text-[#fbbf24]" />
      <Tile label="Veto" value={stats.veto} tone="text-[#f87171]" />
      <Tile
        label="Mean seconds"
        value={stats.mean_seconds.toFixed(2)}
        tone="text-[#a78bfa]"
      />
    </div>
  );
}

function Tile({
  label,
  value,
  tone,
}: {
  label: string;
  value: number | string;
  tone: string;
}) {
  return (
    <div className="rounded border border-[#1e2738] bg-[#0a0e14] px-3 py-2">
      <div className="text-xs text-[#7a8599] uppercase tracking-wide">
        {label}
      </div>
      <div className={`text-xl font-semibold mt-0.5 ${tone}`}>{value}</div>
    </div>
  );
}

function ReviewList({ reviews }: { reviews: PeerReviewDTO[] }) {
  return (
    <ul className="divide-y divide-[#1e2738]">
      {reviews.map((r) => (
        <li key={r.id} className="py-3">
          <div className="flex items-start gap-3">
            <span
              className={`px-2 py-0.5 rounded text-xs border whitespace-nowrap ${DECISION_TONE[r.decision]}`}
            >
              {r.decision}
            </span>
            <div className="flex-1 min-w-0">
              <p className="text-sm text-[#e2e8f0]">
                <span className="text-[#7a8599]">proposal:</span>{' '}
                <span className="italic">{r.proposal_excerpt}</span>
              </p>
              <p className="text-xs text-[#7a8599] mt-1">
                {r.reviewers.join(', ') || 'no reviewers'}
                <span className="mx-1">·</span>
                {r.duration_seconds.toFixed(2)}s
                <span className="mx-1">·</span>
                {new Date(r.requested_at).toLocaleString()}
                {r.triggering_claim_id && (
                  <>
                    <span className="mx-1">·</span>
                    claim{' '}
                    <code className="text-[#22d3ee]">
                      {r.triggering_claim_id}
                    </code>
                  </>
                )}
              </p>
              {r.rationale && (
                <p className="text-xs text-[#a78bfa] mt-1">{r.rationale}</p>
              )}
              {r.suggested_revision && (
                <pre className="text-xs text-[#7a8599] mt-1 whitespace-pre-wrap font-mono bg-[#0a0e14] p-2 rounded border border-[#1e2738]">
                  {r.suggested_revision}
                </pre>
              )}
            </div>
          </div>
        </li>
      ))}
    </ul>
  );
}

function EmptyState() {
  return (
    <div className="rounded border border-[#1e2738] bg-[#0a0e14] p-4 text-sm text-[#7a8599]">
      No peer-review escalations in this window. Reviews fire when the
      calibration gate's <code>suggested_action="peer_review"</code>{' '}
      (typically <code>destructive_without_recheck</code> at CRITICAL
      severity).
    </div>
  );
}

function StatsSkeleton() {
  return (
    <div className="grid grid-cols-5 gap-3">
      {Array.from({ length: 5 }).map((_, i) => (
        <div
          key={i}
          className="h-16 rounded border border-[#1e2738] bg-[#0a0e14] animate-pulse"
        />
      ))}
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
