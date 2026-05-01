// Pushback panel — outcomes of the foundation re-check protocol.
//
// Reads /epistemic/pushback/stats (aggregate counts + mean
// time-to-recheck) and /epistemic/pushback/recent (list of events).
// The panel is the user-facing surface for the adversarial trigger:
// when a user contradicts a claim, did the protocol re-verify the
// foundation or did it expand the investigation? Phase 3 ships this
// view; Phase 5 wires the orchestrator hookup.
import { useState } from 'react';
import {
  usePushbackRecentQuery,
  usePushbackStatsQuery,
} from '../../api/epistemic';
import type {
  PushbackEventDTO,
  PushbackOutcome,
  PushbackStatsReport,
} from '../../types/epistemic';

const OUTCOME_TONE: Record<PushbackOutcome, string> = {
  reverified: 'text-[#34d399] border-[#34d399]/30 bg-[#0e1f15]',
  falsified: 'text-[#f87171] border-[#f87171]/30 bg-[#1f0e0e]',
  unverifiable: 'text-[#fbbf24] border-[#fbbf24]/30 bg-[#1f1c0e]',
};

const WINDOW_OPTIONS = [
  { label: '1h', minutes: 60 },
  { label: '24h', minutes: 1440 },
  { label: '7d', minutes: 7 * 1440 },
] as const;

export function PushbackPanel() {
  const [windowMin, setWindowMin] = useState<number>(1440);
  const statsQuery = usePushbackStatsQuery(windowMin);
  const recentQuery = usePushbackRecentQuery(windowMin, 50);

  return (
    <section className="rounded-lg bg-[#111820] border border-[#1e2738] p-4 space-y-4">
      <header className="flex items-baseline justify-between">
        <div>
          <h2 className="text-lg font-medium text-[#e2e8f0]">Pushback</h2>
          <p className="text-xs text-[#7a8599]">
            Foundation re-checks after user contradiction. The protocol
            runs only the load-bearing claim's verifier — by design, no
            investigation expansion.
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
      ) : recentQuery.data && recentQuery.data.events.length > 0 ? (
        <EventsList events={recentQuery.data.events} />
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

function StatsTiles({ stats }: { stats: PushbackStatsReport }) {
  return (
    <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
      <Tile label="Total" value={stats.total} tone="text-[#e2e8f0]" />
      <Tile
        label="Reverified"
        value={stats.reverified}
        tone="text-[#34d399]"
      />
      <Tile
        label="Falsified"
        value={stats.falsified}
        tone="text-[#f87171]"
      />
      <Tile
        label="Unverifiable"
        value={stats.unverifiable}
        tone="text-[#fbbf24]"
      />
      <Tile
        label="Mean seconds"
        value={stats.mean_seconds_to_recheck.toFixed(2)}
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

function EventsList({ events }: { events: PushbackEventDTO[] }) {
  return (
    <ul className="divide-y divide-[#1e2738]">
      {events.map((e) => (
        <li key={e.id} className="py-3">
          <div className="flex items-start gap-3">
            <span
              className={`px-2 py-0.5 rounded text-xs border whitespace-nowrap ${OUTCOME_TONE[e.outcome]}`}
            >
              {e.outcome}
            </span>
            <div className="flex-1 min-w-0">
              <p className="text-sm text-[#e2e8f0]">
                <span className="text-[#7a8599]">user:</span>{' '}
                <span className="italic">{e.user_evidence}</span>
              </p>
              <p className="text-xs text-[#7a8599] mt-1">
                claim <code className="text-[#22d3ee]">{e.contradicted_claim_id}</code>
                <span className="mx-1">·</span>
                {e.detector}
                <span className="mx-1">·</span>
                {e.duration_seconds.toFixed(2)}s
                <span className="mx-1">·</span>
                {new Date(e.detected_at).toLocaleString()}
              </p>
              {e.new_evidence_excerpt && (
                <pre className="text-xs text-[#7a8599] mt-1 whitespace-pre-wrap font-mono bg-[#0a0e14] p-2 rounded border border-[#1e2738]">
                  {e.new_evidence_excerpt}
                </pre>
              )}
              {e.invalidated_claim_ids.length > 0 && (
                <p className="text-xs text-[#f87171] mt-1">
                  cascaded:{' '}
                  {e.invalidated_claim_ids.map((id) => (
                    <code key={id} className="text-[#f87171] mr-1">
                      {id}
                    </code>
                  ))}
                </p>
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
      No pushback events in this window. The protocol fires when a user
      message contradicts a recent load-bearing claim — see the design
      doc §8 for the detection rules.
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
