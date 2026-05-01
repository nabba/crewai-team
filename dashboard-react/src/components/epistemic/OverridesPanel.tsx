// OverridesPanel — user overrides of epistemic-gate verdicts.
//
// The override IS the strongest available learning signal. The
// dashboard surfaces the rate so operators can decide whether to
// flip blocking-mode on (low force_proceed rate) or tune the bias
// library down (high force_proceed rate).
import { useState } from 'react';
import {
  useOverridesRecentQuery,
  useOverrideStatsQuery,
} from '../../api/epistemic';
import type {
  OverrideDTO,
  OverrideStatsReport,
  OverrideUserAction,
} from '../../types/epistemic';

const ACTION_TONE: Record<OverrideUserAction, string> = {
  force_proceed: 'text-[#fb923c] border-[#fb923c]/30 bg-[#1f1407]',
  use_revision: 'text-[#fbbf24] border-[#fbbf24]/30 bg-[#1f1c0e]',
  abandon: 'text-[#7a8599] border-[#7a8599]/30 bg-[#11151c]',
};

const WINDOW_OPTIONS = [
  { label: '1h', minutes: 60 },
  { label: '24h', minutes: 1440 },
  { label: '7d', minutes: 7 * 1440 },
] as const;

export function OverridesPanel() {
  const [windowMin, setWindowMin] = useState<number>(1440);
  const statsQuery = useOverrideStatsQuery(windowMin);
  const recentQuery = useOverridesRecentQuery(windowMin, 50);

  return (
    <section className="rounded-lg bg-[#111820] border border-[#1e2738] p-4 space-y-4">
      <header className="flex items-baseline justify-between">
        <div>
          <h2 className="text-lg font-medium text-[#e2e8f0]">
            Gate overrides
          </h2>
          <p className="text-xs text-[#7a8599]">
            User pushed past a calibration verdict. Each override flushes
            to the Self-Improver as a USER_CORRECTION learning gap
            (signal_strength=0.9). High <code>force_proceed</code> rate
            means tune the bias library down.
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
      ) : recentQuery.data && recentQuery.data.overrides.length > 0 ? (
        <OverrideList items={recentQuery.data.overrides} />
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

function StatsTiles({ stats }: { stats: OverrideStatsReport }) {
  const fpRate =
    stats.total > 0 ? Math.round((stats.force_proceed / stats.total) * 100) : 0;
  return (
    <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
      <Tile label="Total" value={stats.total} tone="text-[#e2e8f0]" />
      <Tile
        label="Force proceed"
        value={stats.force_proceed}
        tone="text-[#fb923c]"
      />
      <Tile
        label="Use revision"
        value={stats.use_revision}
        tone="text-[#fbbf24]"
      />
      <Tile label="Abandon" value={stats.abandon} tone="text-[#7a8599]" />
      <Tile
        label="False-pos rate"
        value={`${fpRate}%`}
        tone={
          fpRate >= 30
            ? 'text-[#f87171]'
            : fpRate >= 10
              ? 'text-[#fbbf24]'
              : 'text-[#34d399]'
        }
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

function OverrideList({ items }: { items: OverrideDTO[] }) {
  return (
    <ul className="divide-y divide-[#1e2738]">
      {items.map((o) => (
        <li key={o.override_id} className="py-3">
          <div className="flex items-start gap-3">
            <span
              className={`px-2 py-0.5 rounded text-xs border whitespace-nowrap ${ACTION_TONE[o.user_action]}`}
            >
              {o.user_action.replace('_', ' ')}
            </span>
            <div className="flex-1 min-w-0">
              <p className="text-sm text-[#e2e8f0]">
                <span className="text-[#7a8599]">on a {o.blocked_action}:</span>{' '}
                <span className="italic">
                  {o.user_reasoning || '(no reasoning given)'}
                </span>
              </p>
              <p className="text-xs text-[#7a8599] mt-1">
                <code className="text-[#7a8599]">{o.override_id}</code>
                <span className="mx-1">·</span>
                task <code className="text-[#7a8599]">{o.task_id}</code>
                {o.peer_review_id !== null && (
                  <>
                    <span className="mx-1">·</span>
                    review #{o.peer_review_id}
                  </>
                )}
                <span className="mx-1">·</span>
                {new Date(o.overridden_at).toLocaleString()}
              </p>
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
      No overrides in this window. Overrides record only when blocking-mode
      is on (<code>EPISTEMIC_BLOCKING_MODE=true</code>) and the user
      pushes past a verdict.
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
