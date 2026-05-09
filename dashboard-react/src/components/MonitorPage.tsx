import { useMemo } from 'react';
import { Skeleton } from './ui/Skeleton';
import { ErrorPanel } from './ui/ErrorPanel';
import {
  useSystemStatusQuery,
  type SystemCheck,
  type StatusLevel,
} from '../api/queries';

// Comprehensive monitoring pane: containers, gateways, messaging,
// internal subsystems, and external-service credit alerts.
// Polls /api/cp/system-status every 10 s.

const LEVEL_STYLE: Record<StatusLevel, { bg: string; text: string; border: string; label: string }> = {
  ok:    { bg: 'bg-[#34d399]/10', text: 'text-[#34d399]', border: 'border-[#34d399]/30', label: 'OK' },
  warn:  { bg: 'bg-[#fbbf24]/10', text: 'text-[#fbbf24]', border: 'border-[#fbbf24]/30', label: 'WARN' },
  error: { bg: 'bg-[#f87171]/10', text: 'text-[#f87171]', border: 'border-[#f87171]/30', label: 'ERROR' },
};

export function MonitorPage() {
  const q = useSystemStatusQuery();

  const grouped = useMemo(() => {
    const m = new Map<string, SystemCheck[]>();
    for (const c of q.data?.checks ?? []) {
      const arr = m.get(c.category) ?? [];
      arr.push(c);
      m.set(c.category, arr);
    }
    return Array.from(m.entries());
  }, [q.data?.checks]);

  const creditAlerts = useMemo(
    () => (q.data?.checks ?? []).filter((c) => c.status === 'error' && c.link),
    [q.data?.checks],
  );

  const overall = q.data?.overall ?? 'ok';

  return (
    <div className="space-y-4 max-w-5xl">
      <header className="flex items-baseline justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold text-[#e2e8f0]">System monitor</h1>
          <p className="text-xs text-[#7a8599] mt-1">
            Containers · gateways · messaging · internal subsystems · external services.
            Polls every 10 s. Errors with a top-up link mean a provider is out of credit.
          </p>
        </div>
        {q.data?.updated_at && (
          <span className="text-[10px] text-[#7a8599] whitespace-nowrap">
            updated {new Date(q.data.updated_at).toLocaleTimeString()}
          </span>
        )}
      </header>

      {/* Headline strip */}
      <div className={`rounded-xl border px-4 py-3 ${LEVEL_STYLE[overall].bg} ${LEVEL_STYLE[overall].border}`}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className={`text-xs font-mono px-2 py-0.5 rounded ${LEVEL_STYLE[overall].text} bg-black/30 border ${LEVEL_STYLE[overall].border}`}>
              {LEVEL_STYLE[overall].label}
            </span>
            <span className="text-sm text-[#e2e8f0] font-medium">
              {overall === 'ok'
                ? 'All systems nominal'
                : overall === 'warn'
                  ? 'One or more subsystems degraded'
                  : 'One or more subsystems failing'}
            </span>
          </div>
          {q.data?.by_category && (
            <div className="text-[11px] text-[#7a8599] flex gap-3">
              {Object.entries(q.data.by_category).map(([cat, counts]) => (
                <span key={cat}>
                  {cat}: <span className="text-[#34d399]">{counts.ok ?? 0}</span>
                  {(counts.warn ?? 0) > 0 && (
                    <> · <span className="text-[#fbbf24]">{counts.warn} warn</span></>
                  )}
                  {(counts.error ?? 0) > 0 && (
                    <> · <span className="text-[#f87171]">{counts.error} err</span></>
                  )}
                </span>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Credit-alert call-out — pinned at the top when any provider is exhausted */}
      {creditAlerts.length > 0 && (
        <div className="rounded-xl border border-[#f87171]/30 bg-[#f87171]/10 p-4 space-y-2">
          <div className="text-sm font-semibold text-[#f87171]">
            Provider credit exhausted — action required
          </div>
          {creditAlerts.map((c) => (
            <div key={c.name} className="flex items-center justify-between gap-3">
              <div className="min-w-0">
                <div className="text-sm text-[#e2e8f0]">{c.name}</div>
                <div className="text-xs text-[#7a8599] truncate">{c.message}</div>
              </div>
              {c.link && (
                <a
                  href={c.link}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="shrink-0 text-xs px-3 py-1.5 rounded-lg border border-[#f87171]/40 text-[#f87171] hover:bg-[#f87171]/20 whitespace-nowrap"
                >
                  Top up →
                </a>
              )}
            </div>
          ))}
        </div>
      )}

      {q.isLoading ? (
        <Skeleton className="h-64" />
      ) : q.error ? (
        <ErrorPanel error={q.error} onRetry={q.refetch} />
      ) : (
        <div className="space-y-4">
          {grouped.map(([category, items]) => (
            <CategoryCard key={category} category={category} items={items} />
          ))}
        </div>
      )}
    </div>
  );
}

function CategoryCard({ category, items }: { category: string; items: SystemCheck[] }) {
  return (
    <section className="bg-[#111820] border border-[#1e2738] rounded-xl p-3">
      <div className="text-[11px] uppercase tracking-wider text-[#7a8599] mb-2 px-1">
        {category}
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
        {items.map((c) => (
          <CheckRow key={c.name} c={c} />
        ))}
      </div>
    </section>
  );
}

function CheckRow({ c }: { c: SystemCheck }) {
  const s = LEVEL_STYLE[c.status];
  return (
    <div className={`flex items-start gap-3 rounded-lg border ${s.border} ${s.bg} px-3 py-2`}>
      <span
        className={`mt-0.5 text-[10px] font-mono px-1.5 py-0.5 rounded ${s.text} bg-black/30 border ${s.border} whitespace-nowrap`}
      >
        {s.label}
      </span>
      <div className="flex-1 min-w-0">
        <div className="flex items-baseline gap-2 justify-between">
          <span className="text-sm text-[#e2e8f0] font-medium truncate">{c.name}</span>
          {typeof c.latency_ms === 'number' && c.latency_ms > 0 && (
            <span className="text-[10px] text-[#7a8599]">{c.latency_ms} ms</span>
          )}
        </div>
        <div className="text-xs text-[#cbd5e1] break-words">{c.message}</div>
        {c.link && (
          <a
            href={c.link}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-block mt-1 text-[11px] text-[#60a5fa] hover:underline"
          >
            Open →
          </a>
        )}
        {c.since && (
          <div className="text-[10px] text-[#7a8599] mt-0.5">
            since {new Date(c.since).toLocaleString()}
          </div>
        )}
      </div>
    </div>
  );
}
