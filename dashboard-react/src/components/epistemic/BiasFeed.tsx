import type { BiasFeedEntry, Severity } from '../../types/epistemic';

const SEVERITY_TONE: Record<Severity, string> = {
  low: 'bg-[#11151c] text-[#7a8599] border-[#7a8599]/30',
  medium: 'bg-[#1f1c0e] text-[#fbbf24] border-[#fbbf24]/30',
  high: 'bg-[#1f140e] text-[#fb923c] border-[#fb923c]/30',
  critical: 'bg-[#1f0e0e] text-[#f87171] border-[#f87171]/30',
};

export function BiasFeed({
  matches,
  windowMinutes,
}: {
  matches: BiasFeedEntry[];
  windowMinutes: number;
}) {
  return (
    <section className="rounded-lg bg-[#111820] border border-[#1e2738] p-4">
      <header className="flex items-center justify-between mb-3">
        <div>
          <h2 className="text-lg font-medium text-[#e2e8f0]">Bias feed</h2>
          <p className="text-xs text-[#7a8599]">
            Cognitive failures detected in the last{' '}
            <span className="text-[#e2e8f0]">{windowMinutes}</span> minutes
            across all tasks
          </p>
        </div>
        <span className="text-xs text-[#7a8599]">
          {matches.length} match{matches.length === 1 ? '' : 'es'}
        </span>
      </header>

      {matches.length === 0 ? (
        <div className="text-sm text-[#34d399] py-6 text-center">
          No bias firings in the window — calibration looks clean.
        </div>
      ) : (
        <ul className="divide-y divide-[#1e2738]">
          {matches.map((m) => (
            <FeedRow key={m.id} entry={m} />
          ))}
        </ul>
      )}
    </section>
  );
}

function FeedRow({ entry }: { entry: BiasFeedEntry }) {
  const when = formatRelative(entry.detected_at);
  return (
    <li className="py-2.5 flex items-start gap-3">
      <span
        className={`inline-block px-2 py-0.5 rounded text-xs border ${SEVERITY_TONE[entry.severity]}`}
      >
        {entry.severity}
      </span>
      <div className="flex-1 min-w-0">
        <p className="text-sm text-[#e2e8f0]">
          <code className="text-[#22d3ee]">{entry.bias_id}</code>
        </p>
        <p className="text-xs text-[#7a8599] mt-0.5">
          claim{' '}
          <code className="text-[#e2e8f0]">{entry.claim_id}</code> · task{' '}
          <code className="text-[#e2e8f0]">{entry.task_id}</code> · {when}
        </p>
        {entry.detail && Object.keys(entry.detail).length > 0 && (
          <pre className="text-[11px] text-[#7a8599] mt-1 whitespace-pre-wrap">
            {Object.entries(entry.detail)
              .map(([k, v]) => `${k}=${formatValue(v)}`)
              .join(' · ')}
          </pre>
        )}
      </div>
    </li>
  );
}

function formatRelative(iso: string): string {
  try {
    const then = new Date(iso).getTime();
    const now = Date.now();
    const diffSec = Math.round((now - then) / 1000);
    if (diffSec < 60) return `${diffSec}s ago`;
    const diffMin = Math.round(diffSec / 60);
    if (diffMin < 60) return `${diffMin}m ago`;
    const diffHr = Math.round(diffMin / 60);
    return `${diffHr}h ago`;
  } catch {
    return iso;
  }
}

function formatValue(v: unknown): string {
  if (typeof v === 'string') return v;
  if (typeof v === 'number') return String(v);
  if (typeof v === 'boolean') return String(v);
  return JSON.stringify(v);
}
