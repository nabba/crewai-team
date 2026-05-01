import { useMemo } from 'react';
import { Skeleton } from './ui/Skeleton';
import { ErrorPanel } from './ui/ErrorPanel';
import {
  useErrorAuditQuery,
  useAcknowledgeAnomaly,
  type ErrorAuditAnomaly,
  type ErrorAuditPattern,
  type ErrorAuditTrendPoint,
} from '../api/queries';

// Permanent error monitor — surfaces signature-grouped patterns and open
// anomalies from app/observability/error_monitor.py. Shows up as the
// "Monitor" tab on /cp/ops.

function relTime(iso?: string | null): string {
  if (!iso) return '—';
  const t = new Date(iso).getTime();
  if (isNaN(t)) return iso;
  const secs = Math.max(0, Math.round((Date.now() - t) / 1000));
  if (secs < 60) return `${secs}s ago`;
  if (secs < 3600) return `${Math.floor(secs / 60)}m ago`;
  if (secs < 86400) return `${Math.floor(secs / 3600)}h ago`;
  return `${Math.floor(secs / 86400)}d ago`;
}

function severityCls(sev: string): string {
  switch (sev) {
    case 'critical':
      return 'bg-[#f87171]/15 text-[#f87171] border-[#f87171]/40';
    case 'warning':
      return 'bg-[#fbbf24]/15 text-[#fbbf24] border-[#fbbf24]/40';
    default:
      return 'bg-[#60a5fa]/15 text-[#60a5fa] border-[#60a5fa]/40';
  }
}

function trendArrow(trend: string): { glyph: string; cls: string } {
  if (trend === 'rising') return { glyph: '↑', cls: 'text-[#f87171]' };
  if (trend === 'falling') return { glyph: '↓', cls: 'text-[#34d399]' };
  return { glyph: '→', cls: 'text-[#7a8599]' };
}

function anomalyTypeLabel(t: string): string {
  if (t === 'new_pattern') return 'New pattern';
  if (t === 'rate_spike') return 'Rate spike';
  if (t === 'total_rate') return 'Total-rate σ';
  return t;
}

// ── Subcomponents ──────────────────────────────────────────────────────────

function SummaryCard({
  total24h,
  total1h,
  hourlyAvg,
  trend,
}: {
  total24h: number;
  total1h: number;
  hourlyAvg: number;
  trend: string;
}) {
  const arrow = trendArrow(trend);
  return (
    <div className="bg-[#111820] border border-[#1e2738] rounded-lg p-4 grid grid-cols-2 sm:grid-cols-4 gap-4">
      <div>
        <div className="text-[10px] uppercase tracking-wider text-[#7a8599]">Last 24h</div>
        <div className="text-2xl font-semibold text-[#e2e8f0]">{total24h.toLocaleString()}</div>
      </div>
      <div>
        <div className="text-[10px] uppercase tracking-wider text-[#7a8599]">Last hour</div>
        <div className="text-2xl font-semibold text-[#e2e8f0]">
          {total1h.toLocaleString()}
          <span className={`ml-2 text-base ${arrow.cls}`}>{arrow.glyph}</span>
        </div>
      </div>
      <div>
        <div className="text-[10px] uppercase tracking-wider text-[#7a8599]">24h avg / hour</div>
        <div className="text-2xl font-semibold text-[#e2e8f0]">{hourlyAvg.toFixed(1)}</div>
      </div>
      <div>
        <div className="text-[10px] uppercase tracking-wider text-[#7a8599]">Trend</div>
        <div className={`text-2xl font-semibold ${arrow.cls}`}>
          {arrow.glyph} {trend}
        </div>
      </div>
    </div>
  );
}

function AnomalyCard({
  anomaly,
  onAcknowledge,
  isPending,
}: {
  anomaly: ErrorAuditAnomaly;
  onAcknowledge: (id: string) => void;
  isPending: boolean;
}) {
  return (
    <div className={`border rounded-lg p-3 ${severityCls(anomaly.severity)}`}>
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 mb-1 flex-wrap">
            <span className="text-[10px] uppercase tracking-wider font-semibold">
              {anomalyTypeLabel(anomaly.type)}
            </span>
            <span className="text-[10px] uppercase font-medium">
              · {anomaly.severity}
            </span>
            <span className="text-[10px] text-[#7a8599]">
              · {relTime(anomaly.detected_at)}
            </span>
          </div>
          <div className="text-xs text-[#e2e8f0] line-clamp-2 break-words">
            {anomaly.sample || <span className="italic text-[#7a8599]">(no sample)</span>}
          </div>
          <div className="flex items-center gap-3 mt-2 text-[11px] text-[#7a8599]">
            <span>
              <span className="text-[#e2e8f0] font-mono">{anomaly.hourly_rate.toFixed(1)}</span> /h
            </span>
            {anomaly.baseline_rate > 0 && (
              <span>
                vs baseline <span className="text-[#e2e8f0] font-mono">{anomaly.baseline_rate.toFixed(2)}</span>
                <span className="ml-1">
                  ({(anomaly.hourly_rate / anomaly.baseline_rate).toFixed(1)}×)
                </span>
              </span>
            )}
            <span className="font-mono opacity-60">{anomaly.signature.slice(0, 8)}</span>
          </div>
        </div>
        <button
          type="button"
          disabled={isPending}
          onClick={() => onAcknowledge(anomaly.id)}
          className="text-[11px] px-2 py-1 rounded border border-[#1e2738] hover:bg-[#1e2738] text-[#7a8599] hover:text-[#e2e8f0] disabled:opacity-50 whitespace-nowrap"
        >
          {isPending ? '…' : 'Ack'}
        </button>
      </div>
    </div>
  );
}

function TrendBars({ points }: { points: ErrorAuditTrendPoint[] }) {
  const max = useMemo(
    () => Math.max(1, ...points.map((p) => p.count)),
    [points],
  );
  if (points.length === 0) {
    return (
      <p className="text-xs text-[#7a8599] italic">No data yet — first scan in &lt; 5 min.</p>
    );
  }
  // Pad to 24 buckets so the chart x-axis is consistent.
  const tail = points.slice(-24);
  return (
    <div className="flex items-end gap-1 h-24">
      {tail.map((p) => {
        const h = Math.max(2, Math.round((p.count / max) * 100));
        const cls = p.count >= max * 0.8
          ? 'bg-[#f87171]'
          : p.count >= max * 0.5
            ? 'bg-[#fbbf24]'
            : 'bg-[#60a5fa]';
        return (
          <div
            key={p.hour}
            title={`${new Date(p.hour).toLocaleString()} — ${p.count} errors`}
            className={`flex-1 rounded-t ${cls} opacity-80 hover:opacity-100`}
            style={{ height: `${h}%` }}
          />
        );
      })}
    </div>
  );
}

function PatternRow({ pattern }: { pattern: ErrorAuditPattern }) {
  return (
    <div className="bg-[#0a0e14] border border-[#1e2738] rounded p-2 flex items-start gap-3">
      <div className="font-mono text-[11px] text-[#7a8599] mt-0.5 shrink-0">
        {pattern.signature.slice(0, 8)}
      </div>
      <div className="min-w-0 flex-1">
        <div className="text-xs text-[#e2e8f0] line-clamp-2 break-words">
          {pattern.sample || <span className="italic text-[#7a8599]">(empty sample)</span>}
        </div>
      </div>
      <div className="text-right shrink-0">
        <div className="text-sm text-[#e2e8f0] font-medium">{pattern.count.toLocaleString()}</div>
        <div className="text-[10px] text-[#7a8599]">{pattern.share_pct.toFixed(1)}%</div>
      </div>
    </div>
  );
}

// ── Main component ─────────────────────────────────────────────────────────

export function ErrorMonitor() {
  const { data, isLoading, error, refetch } = useErrorAuditQuery();
  const ack = useAcknowledgeAnomaly();

  if (isLoading) return <Skeleton className="h-96" />;
  if (error) return <ErrorPanel error={error} onRetry={refetch} />;

  const summary = data?.summary ?? {
    total_24h: 0, total_1h: 0, hourly_avg_24h: 0, trend: 'stable' as const,
  };
  const anomalies = data?.active_anomalies ?? [];
  const patterns = data?.top_patterns_24h ?? [];
  const trend = data?.trend_hourly ?? [];

  return (
    <div className="space-y-5">
      {data?.error && (
        <div className="text-xs text-[#fbbf24]">{data.error}</div>
      )}

      <SummaryCard
        total24h={summary.total_24h}
        total1h={summary.total_1h}
        hourlyAvg={summary.hourly_avg_24h}
        trend={summary.trend}
      />

      {/* Active anomalies — surface first, they're actionable */}
      <section>
        <h3 className="text-xs font-medium text-[#7a8599] uppercase tracking-wider mb-2">
          Active Anomalies ({anomalies.length})
        </h3>
        {anomalies.length === 0 ? (
          <div className="bg-[#111820] border border-[#1e2738] rounded-lg p-6 text-center">
            <div className="text-2xl mb-1">✅</div>
            <p className="text-[#e2e8f0] text-sm font-medium">No active error anomalies</p>
            <p className="text-[11px] text-[#7a8599] mt-1">
              Detector flags new patterns, rate spikes, and 2σ deviations on total error rate.
            </p>
          </div>
        ) : (
          <div className="space-y-2">
            {anomalies.map((a) => (
              <AnomalyCard
                key={a.id}
                anomaly={a}
                onAcknowledge={(id) => ack.mutate(id)}
                isPending={ack.isPending && ack.variables === a.id}
              />
            ))}
          </div>
        )}
      </section>

      {/* Hourly trend */}
      <section>
        <h3 className="text-xs font-medium text-[#7a8599] uppercase tracking-wider mb-2">
          Hourly Error Volume (24h)
        </h3>
        <div className="bg-[#111820] border border-[#1e2738] rounded-lg p-4">
          <TrendBars points={trend} />
        </div>
      </section>

      {/* Top patterns */}
      <section>
        <h3 className="text-xs font-medium text-[#7a8599] uppercase tracking-wider mb-2">
          Top Patterns (24h, by frequency) — {patterns.length} of 20 shown
        </h3>
        {patterns.length === 0 ? (
          <p className="text-sm text-[#7a8599] italic">No patterns yet.</p>
        ) : (
          <div className="space-y-1.5">
            {patterns.map((p) => (
              <PatternRow key={p.signature} pattern={p} />
            ))}
          </div>
        )}
      </section>

      <p className="text-[10px] text-[#7a8599] italic">
        Updated: {data?.updated_at ? new Date(data.updated_at).toLocaleString() : '—'} ·
        Backend scans every 5 min · React polls every 30 s.
      </p>
    </div>
  );
}
