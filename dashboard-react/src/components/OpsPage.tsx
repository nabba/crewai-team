import { useMemo, useState } from 'react';
import { Skeleton } from './ui/Skeleton';
import { ErrorPanel } from './ui/ErrorPanel';
import {
  useErrorsQuery,
  useAnomaliesQuery,
  useDeploysQuery,
  useSnapshotKinds,
  useErrorAuditQuery,
  type ErrorEntry,
  type AnomalyAlert,
  type DeployEntry,
} from '../api/queries';
import { crewIcon, crewLabel } from '../crews';
import { SnapshotExplorer } from './SnapshotExplorer';
import { ErrorMonitor } from './ErrorMonitor';
import { CompanionTab } from './CompanionTab';

type OpsTab =
  | 'monitor'
  | 'errors'
  | 'anomalies'
  | 'deploys'
  | 'observability'
  | 'companion';

// Port of the old dashboard's "Errors & Self-Healing", "🛡️ Anomaly Detection"
// and "🏗️ Self-Deploy Pipeline" cards — grouped as tabs on a single page.

// Crew icons + labels come from the canonical registry.

function relTime(iso?: string): string {
  if (!iso) return '—';
  const t = new Date(iso).getTime();
  if (isNaN(t)) return iso;
  const secs = Math.max(0, Math.round((Date.now() - t) / 1000));
  if (secs < 60) return `${secs}s ago`;
  if (secs < 3600) return `${Math.floor(secs / 60)}m ago`;
  if (secs < 86400) return `${Math.floor(secs / 3600)}h ago`;
  return `${Math.floor(secs / 86400)}d ago`;
}

// ── Errors tab ──────────────────────────────────────────────────────────────

function ErrorsTab() {
  const { data, isLoading, error, refetch } = useErrorsQuery();
  if (isLoading) return <Skeleton className="h-64" />;
  if (error) return <ErrorPanel error={error} onRetry={refetch} />;
  // self_heal journal records oldest-first; newest on top is what we want.
  const recent = [...(data?.recent ?? [])].reverse();
  const patterns = data?.patterns ?? {};

  const patternEntries = Object.entries(patterns).sort((a, b) => b[1] - a[1]);

  return (
    <div className="space-y-4">
      {data?.error && (
        <div className="text-xs text-[#fbbf24]">{data.error}</div>
      )}

      {/* Patterns */}
      <section>
        <h3 className="text-xs font-medium text-[#7a8599] uppercase tracking-wider mb-2">
          Recurring Patterns ({patternEntries.length})
        </h3>
        {patternEntries.length === 0 ? (
          <p className="text-sm text-[#7a8599] italic">No recurring error patterns.</p>
        ) : (
          <div className="flex flex-wrap gap-2">
            {patternEntries.map(([key, count]) => {
              const [crew, type] = key.split(':');
              return (
                <span
                  key={key}
                  className="text-xs px-2 py-1 rounded border border-[#f87171]/30 bg-[#f87171]/10 text-[#f87171] flex items-center gap-1.5"
                  title={`${count} occurrences`}
                >
                  <span>{crewIcon(crew)}</span>
                  <span className="text-[#a78bfa] font-medium">{crewLabel(crew)}</span>
                  <span className="text-[#f87171]">{type}</span>
                  <span className="text-[#f87171]/70">×{count}</span>
                </span>
              );
            })}
          </div>
        )}
      </section>

      {/* Recent error list */}
      <section>
        <h3 className="text-xs font-medium text-[#7a8599] uppercase tracking-wider mb-2">
          Recent Errors ({recent.length})
        </h3>
        {recent.length === 0 ? (
          <p className="text-sm text-[#7a8599] italic">No recent errors.</p>
        ) : (
          <div className="space-y-2">
            {recent.map((e, i) => (
              <ErrorRow key={i} err={e} />
            ))}
          </div>
        )}
      </section>
    </div>
  );
}

function ErrorRow({ err }: { err: ErrorEntry }) {
  return (
    <div className="bg-[#0a0e14] border border-[#1e2738] rounded-lg p-3">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 mb-1">
            <span>{crewIcon(err.crew)}</span>
            <span className="text-xs text-[#a78bfa] font-semibold">{err.crew ? crewLabel(err.crew) : '—'}</span>
            <span className="text-xs text-[#f87171]">{err.error_type ?? 'unknown'}</span>
            {err.diagnosed && <span className="text-[10px] text-[#60a5fa]">· diagnosed</span>}
            {err.fix_applied && <span className="text-[10px] text-[#34d399]">· fix applied</span>}
          </div>
          {err.error_msg && (
            <div className="text-xs text-[#7a8599] line-clamp-2">{err.error_msg}</div>
          )}
          {err.user_input && (
            <div className="text-[10px] text-[#7a8599]/80 mt-1 truncate">
              input: {err.user_input.slice(0, 140)}
            </div>
          )}
        </div>
        <span className="text-[10px] text-[#7a8599] whitespace-nowrap">{relTime(err.ts)}</span>
      </div>
    </div>
  );
}

// ── Anomalies tab ───────────────────────────────────────────────────────────

function AnomaliesTab() {
  const { data, isLoading, error, refetch } = useAnomaliesQuery();
  if (isLoading) return <Skeleton className="h-64" />;
  if (error) return <ErrorPanel error={error} onRetry={refetch} />;
  // Journals record oldest-first; show newest on top.
  const alerts = [...(data?.recent_alerts ?? [])].reverse();
  if (alerts.length === 0) {
    return (
      <div className="bg-[#111820] border border-[#1e2738] rounded-lg p-8 text-center">
        <div className="text-2xl mb-2">✅</div>
        <p className="text-[#e2e8f0] font-medium">No anomalies — system healthy</p>
        <p className="text-xs text-[#7a8599] mt-1">
          The 2σ detector samples error rate, response time, output quality, and token usage.
        </p>
      </div>
    );
  }
  return (
    <div className="space-y-2">
      {data?.error && <div className="text-xs text-[#fbbf24]">{data.error}</div>}
      {alerts.map((a, i) => <AnomalyRow key={i} alert={a} />)}
    </div>
  );
}

function AnomalyRow({ alert }: { alert: AnomalyAlert }) {
  const sigma = alert.sigma ?? 0;
  const crit = sigma >= 3;
  const badgeCls = crit
    ? 'bg-[#f87171]/15 text-[#f87171] border-[#f87171]/30'
    : 'bg-[#fbbf24]/15 text-[#fbbf24] border-[#fbbf24]/30';
  return (
    <div className="flex items-center gap-3 bg-[#0a0e14] border border-[#1e2738] rounded-lg p-3">
      <span className={`text-[11px] font-bold px-2 py-0.5 rounded border ${badgeCls}`} title="sigma distance from mean">
        {sigma.toFixed(1)}σ
      </span>
      <div className="flex-1 min-w-0">
        <div className="text-sm text-[#e2e8f0] truncate">
          {alert.type ?? alert.metric ?? 'anomaly'}
          {alert.value != null && (
            <span className="text-[#7a8599] text-xs ml-2">= {alert.value.toFixed(3)}</span>
          )}
          {alert.mean != null && (
            <span className="text-[#7a8599] text-xs ml-1">(baseline {alert.mean.toFixed(3)})</span>
          )}
        </div>
        <div className="text-[10px] text-[#7a8599]">
          {alert.metric} · {alert.direction ?? '—'}
        </div>
      </div>
      <span className="text-[10px] text-[#7a8599] whitespace-nowrap">{relTime(alert.ts)}</span>
    </div>
  );
}

// ── Deploys tab ─────────────────────────────────────────────────────────────

function DeploysTab() {
  const { data, isLoading, error, refetch } = useDeploysQuery();
  if (isLoading) return <Skeleton className="h-64" />;
  if (error) return <ErrorPanel error={error} onRetry={refetch} />;
  // Journals record oldest-first; show newest on top.
  const deploys = [...(data?.recent ?? [])].reverse();
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3 text-xs text-[#7a8599]">
        <span>Auto-deploy:</span>
        <span className={`px-2 py-0.5 rounded border ${
          data?.auto_deploy_enabled
            ? 'border-[#34d399]/30 bg-[#34d399]/10 text-[#34d399]'
            : 'border-[#7a8599]/30 bg-[#7a8599]/10 text-[#7a8599]'
        }`}>
          {data?.auto_deploy_enabled ? 'ON' : data?.auto_deploy_enabled === false ? 'OFF' : 'unknown'}
        </span>
        <span className="text-[10px]">
          Toggle via Signal: <code className="text-[#60a5fa]">auto deploy on</code> / <code className="text-[#f87171]">auto deploy off</code>
        </span>
      </div>
      {data?.error && <div className="text-xs text-[#fbbf24]">{data.error}</div>}
      {deploys.length === 0 ? (
        <p className="text-sm text-[#7a8599] italic">No deployments recorded.</p>
      ) : (
        <div className="space-y-2">
          {deploys.map((d, i) => <DeployRow key={i} deploy={d} />)}
        </div>
      )}
    </div>
  );
}

const DEPLOY_ICON: Record<string, string> = {
  success: '✅',
  blocked: '🚫',
  rollback: '⏪',
  auto_rollback: '⏪',
};
const DEPLOY_COLOR: Record<string, string> = {
  success: 'border-[#34d399]/30 text-[#34d399]',
  blocked: 'border-[#f87171]/30 text-[#f87171]',
  rollback: 'border-[#fbbf24]/30 text-[#fbbf24]',
  auto_rollback: 'border-[#fbbf24]/30 text-[#fbbf24]',
};

function DeployRow({ deploy }: { deploy: DeployEntry }) {
  const status = deploy.status ?? 'unknown';
  const icon = DEPLOY_ICON[status] ?? '⏳';
  const clr = DEPLOY_COLOR[status] ?? 'border-[#7a8599]/30 text-[#7a8599]';
  return (
    <div className={`bg-[#0a0e14] border rounded-lg p-3 ${clr.split(' ')[0]}`}>
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-start gap-2 min-w-0">
          <span>{icon}</span>
          <div className="min-w-0">
            <div className={`text-xs font-semibold uppercase tracking-wider ${clr.split(' ')[1]}`}>
              {status.replace(/_/g, ' ')}
            </div>
            <div className="text-xs text-[#e2e8f0] break-words">{deploy.reason ?? '—'}</div>
            {deploy.files && deploy.files.length > 0 && (
              <div className="text-[10px] text-[#7a8599] mt-1">
                {deploy.files.slice(0, 3).map((f) => <code key={f} className="mr-2 text-[#60a5fa]">{f}</code>)}
                {deploy.files.length > 3 && <span>+{deploy.files.length - 3}</span>}
              </div>
            )}
            {deploy.error && (
              <div className="text-[10px] text-[#f87171] mt-1 break-words">{deploy.error}</div>
            )}
          </div>
        </div>
        <span className="text-[10px] text-[#7a8599] whitespace-nowrap">{relTime(deploy.ts)}</span>
      </div>
    </div>
  );
}

// ── Page ────────────────────────────────────────────────────────────────────

export function OpsPage() {
  const [tab, setTab] = useState<OpsTab>('monitor');

  // Small counts in tab labels for at-a-glance status.
  const errorsQ = useErrorsQuery();
  const anomaliesQ = useAnomaliesQuery();
  const deploysQ = useDeploysQuery();
  const snapshotKindsQ = useSnapshotKinds();
  const errorAuditQ = useErrorAuditQuery();

  const counts = useMemo(() => ({
    monitor: errorAuditQ.data?.active_anomalies.length ?? 0,
    errors: errorsQ.data?.recent.length ?? 0,
    anomalies: anomaliesQ.data?.recent_alerts.length ?? 0,
    deploys: deploysQ.data?.recent.length ?? 0,
    observability: snapshotKindsQ.data?.kinds.length ?? 0,
  }), [errorAuditQ.data, errorsQ.data, anomaliesQ.data, deploysQ.data, snapshotKindsQ.data]);

  const tabs: { key: OpsTab; label: string; icon: string; count: number }[] = [
    { key: 'monitor', label: 'Error Monitor', icon: '📈', count: counts.monitor },
    { key: 'errors', label: 'Errors & Self-Healing', icon: '⚠️', count: counts.errors },
    { key: 'anomalies', label: 'Anomaly Detection', icon: '🛡️', count: counts.anomalies },
    { key: 'deploys', label: 'Self-Deploy Pipeline', icon: '🏗️', count: counts.deploys },
    { key: 'observability', label: 'Observability Snapshots', icon: '📊', count: counts.observability },
    { key: 'companion', label: 'Workspace Companion', icon: '🌀', count: 0 },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold text-[#e2e8f0]">Operations</h1>
        <p className="text-sm text-[#7a8599] mt-1">
          Error journal, 2σ anomaly detector, and auto-deploy pipeline.
        </p>
      </div>

      <div className="flex gap-1 bg-[#111820] rounded-lg p-1 border border-[#1e2738] w-fit">
        {tabs.map((t) => (
          <button
            key={t.key}
            onClick={() => setTab(t.key)}
            className={`px-4 py-1.5 rounded-md text-sm transition-colors flex items-center gap-2 ${
              tab === t.key
                ? 'bg-[#60a5fa]/15 text-[#60a5fa] font-medium'
                : 'text-[#7a8599] hover:text-[#e2e8f0]'
            }`}
          >
            <span>{t.icon}</span>
            <span>{t.label}</span>
            <span className="text-[10px] text-[#7a8599]">({t.count})</span>
          </button>
        ))}
      </div>

      <div className="bg-[#111820] border border-[#1e2738] rounded-lg p-4">
        {tab === 'monitor' && <ErrorMonitor />}
        {tab === 'errors' && <ErrorsTab />}
        {tab === 'anomalies' && <AnomaliesTab />}
        {tab === 'deploys' && <DeploysTab />}
        {tab === 'observability' && <SnapshotExplorer />}
        {tab === 'companion' && <CompanionTab />}
      </div>
    </div>
  );
}
