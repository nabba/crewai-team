// CostTrendsCard — multi-year cost-trend visualization for /cp/costs.
// Reads /api/cp/costs/trends; renders monthly history + 6-month forecast
// with 95% CI band + a small anomalies list.
//
// PROGRAM §40 — Q3 Item 14. Read-only. Does not mutate cost data.
import { useMemo } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import { Skeleton } from './ui/Skeleton';
import { ErrorPanel } from './ui/ErrorPanel';
import { useCostTrendsQuery } from '../api/queries';
import { useProject } from '../context/useProject';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler,
);

const PANEL_BG = '#111820';
const PANEL_BORDER = '#1e2738';
const TEXT_DIM = '#7a8599';
const TEXT_BRIGHT = '#e2e8f0';

const COLOR_HISTORY = '#60a5fa';      // blue — observed
const COLOR_FORECAST = '#a78bfa';     // violet — projected
const COLOR_CI_BAND = 'rgba(167, 139, 250, 0.15)';

function formatUsd(n: number, digits = 2): string {
  if (!isFinite(n)) return '—';
  if (n >= 1000) return `$${n.toFixed(0)}`;
  return `$${n.toFixed(digits)}`;
}

export function CostTrendsCard() {
  const { activeProject } = useProject();
  const trendsQ = useCostTrendsQuery(12, 6, 30, 3.0, activeProject?.id);

  const chartData = useMemo(() => {
    if (!trendsQ.data) return null;
    const { monthly, forecast } = trendsQ.data;
    if (monthly.length === 0 && forecast.length === 0) return null;
    const labels: string[] = [
      ...monthly.map((m) => m.month),
      ...forecast.map((f) => f.month),
    ];
    const historyValues: (number | null)[] = [
      ...monthly.map((m) => m.total_cost_usd),
      ...forecast.map(() => null),
    ];
    const projValues: (number | null)[] = [
      ...monthly.map(() => null),
      ...forecast.map((f) => f.projected_usd),
    ];
    // Bridge the seam so history visually flows into forecast.
    if (monthly.length > 0 && forecast.length > 0) {
      projValues[monthly.length - 1] = monthly[monthly.length - 1].total_cost_usd;
    }
    const ciLow: (number | null)[] = [
      ...monthly.map(() => null),
      ...forecast.map((f) => f.ci_low),
    ];
    const ciHigh: (number | null)[] = [
      ...monthly.map(() => null),
      ...forecast.map((f) => f.ci_high),
    ];
    return {
      labels,
      datasets: [
        {
          label: 'Observed',
          data: historyValues,
          borderColor: COLOR_HISTORY,
          backgroundColor: 'rgba(96, 165, 250, 0.10)',
          pointBackgroundColor: COLOR_HISTORY,
          pointRadius: 3,
          fill: false,
          tension: 0.3,
          spanGaps: false,
        },
        {
          label: 'Forecast',
          data: projValues,
          borderColor: COLOR_FORECAST,
          backgroundColor: 'rgba(167, 139, 250, 0.05)',
          pointBackgroundColor: COLOR_FORECAST,
          pointRadius: 3,
          fill: false,
          tension: 0.3,
          borderDash: [6, 4],
          spanGaps: true,
        },
        {
          label: '95% CI low',
          data: ciLow,
          borderColor: 'rgba(167, 139, 250, 0.0)',
          backgroundColor: COLOR_CI_BAND,
          pointRadius: 0,
          fill: '+1', // fill upward to ciHigh
          tension: 0.3,
          spanGaps: false,
        },
        {
          label: '95% CI high',
          data: ciHigh,
          borderColor: 'rgba(167, 139, 250, 0.0)',
          backgroundColor: COLOR_CI_BAND,
          pointRadius: 0,
          fill: false,
          tension: 0.3,
          spanGaps: false,
        },
      ],
    };
  }, [trendsQ.data]);

  const summary = trendsQ.data?.summary;
  const anomalies = trendsQ.data?.anomalies ?? [];

  return (
    <div className="space-y-4">
      <div
        className="rounded-lg p-4 border"
        style={{ background: PANEL_BG, borderColor: PANEL_BORDER }}
      >
        <div className="flex items-baseline justify-between mb-1">
          <h2
            className="text-sm font-medium"
            style={{ color: TEXT_BRIGHT }}
          >
            Multi-year Cost Trend
          </h2>
          <span className="text-[10px]" style={{ color: TEXT_DIM }}>
            12-month history · 6-month forecast (95% CI)
          </span>
        </div>
        <p className="text-[10px] mb-4" style={{ color: TEXT_DIM }}>
          Pure-stdlib OLS regression on monthly totals from{' '}
          <code>audit_log</code>. Forecast is observational — system never
          auto-acts on it.
        </p>

        {trendsQ.isLoading ? (
          <Skeleton className="h-72" />
        ) : trendsQ.error ? (
          <ErrorPanel error={trendsQ.error} onRetry={trendsQ.refetch} />
        ) : !chartData ? (
          <div
            className="h-72 flex items-center justify-center text-sm italic"
            style={{ color: TEXT_DIM }}
          >
            Not enough cost history yet — needs ≥2 months of audit_log data.
          </div>
        ) : (
          <div className="h-72">
            <Line
              data={chartData}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                  legend: {
                    labels: {
                      color: TEXT_DIM,
                      font: { size: 11 },
                      // Hide the CI low/high pseudo-series labels; the band
                      // self-explains via the dashed forecast line.
                      filter: (item: { text: string }) =>
                        !item.text.startsWith('95% CI'),
                    },
                  },
                  tooltip: {
                    backgroundColor: PANEL_BG,
                    borderColor: PANEL_BORDER,
                    borderWidth: 1,
                    titleColor: TEXT_BRIGHT,
                    bodyColor: TEXT_DIM,
                    callbacks: {
                      label: (ctx) => {
                        const v = ctx.raw as number | null;
                        if (v === null) return '';
                        return ` ${ctx.dataset.label}: ${formatUsd(v, 4)}`;
                      },
                    },
                  },
                },
                scales: {
                  x: {
                    ticks: { color: TEXT_DIM, font: { size: 10 }, maxRotation: 45 },
                    grid: { color: PANEL_BORDER },
                  },
                  y: {
                    ticks: {
                      color: TEXT_DIM,
                      font: { size: 10 },
                      callback: (v) => formatUsd(v as number, 2),
                    },
                    grid: { color: PANEL_BORDER },
                  },
                },
              }}
            />
          </div>
        )}
      </div>

      {/* Summary KPIs */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <KPI
          label={`Total — last ${summary?.history_months_observed ?? 0} mo`}
          value={summary ? formatUsd(summary.total_history_usd, 2) : '—'}
          color="#34d399"
          loading={trendsQ.isLoading}
        />
        <KPI
          label="Trend (compounded)"
          value={
            summary?.trend_pct_per_month == null
              ? '—'
              : `${summary.trend_pct_per_month >= 0 ? '+' : ''}${summary.trend_pct_per_month.toFixed(1)}% / mo`
          }
          color={
            summary?.trend_pct_per_month == null
              ? TEXT_DIM
              : summary.trend_pct_per_month > 5
              ? '#f87171'
              : summary.trend_pct_per_month < -5
              ? '#34d399'
              : '#fbbf24'
          }
          loading={trendsQ.isLoading}
        />
        <KPI
          label="Projected — next 12 mo"
          value={summary ? formatUsd(summary.projected_next_12mo_usd, 0) : '—'}
          color={COLOR_FORECAST}
          loading={trendsQ.isLoading}
        />
      </div>

      {/* Anomalies */}
      {anomalies.length > 0 && (
        <div
          className="rounded-lg overflow-hidden border"
          style={{ background: PANEL_BG, borderColor: PANEL_BORDER }}
        >
          <div className="px-4 py-3 border-b" style={{ borderColor: PANEL_BORDER }}>
            <h3 className="text-sm font-medium" style={{ color: TEXT_BRIGHT }}>
              Daily Anomalies
            </h3>
            <p className="text-[10px]" style={{ color: TEXT_DIM }}>
              Days outside ±{trendsQ.data?.params.anomaly_z_threshold ?? 3}σ of the
              trailing {trendsQ.data?.params.anomaly_window ?? 30}-day rolling window.
            </p>
          </div>
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b" style={{ borderColor: PANEL_BORDER }}>
                <th
                  className="text-left px-4 py-2.5 text-xs font-medium uppercase tracking-wider"
                  style={{ color: TEXT_DIM }}
                >
                  Day
                </th>
                <th
                  className="text-right px-4 py-2.5 text-xs font-medium uppercase tracking-wider"
                  style={{ color: TEXT_DIM }}
                >
                  Actual
                </th>
                <th
                  className="text-right px-4 py-2.5 text-xs font-medium uppercase tracking-wider"
                  style={{ color: TEXT_DIM }}
                >
                  Expected
                </th>
                <th
                  className="text-right px-4 py-2.5 text-xs font-medium uppercase tracking-wider"
                  style={{ color: TEXT_DIM }}
                >
                  z-score
                </th>
              </tr>
            </thead>
            <tbody className="divide-y" style={{ borderColor: PANEL_BORDER }}>
              {anomalies.slice(0, 20).map((a) => (
                <tr key={a.day}>
                  <td className="px-4 py-2 text-sm" style={{ color: TEXT_BRIGHT }}>
                    {a.day}
                  </td>
                  <td
                    className="px-4 py-2 text-sm text-right"
                    style={{ color: a.kind === 'spike' ? '#f87171' : '#fbbf24' }}
                  >
                    {formatUsd(a.total_cost_usd, 4)}
                  </td>
                  <td className="px-4 py-2 text-sm text-right" style={{ color: TEXT_DIM }}>
                    {formatUsd(a.expected_usd, 4)}
                  </td>
                  <td
                    className="px-4 py-2 text-sm text-right"
                    style={{ color: a.kind === 'spike' ? '#f87171' : '#fbbf24' }}
                  >
                    {a.z_score >= 0 ? '+' : ''}
                    {a.z_score.toFixed(2)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function KPI({
  label,
  value,
  color,
  loading,
}: {
  label: string;
  value: string;
  color: string;
  loading: boolean;
}) {
  return (
    <div
      className="rounded-lg p-4 border"
      style={{ background: PANEL_BG, borderColor: PANEL_BORDER }}
    >
      <div className="text-xs mb-1" style={{ color: TEXT_DIM }}>
        {label}
      </div>
      {loading ? (
        <Skeleton className="h-7 w-24" />
      ) : (
        <div className="text-xl font-bold" style={{ color }}>
          {value}
        </div>
      )}
    </div>
  );
}
