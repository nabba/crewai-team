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
import { Line, Bar } from 'react-chartjs-2';
import { useApi } from '../hooks/useApi';
import type { CostEntry, AgentCost } from '../types/index.ts';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

function Skeleton({ className = '' }: { className?: string }) {
  return <div className={`animate-pulse bg-[#1e2738] rounded ${className}`} />;
}

const CHART_DEFAULTS = {
  plugins: {
    legend: {
      labels: { color: '#7a8599', font: { size: 12 } },
    },
    tooltip: {
      backgroundColor: '#111820',
      borderColor: '#1e2738',
      borderWidth: 1,
      titleColor: '#e2e8f0',
      bodyColor: '#7a8599',
    },
  },
  scales: {
    x: {
      ticks: { color: '#7a8599', font: { size: 11 } },
      grid: { color: '#1e2738' },
    },
    y: {
      ticks: { color: '#7a8599', font: { size: 11 } },
      grid: { color: '#1e2738' },
    },
  },
  responsive: true,
  maintainAspectRatio: false,
};

export function CostCharts() {
  const { data: dailyCosts, loading: dailyLoading, error: dailyError } = useApi<CostEntry[]>(
    '/costs/daily?days=30',
    60000
  );
  const { data: agentCostsRaw, loading: agentLoading, error: agentError } = useApi<{by_actor: AgentCost[], total_cost: number}>(
    '/costs/by-agent',
    60000
  );
  const agentCosts = agentCostsRaw?.by_actor ?? [];

  const dailyChartData = dailyCosts
    ? {
        labels: dailyCosts.map((c) => {
          const d = new Date(c.day);
          return `${d.getMonth() + 1}/${d.getDate()}`;
        }),
        datasets: [
          {
            label: 'Daily Cost ($)',
            data: dailyCosts.map((c) => c.total_cost ?? 0),
            borderColor: '#60a5fa',
            backgroundColor: 'rgba(96, 165, 250, 0.1)',
            pointBackgroundColor: '#60a5fa',
            pointRadius: 3,
            fill: true,
            tension: 0.4,
          },
        ],
      }
    : null;

  const agentChartData = agentCosts
    ? {
        labels: agentCosts.map((c) => c.actor ?? '?'),
        datasets: [
          {
            label: 'Total Cost ($)',
            data: agentCosts.map((c) => c.total_cost ?? 0),
            backgroundColor: [
              'rgba(96, 165, 250, 0.7)',
              'rgba(52, 211, 153, 0.7)',
              'rgba(167, 139, 250, 0.7)',
              'rgba(251, 191, 36, 0.7)',
              'rgba(248, 113, 113, 0.7)',
              'rgba(251, 146, 60, 0.7)',
              'rgba(34, 211, 238, 0.7)',
              'rgba(244, 114, 182, 0.7)',
            ],
            borderColor: [
              '#60a5fa',
              '#34d399',
              '#a78bfa',
              '#fbbf24',
              '#f87171',
              '#fb923c',
              '#22d3ee',
              '#f472b6',
            ],
            borderWidth: 1,
            borderRadius: 4,
          },
        ],
      }
    : null;

  // Summary stats from agent costs
  const totalCost = agentCosts?.reduce((s, c) => s + (c.total_cost ?? 0), 0) ?? 0;
  const topAgent = agentCosts?.length ? agentCosts.reduce(
    (top, c) => ((c.total_cost ?? 0) > (top.total_cost ?? 0) ? c : top),
    agentCosts[0]
  ) : null;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold text-[#e2e8f0]">Cost Analytics</h1>
        <p className="text-sm text-[#7a8599] mt-1">Spending trends and per-agent breakdown</p>
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="bg-[#111820] border border-[#1e2738] rounded-lg p-4">
          <div className="text-xs text-[#7a8599] mb-1">Total Spend (all time)</div>
          {agentLoading ? (
            <Skeleton className="h-7 w-24" />
          ) : (
            <div className="text-xl font-bold text-[#34d399]">${totalCost.toFixed(4)}</div>
          )}
        </div>
        <div className="bg-[#111820] border border-[#1e2738] rounded-lg p-4">
          <div className="text-xs text-[#7a8599] mb-1">Top Spending Agent</div>
          {agentLoading ? (
            <Skeleton className="h-7 w-24" />
          ) : (
            <div className="text-xl font-bold text-[#60a5fa]">
              {topAgent?.actor ?? '—'}
            </div>
          )}
        </div>
        <div className="bg-[#111820] border border-[#1e2738] rounded-lg p-4">
          <div className="text-xs text-[#7a8599] mb-1">Active Agents</div>
          {agentLoading ? (
            <Skeleton className="h-7 w-16" />
          ) : (
            <div className="text-xl font-bold text-[#a78bfa]">
              {agentCosts?.length ?? 0}
            </div>
          )}
        </div>
      </div>

      {/* Daily line chart */}
      <div className="bg-[#111820] border border-[#1e2738] rounded-lg p-4">
        <h2 className="text-sm font-medium text-[#e2e8f0] mb-4">Daily Costs — Last 30 Days</h2>
        {dailyLoading ? (
          <Skeleton className="h-64" />
        ) : dailyError ? (
          <div className="h-64 flex items-center justify-center text-[#f87171] text-sm">
            {dailyError}
          </div>
        ) : !dailyChartData || dailyCosts?.length === 0 ? (
          <div className="h-64 flex items-center justify-center text-[#7a8599] text-sm">
            No cost data available.
          </div>
        ) : (
          <div className="h-64">
            <Line
              data={dailyChartData}
              options={{
                ...CHART_DEFAULTS,
                plugins: {
                  ...CHART_DEFAULTS.plugins,
                  legend: { display: false },
                  tooltip: {
                    ...CHART_DEFAULTS.plugins.tooltip,
                    callbacks: {
                      label: (ctx) => ` $${(ctx.raw as number).toFixed(4)}`,
                    },
                  },
                },
              }}
            />
          </div>
        )}
      </div>

      {/* Agent bar chart */}
      <div className="bg-[#111820] border border-[#1e2738] rounded-lg p-4">
        <h2 className="text-sm font-medium text-[#e2e8f0] mb-4">Cost by Agent</h2>
        {agentLoading ? (
          <Skeleton className="h-64" />
        ) : agentError ? (
          <div className="h-64 flex items-center justify-center text-[#f87171] text-sm">
            {agentError}
          </div>
        ) : !agentChartData || agentCosts?.length === 0 ? (
          <div className="h-64 flex items-center justify-center text-[#7a8599] text-sm">
            No agent cost data available.
          </div>
        ) : (
          <div className="h-64">
            <Bar
              data={agentChartData}
              options={{
                ...CHART_DEFAULTS,
                plugins: {
                  ...CHART_DEFAULTS.plugins,
                  legend: { display: false },
                  tooltip: {
                    ...CHART_DEFAULTS.plugins.tooltip,
                    callbacks: {
                      label: (ctx) => ` $${(ctx.raw as number).toFixed(4)}`,
                    },
                  },
                },
                scales: {
                  ...CHART_DEFAULTS.scales,
                  x: {
                    ...CHART_DEFAULTS.scales.x,
                    ticks: {
                      ...CHART_DEFAULTS.scales.x.ticks,
                      maxRotation: 45,
                    },
                  },
                },
              }}
            />
          </div>
        )}
      </div>

      {/* Agent cost table */}
      {!agentLoading && agentCosts && agentCosts.length > 0 && (
        <div className="bg-[#111820] border border-[#1e2738] rounded-lg overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-[#1e2738]">
                <th className="text-left px-4 py-3 text-xs font-medium text-[#7a8599] uppercase tracking-wider">
                  Agent
                </th>
                <th className="text-right px-4 py-3 text-xs font-medium text-[#7a8599] uppercase tracking-wider">
                  Total Cost
                </th>
                <th className="text-right px-4 py-3 text-xs font-medium text-[#7a8599] uppercase tracking-wider">
                  Calls
                </th>
                <th className="text-right px-4 py-3 text-xs font-medium text-[#7a8599] uppercase tracking-wider">
                  Avg/Call
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#1e2738]">
              {[...agentCosts]
                .sort((a, b) => b.total - a.total)
                .map((ac) => (
                  <tr key={ac.agent} className="hover:bg-[#1e2738]/50 transition-colors">
                    <td className="px-4 py-2.5 text-[#e2e8f0]">{ac.agent}</td>
                    <td className="px-4 py-2.5 text-right text-[#34d399]">
                      ${ac.total.toFixed(4)}
                    </td>
                    <td className="px-4 py-2.5 text-right text-[#7a8599]">
                      {ac.count ?? '—'}
                    </td>
                    <td className="px-4 py-2.5 text-right text-[#7a8599]">
                      {ac.count && ac.count > 0
                        ? `$${(ac.total / ac.count).toFixed(4)}`
                        : '—'}
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
