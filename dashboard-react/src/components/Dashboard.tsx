import { useApi } from '../hooks/useApi';
import { useProject } from '../context/ProjectContext';
import type { Ticket, Budget, AuditEntry, GovernanceRequest, HealthStatus } from '../types/index.ts';

function StatCard({
  label,
  value,
  color,
  icon,
}: {
  label: string;
  value: number | string;
  color: string;
  icon: string;
}) {
  return (
    <div className="bg-[#111820] border border-[#1e2738] rounded-lg p-4 flex items-center gap-4">
      <div className={`text-2xl w-10 h-10 flex items-center justify-center rounded-lg bg-[#1e2738]`}>
        {icon}
      </div>
      <div>
        <div className={`text-xl font-bold ${color}`}>{value}</div>
        <div className="text-xs text-[#7a8599] mt-0.5">{label}</div>
      </div>
    </div>
  );
}

function Skeleton({ className = '' }: { className?: string }) {
  return <div className={`animate-pulse bg-[#1e2738] rounded ${className}`} />;
}

export function Dashboard() {
  const { activeProject } = useProject();
  const projectParam = activeProject ? `?project_id=${activeProject.id}` : '';

  const { data: tickets, loading: ticketsLoading } = useApi<Ticket[]>(
    `/tickets${projectParam}`,
    30000
  );
  const { data: budgets, loading: budgetsLoading } = useApi<Budget[]>(
    `/budgets${projectParam}`,
    30000
  );
  const { data: auditEntries, loading: auditLoading } = useApi<AuditEntry[]>(
    '/audit?limit=10',
    10000
  );
  const { data: governance, loading: govLoading } = useApi<GovernanceRequest[]>(
    '/governance/pending',
    15000
  );
  const { data: health } = useApi<HealthStatus>('/health', 30000);

  // Ticket counts
  const ticketCounts = {
    todo: tickets?.filter((t) => t.status === 'todo').length ?? 0,
    in_progress: tickets?.filter((t) => t.status === 'in_progress').length ?? 0,
    review: tickets?.filter((t) => t.status === 'review').length ?? 0,
    done: tickets?.filter((t) => t.status === 'done').length ?? 0,
    failed: tickets?.filter((t) => t.status === 'failed').length ?? 0,
  };

  // Budget totals
  const totalSpent = budgets?.reduce((s, b) => s + (b.spent_usd || 0), 0) ?? 0;
  const totalLimit = budgets?.reduce((s, b) => s + (b.limit_usd || 0), 0) ?? 0;
  const budgetPct = totalLimit > 0 ? Math.round((totalSpent / totalLimit) * 100) : 0;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold text-[#e2e8f0]">Dashboard</h1>
        <p className="text-sm text-[#7a8599] mt-1">
          {activeProject ? activeProject.name : 'All projects overview'}
        </p>
      </div>

      {/* Health banner */}
      {health && health.status !== 'ok' && (
        <div
          className={`p-3 rounded-lg border text-sm ${
            health.status === 'degraded'
              ? 'bg-[#fbbf24]/10 border-[#fbbf24]/30 text-[#fbbf24]'
              : 'bg-[#f87171]/10 border-[#f87171]/30 text-[#f87171]'
          }`}
        >
          System status: <strong className="capitalize">{health.status}</strong>
        </div>
      )}

      {/* Ticket status cards */}
      <section>
        <h2 className="text-sm font-medium text-[#7a8599] uppercase tracking-wider mb-3">
          Ticket Status
        </h2>
        {ticketsLoading ? (
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
            {Array.from({ length: 5 }).map((_, i) => (
              <Skeleton key={i} className="h-20" />
            ))}
          </div>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
            <StatCard label="To Do" value={ticketCounts.todo} color="text-[#e2e8f0]" icon="📋" />
            <StatCard label="In Progress" value={ticketCounts.in_progress} color="text-[#60a5fa]" icon="⚙️" />
            <StatCard label="Review" value={ticketCounts.review} color="text-[#a78bfa]" icon="🔍" />
            <StatCard label="Done" value={ticketCounts.done} color="text-[#34d399]" icon="✅" />
            <StatCard label="Failed" value={ticketCounts.failed} color="text-[#f87171]" icon="❌" />
          </div>
        )}
      </section>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Budget overview */}
        <section>
          <h2 className="text-sm font-medium text-[#7a8599] uppercase tracking-wider mb-3">
            Budget Overview
          </h2>
          <div className="bg-[#111820] border border-[#1e2738] rounded-lg p-4 space-y-4">
            {budgetsLoading ? (
              <div className="space-y-3">
                {Array.from({ length: 4 }).map((_, i) => (
                  <Skeleton key={i} className="h-8" />
                ))}
              </div>
            ) : !budgets || budgets.length === 0 ? (
              <p className="text-sm text-[#7a8599]">No budgets configured.</p>
            ) : (
              <>
                <div className="flex items-center justify-between text-sm mb-2">
                  <span className="text-[#7a8599]">Total spent</span>
                  <span className="text-[#e2e8f0] font-medium">
                    ${totalSpent.toFixed(4)} / ${totalLimit.toFixed(4)}
                  </span>
                </div>
                <div className="w-full bg-[#1e2738] rounded-full h-2 mb-4">
                  <div
                    className={`h-2 rounded-full transition-all ${
                      budgetPct > 85
                        ? 'bg-[#f87171]'
                        : budgetPct > 60
                        ? 'bg-[#fbbf24]'
                        : 'bg-[#34d399]'
                    }`}
                    style={{ width: `${Math.min(budgetPct, 100)}%` }}
                  />
                </div>
                {budgets.slice(0, 5).map((b, i) => {
                  const pct = b.limit_usd > 0 ? Math.min(((b.spent_usd || 0) / b.limit_usd) * 100, 100) : 0;
                  return (
                    <div key={b.agent_role || i} className="space-y-1">
                      <div className="flex justify-between text-xs">
                        <span className="text-[#e2e8f0] flex items-center gap-1.5">
                          {b.is_paused && (
                            <span className="px-1.5 py-0.5 rounded bg-[#f87171]/20 text-[#f87171] text-[10px]">
                              PAUSED
                            </span>
                          )}
                          {b.agent_role || 'project-wide'}
                        </span>
                        <span className="text-[#7a8599]">
                          ${(b.spent_usd || 0).toFixed(4)} / ${(b.limit_usd || 0).toFixed(4)}
                        </span>
                      </div>
                      <div className="w-full bg-[#1e2738] rounded-full h-1.5">
                        <div
                          className={`h-1.5 rounded-full ${
                            pct > 85 ? 'bg-[#f87171]' : pct > 60 ? 'bg-[#fbbf24]' : 'bg-[#34d399]'
                          }`}
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
                {budgets.length > 5 && (
                  <p className="text-xs text-[#7a8599]">+{budgets.length - 5} more agents</p>
                )}
              </>
            )}
          </div>
        </section>

        {/* Governance */}
        <section>
          <h2 className="text-sm font-medium text-[#7a8599] uppercase tracking-wider mb-3">
            Governance Queue
          </h2>
          <div className="bg-[#111820] border border-[#1e2738] rounded-lg p-4">
            {govLoading ? (
              <div className="space-y-3">
                {Array.from({ length: 3 }).map((_, i) => (
                  <Skeleton key={i} className="h-12" />
                ))}
              </div>
            ) : !governance || governance.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-8 text-[#7a8599]">
                <span className="text-3xl mb-2">✅</span>
                <p className="text-sm">No pending requests</p>
              </div>
            ) : (
              <div className="space-y-3">
                <div className="flex items-center gap-2 text-sm mb-2">
                  <span className="px-2 py-0.5 rounded-full bg-[#fbbf24]/20 text-[#fbbf24] font-medium">
                    {governance.length} pending
                  </span>
                </div>
                {governance.slice(0, 5).map((req) => (
                  <div
                    key={req.id}
                    className="flex items-start gap-3 p-3 rounded-lg bg-[#1e2738] border border-[#1e2738]"
                  >
                    <span className="text-[#fbbf24] text-sm mt-0.5">⚖️</span>
                    <div className="flex-1 min-w-0">
                      <div className="text-sm text-[#e2e8f0] truncate">{req.title}</div>
                      <div className="text-xs text-[#7a8599] mt-0.5">
                        {req.type} · by {req.requested_by}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </section>
      </div>

      {/* Recent audit */}
      <section>
        <h2 className="text-sm font-medium text-[#7a8599] uppercase tracking-wider mb-3">
          Recent Activity
        </h2>
        <div className="bg-[#111820] border border-[#1e2738] rounded-lg overflow-hidden">
          {auditLoading ? (
            <div className="p-4 space-y-2">
              {Array.from({ length: 5 }).map((_, i) => (
                <Skeleton key={i} className="h-8" />
              ))}
            </div>
          ) : !auditEntries || auditEntries.length === 0 ? (
            <div className="p-8 text-center text-[#7a8599] text-sm">No activity yet.</div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-[#1e2738]">
                    <th className="text-left px-4 py-3 text-xs font-medium text-[#7a8599] uppercase tracking-wider">
                      Time
                    </th>
                    <th className="text-left px-4 py-3 text-xs font-medium text-[#7a8599] uppercase tracking-wider">
                      Actor
                    </th>
                    <th className="text-left px-4 py-3 text-xs font-medium text-[#7a8599] uppercase tracking-wider">
                      Action
                    </th>
                    <th className="text-left px-4 py-3 text-xs font-medium text-[#7a8599] uppercase tracking-wider">
                      Resource
                    </th>
                    <th className="text-right px-4 py-3 text-xs font-medium text-[#7a8599] uppercase tracking-wider">
                      Cost
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#1e2738]">
                  {auditEntries.map((entry) => (
                    <tr key={entry.id} className="hover:bg-[#1e2738]/50 transition-colors">
                      <td className="px-4 py-2.5 text-[#7a8599] whitespace-nowrap text-xs">
                        {entry.timestamp ? new Date(entry.timestamp).toLocaleTimeString() : '—'}
                      </td>
                      <td className="px-4 py-2.5 text-[#60a5fa] whitespace-nowrap">{entry.actor}</td>
                      <td className="px-4 py-2.5 text-[#e2e8f0] whitespace-nowrap">{entry.action}</td>
                      <td className="px-4 py-2.5 text-[#7a8599] truncate max-w-[200px]">
                        {entry.resource_type ?? '—'}
                      </td>
                      <td className="px-4 py-2.5 text-right text-[#34d399] whitespace-nowrap">
                        {entry.cost_usd != null ? `$${entry.cost_usd.toFixed(4)}` : '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </section>
    </div>
  );
}
