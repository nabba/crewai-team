import { useApi } from '../hooks/useApi';
import type { OrgChartAgent } from '../types/index.ts';

function Skeleton({ className = '' }: { className?: string }) {
  return <div className={`animate-pulse bg-[#1e2738] rounded ${className}`} />;
}

const STATUS_COLORS: Record<string, string> = {
  active: 'bg-[#34d399]',
  idle: 'bg-[#fbbf24]',
  offline: 'bg-[#7a8599]',
};

const ROLE_ICONS: Record<string, string> = {
  commander: '👑',
  researcher: '🔬',
  coder: '💻',
  writer: '✍️',
  media: '🎨',
  critic: '🎯',
  'self-improver': '🔄',
  introspector: '🧠',
  default: '🤖',
};

function getRoleIcon(role: string): string {
  const lower = role.toLowerCase();
  for (const [key, icon] of Object.entries(ROLE_ICONS)) {
    if (lower.includes(key)) return icon;
  }
  return ROLE_ICONS.default;
}

function AgentNode({ agent, depth = 0 }: { agent: OrgChartAgent & {children?: OrgChartAgent[]} ; depth?: number }) {
  const icon = getRoleIcon(agent.agent_role);
  const hasChildren = (agent as any).children && (agent as any).children.length > 0;

  return (
    <div className="flex flex-col items-center">
      {/* Node */}
      <div className="bg-[#111820] border border-[#1e2738] rounded-lg p-3 w-36 text-center hover:border-[#60a5fa]/40 transition-colors relative">
        <div className="text-2xl mb-1">{icon}</div>
        <div className="text-xs font-medium text-[#e2e8f0] truncate">{agent.display_name}</div>
        <div className="text-xs text-[#7a8599] capitalize mt-0.5 truncate">{agent.agent_role}</div>
        {agent.job_description && (
          <div className="text-[10px] text-[#7a8599] mt-1 truncate" title={agent.job_description}>
            {agent.job_description.slice(0, 30)}
          </div>
        )}
      </div>

      {/* Children */}
      {hasChildren && (
        <>
          {/* Vertical connector */}
          <div className="w-px h-6 bg-[#1e2738]" />

          {/* Horizontal line spanning children */}
          {agent.children!.length > 1 && (
            <div
              className="h-px bg-[#1e2738]"
              style={{ width: `${Math.min(agent.children!.length * 160, 900)}px` }}
            />
          )}

          {/* Children row */}
          <div className="flex gap-4 items-start">
            {(agent as any).children!.map((child: OrgChartAgent) => (
              <div key={child.agent_role} className="flex flex-col items-center">
                {agent.children!.length > 1 && <div className="w-px h-6 bg-[#1e2738]" />}
                <AgentNode agent={child} depth={depth + 1} />
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

function buildTree(agents: OrgChartAgent[]): (OrgChartAgent & {children: OrgChartAgent[]})[] {
  type TreeNode = OrgChartAgent & {children: OrgChartAgent[]};
  const map = new Map<string, TreeNode>();
  const roots: TreeNode[] = [];

  agents.forEach((a) => {
    map.set(a.agent_role, { ...a, children: [] });
  });

  map.forEach((agent) => {
    if (agent.reports_to && map.has(agent.reports_to)) {
      map.get(agent.reports_to)!.children.push(agent);
    } else {
      roots.push(agent);
    }
  });

  return roots;
}

// No fallback needed — API always returns org chart from PostgreSQL

export function OrgChart() {
  const { data: agents, loading, error } = useApi<OrgChartAgent[]>('/org-chart', 60000);

  const roots = agents ? buildTree(agents) : [];

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-semibold text-[#e2e8f0]">Org Chart</h1>
        <p className="text-sm text-[#7a8599] mt-1">Agent hierarchy and status</p>
      </div>

      {/* Legend */}
      <div className="flex items-center gap-4 text-xs text-[#7a8599]">
        <div className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-[#34d399]" />
          Active
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-[#fbbf24]" />
          Idle
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-[#7a8599]" />
          Offline
        </div>
      </div>

      <div className="bg-[#111820] border border-[#1e2738] rounded-lg p-6 overflow-x-auto">
        {loading ? (
          <div className="flex justify-center">
            <div className="space-y-4 text-center">
              <Skeleton className="h-24 w-36 mx-auto" />
              <div className="flex gap-4 justify-center">
                {Array.from({ length: 4 }).map((_, i) => (
                  <Skeleton key={i} className="h-20 w-32" />
                ))}
              </div>
            </div>
          </div>
        ) : error ? (
          <div className="text-center py-8 text-[#f87171] text-sm">{error}</div>
        ) : (
          <div className="flex flex-col items-center gap-0 min-w-max mx-auto">
            {roots.map((root) => (
              <AgentNode key={root.agent_role} agent={root} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
