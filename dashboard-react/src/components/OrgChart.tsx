import type { OrgChartAgent } from '../types';
import { Skeleton } from './ui/Skeleton';
import { ErrorPanel } from './ui/ErrorPanel';
import {
  useOrgChartQuery,
  useDelegationSettingsQuery,
  useSetDelegationSetting,
  useMetaAgentSettingsQuery,
  useSetMetaAgentSetting,
} from '../api/queries';
import { CREW_REGISTRY, crewMeta, type CrewKind } from '../crews';

// Map the canonical crew name to the PostgreSQL org_chart agent role. When
// the mapping exists, the crew is already represented on the chart via its
// agent; entries without a match get synthesised as crew-only nodes under
// commander so every crew the registry knows about is visible.
const CREW_TO_AGENT_ROLE: Record<string, string> = {
  commander: 'commander',
  research: 'researcher',
  coding: 'coder',
  writing: 'writer',
  media: 'media_analyst',
  critic: 'critic',
  self_improvement: 'self_improver',
};

// Per-role fallback icons for the org-chart agents (researcher, coder, ...).
// The registry supplies icons for crew names (research, coding, ...), but the
// org_chart rows use agent role names. This table covers the ones that don't
// share a name with a crew.
const AGENT_ROLE_ICONS: Record<string, string> = {
  commander: '🧭',
  researcher: '🔬',
  coder: '💻',
  writer: '✍️',
  media_analyst: '📸',
  critic: '🎯',
  self_improver: '🧠',
  introspector: '🪞',
};

type NodeBadge = { label: string; cls: string };

interface ChartNode extends OrgChartAgent {
  children: ChartNode[];
  _kind?: CrewKind | 'agent';
  _synthetic?: boolean;
  _icon: string;
}

function iconFor(role: string): string {
  const byAgent = AGENT_ROLE_ICONS[role.toLowerCase()];
  if (byAgent) return byAgent;
  // The registry handles crew names — falls back to 🤖 on unknown.
  return crewMeta(role).icon;
}

function buildMergedTree(agents: OrgChartAgent[]): ChartNode[] {
  const map = new Map<string, ChartNode>();

  // 1. Real org_chart rows first — they carry the authoritative hierarchy
  //    and the human-written job_description.
  for (const a of agents) {
    map.set(a.agent_role, {
      ...a,
      children: [],
      _icon: iconFor(a.agent_role),
      _kind: 'agent',
    });
  }

  // 2. Mark which crews are already represented via an agent row.
  const represented = new Set<string>();
  for (const [crew, role] of Object.entries(CREW_TO_AGENT_ROLE)) {
    if (map.has(role)) represented.add(crew);
  }

  // 3. Synthesise a node under commander for every crew in the registry that
  //    doesn't map to an org_chart row. This fills the 7 missing user crews
  //    (creative, pim, financial, desktop, repo_analysis, devops, tech_radar)
  //    plus the retrospective internal crew.
  let synthOrder = 1000;
  for (const meta of CREW_REGISTRY) {
    if (represented.has(meta.name)) continue;
    if (map.has(meta.name)) continue; // already covered, e.g. critic
    map.set(meta.name, {
      agent_role: meta.name,
      display_name: meta.label,
      reports_to: 'commander',
      job_description: meta.description,
      soul_file: '',
      default_model: '',
      sort_order: synthOrder++,
      children: [],
      _kind: meta.kind,
      _synthetic: true,
      _icon: meta.icon,
    });
  }

  // 4. Wire children. Anything reporting to something we know about becomes
  //    a child; everything else is a root.
  const roots: ChartNode[] = [];
  const sortKey = (n: ChartNode) => n.sort_order ?? 9999;
  map.forEach((node) => {
    if (node.reports_to && map.has(node.reports_to)) {
      map.get(node.reports_to)!.children.push(node);
    } else {
      roots.push(node);
    }
  });

  // Stable order — agent rows first (low sort_order), synthetic crews last.
  map.forEach((n) => n.children.sort((a, b) => sortKey(a) - sortKey(b)));
  roots.sort((a, b) => sortKey(a) - sortKey(b));
  return roots;
}

function NodeBadgeView({ badge }: { badge: NodeBadge }) {
  return (
    <span className={`text-[10px] px-1.5 py-0.5 rounded border uppercase tracking-wider font-medium ${badge.cls}`}>
      {badge.label}
    </span>
  );
}

function badgesForNode(node: ChartNode): NodeBadge[] {
  const out: NodeBadge[] = [];
  if (node._synthetic) {
    const color = node._kind === 'internal'
      ? 'border-[#a78bfa]/30 bg-[#a78bfa]/10 text-[#a78bfa]'
      : 'border-[#60a5fa]/30 bg-[#60a5fa]/10 text-[#60a5fa]';
    out.push({ label: node._kind === 'internal' ? 'internal crew' : 'crew', cls: color });
  } else {
    out.push({ label: 'agent', cls: 'border-[#7a8599]/30 bg-[#7a8599]/10 text-[#7a8599]' });
  }
  return out;
}

function AgentNode({ node, depth }: { node: ChartNode; depth: number }) {
  const hasChildren = node.children.length > 0;
  const badges = badgesForNode(node);

  return (
    <div className="space-y-2">
      <div className="flex items-start gap-3 p-3 bg-[#111820] border border-[#1e2738] rounded-lg hover:border-[#60a5fa]/40 transition-colors">
        <span className="text-xl sm:text-2xl leading-none flex-shrink-0">{node._icon}</span>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-sm font-semibold text-[#e2e8f0]">{node.display_name}</span>
            {badges.map((b) => <NodeBadgeView key={b.label} badge={b} />)}
          </div>
          <div className="text-[11px] text-[#7a8599] mt-0.5 font-mono">
            {node.agent_role}
            {node.reports_to && (
              <span className="text-[#7a8599]/70"> · reports to {node.reports_to}</span>
            )}
          </div>
          {node.job_description && (
            <p className="text-xs text-[#7a8599] mt-1 leading-snug">{node.job_description}</p>
          )}
          {node.default_model && !node._synthetic && (
            <div className="text-[10px] text-[#7a8599]/80 mt-1 font-mono truncate">
              default model: {node.default_model}
            </div>
          )}
        </div>
      </div>

      {hasChildren && (
        <div className="pl-3 sm:pl-5 border-l-2 border-[#1e2738] space-y-2">
          {node.children.map((child) => (
            <AgentNode key={child.agent_role} node={child} depth={depth + 1} />
          ))}
        </div>
      )}
    </div>
  );
}

// ── Delegation Mode toggles ─────────────────────────────────────────────────
// When ON for a crew, tasks dispatch to Coordinator + specialists instead of
// a single monolithic agent.  Preserves the full tool palette on providers
// with tight tool limits (Anthropic).

const DELEGATION_DESCRIPTIONS: Record<string, string> = {
  research:
    'Research crew → Coordinator + Web + Document + Synthesis specialists. Each sub-agent keeps ≤ 18 tools so Anthropic strict-mode works. ~2× LLM calls, full tool palette preserved.',
  coding:
    'Coding crew → Coordinator + Execution + Debug specialists. Coordinator writes the code, Execution runs it in the sandbox, Debug diagnoses failures from journal/tensions. Great for multi-step debugging.',
  writing:
    'Writing crew → Coordinator + Research + Synthesis specialists. Research gathers facts into a brief, Synthesis produces the finished prose with dialectics/philosophy. Best for longer substantive pieces.',
};

function DelegationPanel() {
  const { data, isLoading } = useDelegationSettingsQuery();
  const setMut = useSetDelegationSetting();
  const settings = data?.settings ?? {};

  return (
    <div className="bg-[#111820] border border-[#1e2738] rounded-lg p-5">
      <div className="mb-3">
        <h2 className="text-base font-semibold text-[#e2e8f0]">Delegation Mode</h2>
        <p className="text-xs text-[#7a8599] mt-1">
          Split each crew into a Coordinator + specialist sub-agents. Every agent stays
          under any provider's tool limit. Tradeoff: ~2× LLM calls per task.
        </p>
      </div>

      {isLoading ? (
        <Skeleton className="h-24" />
      ) : (
        <div className="space-y-2">
          {Object.entries(settings).map(([crew, enabled]) => {
            const desc = DELEGATION_DESCRIPTIONS[crew] ?? '';
            const pending = setMut.isPending && setMut.variables?.crew === crew;
            return (
              <div
                key={crew}
                className={`flex items-start gap-3 p-3 rounded border ${
                  enabled
                    ? 'bg-[#34d399]/5 border-[#34d399]/30'
                    : 'bg-[#0a0e14] border-[#1e2738]'
                }`}
              >
                <button
                  disabled={pending}
                  onClick={() => setMut.mutate({ crew, enabled: !enabled })}
                  className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors flex-shrink-0 ${
                    enabled ? 'bg-[#34d399]' : 'bg-[#1e2738]'
                  } ${pending ? 'cursor-not-allowed opacity-60' : 'cursor-pointer'}`}
                  aria-label={`Toggle delegation for ${crew}`}
                >
                  <span
                    className={`inline-block h-4 w-4 rounded-full bg-white transition-transform ${
                      enabled ? 'translate-x-6' : 'translate-x-1'
                    }`}
                  />
                </button>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium text-[#e2e8f0] capitalize">
                      {crew}
                    </span>
                    <span
                      className={`text-[10px] px-1.5 py-0.5 rounded font-medium ${
                        enabled
                          ? 'bg-[#34d399]/20 text-[#34d399]'
                          : 'bg-[#1e2738] text-[#7a8599]'
                      }`}
                    >
                      {enabled ? 'DELEGATION ON' : 'SINGLE AGENT'}
                    </span>
                  </div>
                  {desc && <p className="text-xs text-[#7a8599] mt-1">{desc}</p>}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {setMut.isError && (
        <p className="text-xs text-[#f87171] mt-2">
          Failed to toggle: {(setMut.error as Error).message}
        </p>
      )}
    </div>
  );
}

// ── Meta-Agent toggles ──────────────────────────────────────────────────────
// When ON, run_single_agent_crew routes through the meta-agent selector —
// picks a learned recipe (force_tier × extra_tools × task_hint) from
// cross-run history and applies it as a bounded augmentation on top of the
// existing agent factory.  Orthogonal to delegation; meta-agent fires only
// on the single-agent dispatch path.

const META_AGENT_DESCRIPTIONS: Record<string, string> = {
  research:
    'Learn which research recipe (LLM tier, task hint, extra tools) works best for similar topics across runs. UCB1 + similarity selection. Cold-start prefers factory defaults until the baseline arm has enough evidence.',
  coding:
    'Learn which coding recipe wins for similar tasks (e.g. premium tier for refactors, local tier for one-shot scripts). Bounded — never edits backstory or LLM rules.',
  writing:
    'Learn which writing recipe fits the task type (briefer prompt for posts, premium tier + extended deadline for long-form). Recipes never replace the writer factory; they augment it.',
};

function MetaAgentPanel() {
  const { data, isLoading, error } = useMetaAgentSettingsQuery();
  const setMut = useSetMetaAgentSetting();
  const settings = data?.settings ?? {};
  const masterEnvOn = data?.master_env_on ?? false;
  const envOverrides = data?.env_overrides ?? {};
  const noSettings = !isLoading && !error && Object.keys(settings).length === 0;

  return (
    <div className="bg-[#111820] border border-[#1e2738] rounded-lg p-5">
      <div className="mb-3">
        <div className="flex items-center gap-2">
          <h2 className="text-base font-semibold text-[#e2e8f0]">Meta-Agent</h2>
          <span className="text-[10px] px-1.5 py-0.5 rounded font-medium bg-[#3b82f6]/20 text-[#60a5fa]">
            EXPERIMENTAL
          </span>
        </div>
        <p className="text-xs text-[#7a8599] mt-1">
          Cross-run recipe learning over force_tier × extra_tools × task_hint.
          Bounded augmentation on top of the agent factory — never replaces it.
          Orthogonal to delegation: only fires on the single-agent dispatch path.
        </p>
        {masterEnvOn && (
          <p className="text-xs text-[#fbbf24] mt-2">
            <strong>META_AGENT=1</strong> env var is set — all crews are forced
            ON regardless of these toggles.
          </p>
        )}
      </div>

      {isLoading ? (
        <Skeleton className="h-24" />
      ) : error ? (
        <p className="text-xs text-[#f87171]">
          Endpoint unavailable: <code>{(error as Error).message}</code>.
          Restart the gateway to pick up the new <code>/api/cp/meta-agent</code> route.
        </p>
      ) : noSettings ? (
        <p className="text-xs text-[#7a8599] italic">
          No crews configured. The gateway may need to be restarted to register
          the new endpoint.
        </p>
      ) : (
        <div className="space-y-2">
          {Object.entries(settings).map(([crew, enabled]) => {
            const desc = META_AGENT_DESCRIPTIONS[crew] ?? '';
            const pending = setMut.isPending && setMut.variables?.crew === crew;
            const envOverride = envOverrides[crew];
            const hasEnvOverride = envOverride !== undefined;
            // Effective state: env override wins; otherwise master env wins;
            // otherwise the JSON toggle.
            const effectivelyOn = hasEnvOverride
              ? envOverride === '1'
              : masterEnvOn || enabled;
            return (
              <div
                key={crew}
                className={`flex items-start gap-3 p-3 rounded border ${
                  effectivelyOn
                    ? 'bg-[#60a5fa]/5 border-[#60a5fa]/30'
                    : 'bg-[#0a0e14] border-[#1e2738]'
                }`}
              >
                <button
                  disabled={pending || hasEnvOverride}
                  onClick={() => setMut.mutate({ crew, enabled: !enabled })}
                  className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors flex-shrink-0 ${
                    enabled ? 'bg-[#60a5fa]' : 'bg-[#1e2738]'
                  } ${
                    pending || hasEnvOverride
                      ? 'cursor-not-allowed opacity-60'
                      : 'cursor-pointer'
                  }`}
                  aria-label={`Toggle meta-agent for ${crew}`}
                  title={
                    hasEnvOverride
                      ? `Locked by env: META_AGENT_${crew.toUpperCase()}=${envOverride}`
                      : undefined
                  }
                >
                  <span
                    className={`inline-block h-4 w-4 rounded-full bg-white transition-transform ${
                      enabled ? 'translate-x-6' : 'translate-x-1'
                    }`}
                  />
                </button>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="text-sm font-medium text-[#e2e8f0] capitalize">
                      {crew}
                    </span>
                    <span
                      className={`text-[10px] px-1.5 py-0.5 rounded font-medium ${
                        effectivelyOn
                          ? 'bg-[#60a5fa]/20 text-[#60a5fa]'
                          : 'bg-[#1e2738] text-[#7a8599]'
                      }`}
                    >
                      {effectivelyOn ? 'META-AGENT ON' : 'FACTORY DEFAULTS'}
                    </span>
                    {hasEnvOverride && (
                      <span
                        className="text-[10px] px-1.5 py-0.5 rounded font-medium bg-[#fbbf24]/20 text-[#fbbf24]"
                        title={`META_AGENT_${crew.toUpperCase()}=${envOverride}`}
                      >
                        ENV LOCK
                      </span>
                    )}
                  </div>
                  {desc && <p className="text-xs text-[#7a8599] mt-1">{desc}</p>}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {setMut.isError && (
        <p className="text-xs text-[#f87171] mt-2">
          Failed to toggle: {(setMut.error as Error).message}
        </p>
      )}
    </div>
  );
}

export function OrgChart() {
  const { data: agents, isLoading, error, refetch } = useOrgChartQuery();
  const roots = agents ? buildMergedTree(agents) : [];

  // Flat count for the header.
  let total = 0;
  const count = (n: ChartNode) => { total += 1; n.children.forEach(count); };
  roots.forEach(count);

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-semibold text-[#e2e8f0]">Org Chart</h1>
        <p className="text-sm text-[#7a8599] mt-1">
          {total ? `${total} roles · ` : ''}agents from the PostgreSQL org chart plus every crew from the registry.
          Crews without a dedicated agent row are shown under <code>commander</code> as synthetic nodes.
        </p>
      </div>

      <div className="bg-[#111820] border border-[#1e2738] rounded-lg p-3 sm:p-4">
        {isLoading ? (
          <div className="space-y-3">
            <Skeleton className="h-14" />
            <div className="pl-4 border-l-2 border-[#1e2738] space-y-3">
              {Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-14" />)}
            </div>
          </div>
        ) : error ? (
          <ErrorPanel error={error} onRetry={refetch} />
        ) : roots.length === 0 ? (
          <p className="text-sm text-[#7a8599] italic">No agents or crews registered.</p>
        ) : (
          <div className="space-y-2">
            {roots.map((root) => (
              <AgentNode key={root.agent_role} node={root} depth={0} />
            ))}
          </div>
        )}
      </div>

      <DelegationPanel />

      <MetaAgentPanel />
    </div>
  );
}
