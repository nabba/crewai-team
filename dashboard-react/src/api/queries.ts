import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from './client';
import { endpoints } from './endpoints';
import type {
  Project,
  Ticket,
  Budget,
  AuditEntry,
  GovernanceRequest,
  OrgChartAgent,
  HealthStatus,
  CostEntry,
  AgentCost,
  KanbanBoard,
  WorkspaceList,
  WorkspaceItems,
  MetaWorkspace,
} from '../types';

const POLL = {
  fast: 5_000,
  normal: 10_000,
  slow: 15_000,
  verySlow: 30_000,
  oneMin: 60_000,
} as const;

export const keys = {
  projects: ['projects'] as const,
  tickets: (projectId?: string) => ['tickets', projectId ?? 'all'] as const,
  ticketsBoard: (projectId?: string) => ['tickets', 'board', projectId ?? 'all'] as const,
  budgets: (projectId?: string) => ['budgets', projectId ?? 'all'] as const,
  audit: (limit: number) => ['audit', limit] as const,
  governancePending: ['governance', 'pending'] as const,
  orgChart: ['org-chart'] as const,
  costsDaily: (days: number) => ['costs', 'daily', days] as const,
  costsByAgent: ['costs', 'by-agent'] as const,
  costsByCrew: ['costs', 'by-crew'] as const,
  costsByInternalAgent: ['costs', 'by-internal-agent'] as const,
  health: ['health'] as const,
  evolutionSummary: ['evolution', 'summary'] as const,
  evolutionResults: (engine: string, status: string) => ['evolution', 'results', engine, status] as const,
  evolutionEngine: ['evolution', 'engine'] as const,
  workspaces: ['workspaces'] as const,
  workspaceItems: (projectId: string) => ['workspaces', projectId, 'items'] as const,
  workspacesMeta: ['workspaces', 'meta'] as const,
  creativeMode: ['creative-mode'] as const,
  kbStats: (kind: string) => ['kb', 'stats', kind] as const,
  kbBusinesses: ['kb', 'businesses'] as const,
  consciousness: (limit: number) => ['consciousness', limit] as const,
  tokens: ['tokens'] as const,
  crewTasks: (limit: number) => ['crew-tasks', limit] as const,
  errors: (limit: number) => ['errors', limit] as const,
  anomalies: (limit: number) => ['anomalies', limit] as const,
  deploys: (limit: number) => ['deploys', limit] as const,
  errorAudit: ['error-audit'] as const,
  techRadar: (limit: number) => ['tech-radar', limit] as const,
  snapshotKinds: ['snapshots', 'kinds'] as const,
  snapshotLatest: (kind: string) => ['snapshots', kind, 'latest'] as const,
  snapshotRecent: (kind: string, limit: number) => ['snapshots', kind, 'recent', limit] as const,
  llmMode: ['llms', 'mode'] as const,
  llmCatalog: ['llms', 'catalog'] as const,
  llmRoles: ['llms', 'roles'] as const,
  llmDiscovery: (limit: number) => ['llms', 'discovery', limit] as const,
  llmPromotions: ['llms', 'promotions'] as const,
  llmPins: ['llms', 'pins'] as const,
  evolutionVariants: (n: number) => ['evolution', 'variants', n] as const,
  evolutionVariantLineage: (id: string) => ['evolution', 'variant-lineage', id] as const,
  notesRoots: ['notes', 'roots'] as const,
  notesTree: (root: string) => ['notes', 'tree', root] as const,
  notesFile: (root: string, path: string) => ['notes', 'file', root, path] as const,
  notesGraph: (root: string) => ['notes', 'graph', root] as const,
  notesSearch: (root: string, q: string) => ['notes', 'search', root, q] as const,
  notesTags: (root: string) => ['notes', 'tags', root] as const,
  runtimeSettings: ['runtime-settings'] as const,
  backgroundTasks: ['background-tasks'] as const,
  chatMessages: (sender: string, limit: number) =>
    ['chat', 'messages', sender, limit] as const,
  signalCommands: ['signal-commands'] as const,
  systemStatus: ['system-status'] as const,
  webPushSubscriptions: ['web-push', 'subscriptions'] as const,
  vapidPublicKey: ['web-push', 'vapid'] as const,
  skills: ['skills'] as const,
  skill: (name: string) => ['skills', name] as const,
  files: ['files'] as const,
};

// ── Projects ────────────────────────────────────────────────────────────────
export function useProjectsQuery() {
  return useQuery({
    queryKey: keys.projects,
    queryFn: () => api<Project[]>(endpoints.projects()),
    refetchInterval: POLL.oneMin,
  });
}

// ── Tickets ─────────────────────────────────────────────────────────────────
export function useTicketsQuery(projectId?: string, interval: number = POLL.verySlow) {
  return useQuery({
    queryKey: keys.tickets(projectId),
    queryFn: () => api<Ticket[]>(endpoints.tickets(projectId)),
    refetchInterval: interval,
  });
}

export function useTicketBoardQuery(projectId?: string) {
  return useQuery({
    queryKey: keys.ticketsBoard(projectId),
    queryFn: () => api<KanbanBoard>(endpoints.ticketsBoard(projectId)),
    refetchInterval: POLL.slow,
  });
}

export interface TicketUpdateResult {
  status: string;
  requeued?: boolean;
}

export function useUpdateTicketStatus() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, status }: { id: string; status: string }) =>
      api<TicketUpdateResult>(endpoints.ticket(id), { method: 'PUT', body: JSON.stringify({ status }) }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['tickets'] });
      qc.invalidateQueries({ queryKey: ['crew-tasks'] });
    },
  });
}

export function useAddTicketComment() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, body }: { id: string; body: string }) =>
      api<void>(endpoints.ticketComments(id), { method: 'POST', body: JSON.stringify({ body }) }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['tickets'] });
    },
  });
}

// ── Budgets ─────────────────────────────────────────────────────────────────
export function useBudgetsQuery(projectId?: string, interval: number = POLL.slow) {
  return useQuery({
    queryKey: keys.budgets(projectId),
    queryFn: () => api<Budget[]>(endpoints.budgets(projectId)),
    refetchInterval: interval,
  });
}

export function useOverrideBudget() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: { budget_id: string; new_limit: number; reason?: string }) =>
      api<void>(endpoints.budgetsOverride(), { method: 'POST', body: JSON.stringify(body) }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['budgets'] });
    },
  });
}

export function useBudgetPauseToggle() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: { project_id: string; agent_role: string; paused: boolean }) =>
      api<{ status: string }>(endpoints.budgetsPause(), {
        method: 'POST',
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['budgets'] });
    },
  });
}

// ── Audit ───────────────────────────────────────────────────────────────────
export function useAuditQuery(limit = 100, projectId?: string, interval: number = POLL.normal) {
  return useQuery({
    queryKey: [...keys.audit(limit), projectId ?? 'all'],
    queryFn: () => api<AuditEntry[]>(endpoints.audit(limit, projectId)),
    refetchInterval: interval,
  });
}

// ── Governance ──────────────────────────────────────────────────────────────
export function useGovernancePendingQuery(projectId?: string, interval: number = POLL.slow) {
  return useQuery({
    queryKey: [...keys.governancePending, projectId ?? 'all'],
    queryFn: () => api<GovernanceRequest[]>(endpoints.governancePending(projectId)),
    refetchInterval: interval,
  });
}

export function useApproveGovernance() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) =>
      api<void>(endpoints.governanceApprove(id), { method: 'POST', body: JSON.stringify({}) }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['governance'] }),
  });
}

export function useRejectGovernance() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, reason }: { id: string; reason?: string }) =>
      api<void>(endpoints.governanceReject(id), {
        method: 'POST',
        body: JSON.stringify(reason ? { reason } : {}),
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['governance'] }),
  });
}

// ── Org chart ───────────────────────────────────────────────────────────────
export function useOrgChartQuery() {
  return useQuery({
    queryKey: keys.orgChart,
    queryFn: () => api<OrgChartAgent[]>(endpoints.orgChart()),
    refetchInterval: POLL.oneMin,
  });
}

// ── Delegation-mode toggles (Org Chart page) ────────────────────────────────
// When ON for a crew, the dispatch uses Coordinator + specialists instead of a
// single monolithic agent.  Preserves full tool palette on providers with
// tight tool limits (Anthropic).
export interface DelegationSettings {
  settings: Record<string, boolean>;
}

export function useDelegationSettingsQuery() {
  return useQuery({
    queryKey: ['delegation-settings'] as const,
    queryFn: () => api<DelegationSettings>(endpoints.delegationSettings()),
    refetchInterval: POLL.oneMin,
  });
}

export function useSetDelegationSetting() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ crew, enabled }: { crew: string; enabled: boolean }) =>
      api<DelegationSettings & { crew: string; enabled: boolean }>(
        endpoints.delegationCrew(crew),
        { method: 'POST', body: JSON.stringify({ enabled }) },
      ),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['delegation-settings'] });
    },
  });
}

// ── Meta-agent toggles (Org Chart page) ─────────────────────────────────────
// When ON for a crew, run_single_agent_crew routes through the meta-agent
// selector — picks a learned recipe from cross-run history, applies it as a
// bounded augmentation. Orthogonal to delegation. See
// app/self_improvement/meta_agent/.
//
// `master_env_on` and `env_overrides` reflect the env-var layer; when set,
// the JSON toggle has no effect for that crew until the env var is unset.
export interface MetaAgentSettings {
  settings: Record<string, boolean>;
  master_env_on: boolean;
  env_overrides: Record<string, string>;
}

export function useMetaAgentSettingsQuery() {
  return useQuery({
    queryKey: ['meta-agent-settings'] as const,
    queryFn: () => api<MetaAgentSettings>(endpoints.metaAgentSettings()),
    refetchInterval: POLL.oneMin,
  });
}

export function useSetMetaAgentSetting() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ crew, enabled }: { crew: string; enabled: boolean }) =>
      api<MetaAgentSettings & { crew: string; enabled: boolean }>(
        endpoints.metaAgentCrew(crew),
        { method: 'POST', body: JSON.stringify({ enabled }) },
      ),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['meta-agent-settings'] });
    },
  });
}

// ── Costs ───────────────────────────────────────────────────────────────────
export function useDailyCostsQuery(days = 30, projectId?: string) {
  return useQuery({
    queryKey: [...keys.costsDaily(days), projectId ?? 'all'],
    queryFn: () => api<CostEntry[]>(endpoints.costsDaily(days, projectId)),
    refetchInterval: POLL.oneMin,
  });
}

export function useAgentCostsQuery(projectId?: string) {
  return useQuery({
    queryKey: [...keys.costsByAgent, projectId ?? 'all'],
    queryFn: () => api<{ by_actor: AgentCost[]; total_cost: number }>(endpoints.costsByAgent(projectId)),
    refetchInterval: POLL.oneMin,
  });
}

export function useCrewCostsQuery(projectId?: string) {
  return useQuery({
    queryKey: [...keys.costsByCrew, projectId ?? 'all'],
    queryFn: () => api<{ by_actor: AgentCost[]; total_cost: number }>(endpoints.costsByCrew(projectId)),
    refetchInterval: POLL.oneMin,
  });
}

export function useInternalAgentCostsQuery(projectId?: string) {
  return useQuery({
    queryKey: [...keys.costsByInternalAgent, projectId ?? 'all'],
    queryFn: () => api<{ by_actor: AgentCost[]; total_cost: number }>(endpoints.costsByInternalAgent(projectId)),
    refetchInterval: POLL.oneMin,
  });
}

// ── Health ──────────────────────────────────────────────────────────────────
export function useHealthQuery(interval: number = POLL.verySlow) {
  return useQuery({
    queryKey: keys.health,
    queryFn: () => api<HealthStatus>(endpoints.health()),
    refetchInterval: interval,
  });
}

// ── Evolution ───────────────────────────────────────────────────────────────
export interface EvolutionResult {
  ts: string;
  experiment_id: string;
  hypothesis: string;
  change_type: string;
  status: string;
  delta: number;
  metric_before: number;
  metric_after: number;
  detail: string;
  engine: string;
  files_changed: string[];
}

export interface EngineStat {
  total: number;
  kept: number;
  kept_ratio: number;
}

export interface EvolutionSummary {
  total_experiments: number;
  kept: number;
  discarded: number;
  crashed: number;
  kept_ratio: number;
  best_score: number;
  current_score: number;
  score_trend: number[];
  current_engine: string;
  subia_safety: number;
  engines: Record<string, EngineStat>;
}

export interface EngineInfo {
  config_mode: string;
  selected_engine: string;
  shinka_available: boolean;
}

export function useEvolutionSummaryQuery() {
  return useQuery({
    queryKey: keys.evolutionSummary,
    queryFn: () => api<EvolutionSummary>(endpoints.evolutionSummary()),
    refetchInterval: POLL.slow,
  });
}

export function useEvolutionResultsQuery(engine: string, status: string) {
  return useQuery({
    queryKey: keys.evolutionResults(engine, status),
    queryFn: () =>
      api<{ results: EvolutionResult[] }>(
        endpoints.evolutionResults({ limit: 100, engine: engine || undefined, status: status || undefined }),
      ),
    refetchInterval: POLL.slow,
  });
}

export function useEvolutionEngineQuery() {
  return useQuery({
    queryKey: keys.evolutionEngine,
    queryFn: () => api<EngineInfo>(endpoints.evolutionEngine()),
    refetchInterval: POLL.verySlow,
  });
}

// ── Workspaces ──────────────────────────────────────────────────────────────
export function useWorkspacesQuery() {
  return useQuery({
    queryKey: keys.workspaces,
    queryFn: () => api<WorkspaceList>(endpoints.workspaces()),
    refetchInterval: POLL.normal,
  });
}

export function useWorkspaceItemsQuery(projectId: string | null) {
  return useQuery({
    queryKey: projectId ? keys.workspaceItems(projectId) : ['workspaces', 'items', 'none'],
    queryFn: () => api<WorkspaceItems>(endpoints.workspaceItems(projectId as string)),
    enabled: !!projectId,
    refetchInterval: POLL.normal,
  });
}

export function useWorkspacesMetaQuery() {
  return useQuery({
    queryKey: keys.workspacesMeta,
    queryFn: () => api<MetaWorkspace>(endpoints.workspacesMeta()),
    refetchInterval: POLL.slow,
  });
}

export function useCreateWorkspace() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: { project_id: string; capacity: number }) =>
      api<void>(endpoints.workspaceCreate(), { method: 'POST', body: JSON.stringify(body) }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['workspaces'] }),
  });
}

// ── Creative mode ───────────────────────────────────────────────────────────
export interface CreativeSettings {
  creative_run_budget_usd: number;
  originality_wiki_weight: number;
  mem0_weight: number;
}

export function useCreativeModeQuery() {
  return useQuery({
    queryKey: keys.creativeMode,
    queryFn: () => api<CreativeSettings>(endpoints.creativeMode()),
  });
}

export function useUpdateCreativeMode() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: { creative_run_budget_usd: number; originality_wiki_weight: number }) =>
      api<CreativeSettings & { status: string }>(endpoints.creativeMode(), {
        method: 'POST',
        body: JSON.stringify(body),
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: keys.creativeMode }),
  });
}

export interface CreativeRunResult {
  final_output: string;
  scores: {
    fluency: number;
    flexibility: number;
    originality: number;
    elaboration: number;
    diagnostics?: Record<string, unknown>;
  } | null;
  cost_usd: number;
  aborted_reason: string | null;
  phases: number;
}

export function useCreativeRun() {
  return useMutation({
    mutationFn: (body: { task: string; creativity?: 'high' | 'medium' }) =>
      api<CreativeRunResult>(endpoints.creativeRun(), {
        method: 'POST',
        body: JSON.stringify({ creativity: 'high', ...body }),
      }),
  });
}

// ── Runtime settings (personal-agent surface) ─────────────────────────────
export type VoiceMode = 'off' | 'local' | 'cloud';

export interface RuntimeSettings {
  voice_mode: VoiceMode;
  vision_cu_enabled: boolean;
  vision_cu_monthly_cap_usd: number;
  concierge_persona_enabled: boolean;
  tier3_amendment_enabled: boolean;
  // Self-heal subsystem master switches (Wave 4 follow-up, 2026-05-09)
  error_runbooks_enabled: boolean;
  tool_supervisor_enabled: boolean;
  recovery_loop_enabled: boolean;
  // Goodhart hard-gate three-way control
  goodhart_hard_gate_disabled: boolean;
  goodhart_hard_gate_enforcing: boolean;
}

export function useRuntimeSettingsQuery() {
  return useQuery({
    queryKey: keys.runtimeSettings,
    queryFn: () => api<RuntimeSettings>(endpoints.runtimeSettings()),
  });
}

export function useUpdateRuntimeSettings() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: Partial<RuntimeSettings>) =>
      api<RuntimeSettings & { status: string }>(endpoints.runtimeSettings(), {
        method: 'POST',
        body: JSON.stringify(body),
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: keys.runtimeSettings }),
  });
}

// ── Background tasks kill switch ──────────────────────────────────────────
export interface BackgroundTasksState {
  enabled: boolean;
}

export function useBackgroundTasksQuery() {
  return useQuery({
    queryKey: keys.backgroundTasks,
    queryFn: () => api<BackgroundTasksState>(endpoints.backgroundTasks()),
    refetchInterval: POLL.normal,
  });
}

export function useSetBackgroundTasks() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (enabled: boolean) =>
      api<{ status: string; enabled: boolean }>(endpoints.backgroundTasks(), {
        method: 'POST',
        body: JSON.stringify({ enabled }),
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: keys.backgroundTasks }),
  });
}

// ── Chat (Signal mirror) ──────────────────────────────────────────────────
export interface ChatMessage {
  role: 'user' | 'assistant' | string;
  content: string;
  ts: number;
}

export interface ChatHistoryResponse {
  sender: string;
  messages: ChatMessage[];
  error?: string;
}

export function useChatMessagesQuery(sender = 'andrus', limit = 50) {
  return useQuery({
    queryKey: keys.chatMessages(sender, limit),
    queryFn: () => api<ChatHistoryResponse>(endpoints.chatMessages(sender, limit)),
    refetchInterval: POLL.fast,
  });
}

export function useChatSend() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: { sender?: string; message: string }) =>
      api<{ sender: string; message: string; reply: string }>(endpoints.chatSend(), {
        method: 'POST',
        body: JSON.stringify(body),
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['chat'] }),
  });
}

// ── Signal commands catalogue ─────────────────────────────────────────────
export interface SignalCommandEntry {
  command: string;
  aliases: string[];
  syntax: string;
  description: string;
  category: string;
}

export interface SignalCommandsResponse {
  categories: string[];
  commands: SignalCommandEntry[];
}

export function useSignalCommandsQuery() {
  return useQuery({
    queryKey: keys.signalCommands,
    queryFn: () => api<SignalCommandsResponse>(endpoints.signalCommands()),
    staleTime: 5 * 60_000,  // catalogue rarely changes
  });
}

// ── System status (monitoring pane) ───────────────────────────────────────
export type StatusLevel = 'ok' | 'warn' | 'error';

export interface SystemCheck {
  name: string;
  category: string;
  status: StatusLevel;
  message: string;
  link?: string | null;
  latency_ms?: number;
  since?: string;
}

export interface SystemStatusResponse {
  checks: SystemCheck[];
  by_category: Record<string, Record<StatusLevel, number>>;
  overall: StatusLevel;
  updated_at: string;
}

export function useSystemStatusQuery() {
  return useQuery({
    queryKey: keys.systemStatus,
    queryFn: () => api<SystemStatusResponse>(endpoints.systemStatus()),
    refetchInterval: POLL.normal,
  });
}

// ── Governance ratchet (Wave 3 #6 — May 2026) ─────────────────────────────

export interface GovernanceRatchetEntry {
  ts: string;
  direction: 'up' | 'down' | 'baseline';
  old_value: number;
  new_value: number;
  source: string;
  reason: string;
  audit_chain: string;
}

export interface GovernanceRatchetThreshold {
  name: 'safety_minimum' | 'quality_minimum';
  floor: number;
  current: number;
  effective: number;
  history: GovernanceRatchetEntry[];
}

export interface GovernanceRatchetState {
  thresholds: GovernanceRatchetThreshold[];
}

export function useGovernanceRatchetQuery() {
  return useQuery({
    queryKey: ['governance_ratchet'],
    queryFn: () =>
      api<GovernanceRatchetState>('/config/governance_ratchet/state'),
  });
}

export function useSetGovernanceRatchet() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: { name: string; new_value: number; reason: string }) =>
      api<{ status: string; state: GovernanceRatchetThreshold }>(
        '/config/governance_ratchet/set',
        {
          method: 'POST',
          body: JSON.stringify(body),
        },
      ),
    onSuccess: () =>
      qc.invalidateQueries({ queryKey: ['governance_ratchet'] }),
  });
}

// ── Per-runbook settings (Wave 4 follow-up, 2026-05-09) ───────────────────

export interface RunbookEntry {
  enabled: boolean;
  min_recurrence: number;
  _comment?: string;
}

export interface RunbookSettings {
  runbooks: Record<string, RunbookEntry>;
}

export function useRunbookSettingsQuery() {
  return useQuery({
    queryKey: ['runbook_settings'],
    queryFn: () => api<RunbookSettings>('/config/runbook_settings'),
  });
}

export function useToggleRunbook() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: { name: string; enabled: boolean; min_recurrence?: number }) =>
      api<{ status: string } & RunbookSettings>('/config/runbook_settings', {
        method: 'POST',
        body: JSON.stringify(body),
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['runbook_settings'] }),
  });
}

export function useRelaxGovernanceRatchet() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: {
      name: string;
      new_value: number;
      reason: string;
      confirmation: string;
    }) =>
      api<{ status: string; state: GovernanceRatchetThreshold }>(
        '/config/governance_ratchet/relax',
        {
          method: 'POST',
          body: JSON.stringify(body),
        },
      ),
    onSuccess: () =>
      qc.invalidateQueries({ queryKey: ['governance_ratchet'] }),
  });
}

// ── Web Push ──────────────────────────────────────────────────────────────

export interface VapidPublicKey { public_key: string; }

export interface WebPushDevice {
  user_agent: string;
  added_at: string;
  endpoint_host: string;
}

export interface WebPushSubscriptionsList {
  configured: boolean;
  count: number;
  devices: WebPushDevice[];
}

export function useVapidPublicKeyQuery() {
  return useQuery({
    queryKey: keys.vapidPublicKey,
    queryFn: () => api<VapidPublicKey>(endpoints.vapidPublicKey()),
    staleTime: 60 * 60 * 1000, // VAPID key effectively never changes per deployment
  });
}

export function useWebPushSubscriptionsQuery() {
  return useQuery({
    queryKey: keys.webPushSubscriptions,
    queryFn: () => api<WebPushSubscriptionsList>(endpoints.webPushSubscriptions()),
  });
}

export function useWebPushSubscribe() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: { endpoint: string; keys: { p256dh: string; auth: string }; userAgent: string }) =>
      api<{ status: string }>(endpoints.webPushSubscribe(), {
        method: 'POST',
        body: JSON.stringify(body),
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: keys.webPushSubscriptions }),
  });
}

export function useWebPushUnsubscribe() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: { endpoint: string }) =>
      api<{ removed: boolean }>(endpoints.webPushUnsubscribe(), {
        method: 'POST',
        body: JSON.stringify(body),
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: keys.webPushSubscriptions }),
  });
}

export function useWebPushTest() {
  return useMutation({
    mutationFn: () =>
      api<{ delivered: number }>(endpoints.webPushTest(), {
        method: 'POST',
        body: JSON.stringify({}),
      }),
  });
}

// ── Skills (Hermes-style saved workflows) ─────────────────────────────────

export interface Skill {
  name: string;
  task_template: string;
  description: string;
  args_schema: string[];
  force_tier: string | null;
  extra_tools: string[];
  task_hint: string;
  created_at: string;
  last_run_at: string | null;
  run_count: number;
  success_count: number;
}

export function useSkillsQuery() {
  return useQuery({
    queryKey: keys.skills,
    queryFn: () => api<{ skills: Skill[] }>(endpoints.skills()),
  });
}

export function useSaveSkill() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: {
      name: string;
      task_template: string;
      description?: string;
      task_hint?: string;
    }) =>
      api<Skill>(endpoints.skills(), {
        method: 'POST',
        body: JSON.stringify(body),
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: keys.skills }),
  });
}

export function useDeleteSkill() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (name: string) =>
      api<{ removed: boolean }>(endpoints.skill(name), { method: 'DELETE' }),
    onSuccess: () => qc.invalidateQueries({ queryKey: keys.skills }),
  });
}

export function useRunSkill() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ name, args }: { name: string; args: Record<string, string> }) =>
      api<{ result: string }>(endpoints.skillRun(name), {
        method: 'POST',
        body: JSON.stringify({ args }),
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: keys.skills }),
  });
}

// ── Files / artifacts ────────────────────────────────────────────────────

export interface FileEntry {
  name: string;
  path: string;
  size: number;
  modified: string;
  extension: string;
  mime: string;
}

export type FilesRoot = 'output' | 'skills' | 'notes';

export interface FilesList {
  roots: Record<FilesRoot, FileEntry[]>;
}

export type SendChannel = 'signal' | 'email' | 'discord';

export interface SendFileBody {
  channel: SendChannel;
  path: string;
  body?: string;
  to?: string;       // email-only
  subject?: string;  // email-only
}

export function useFilesQuery() {
  return useQuery({
    queryKey: keys.files,
    queryFn: () => api<FilesList>(endpoints.files()),
  });
}

export function useSendFile() {
  return useMutation({
    mutationFn: (body: SendFileBody) =>
      api<{ status: string; detail: string }>(endpoints.fileSend(), {
        method: 'POST',
        body: JSON.stringify(body),
      }),
  });
}

// ── KB businesses ───────────────────────────────────────────────────────────
export interface BusinessKB {
  business_name: string;
  collection_name: string;
  total_chunks: number;
  total_documents?: number;
  total_characters?: number;
  categories?: Record<string, number>;
}

// ── Notes viewer ────────────────────────────────────────────────────────────
export interface NoteRoot {
  name: string;
  path: string;
}

export interface NotesRootsReport {
  roots: NoteRoot[];
  default_root?: string | null;
}

export type NoteTreeNodeType = 'dir' | 'note' | 'attachment';

export interface NoteTreeNode {
  name: string;
  path: string;
  type: NoteTreeNodeType;
  size?: number;
  children?: NoteTreeNode[];
}

export interface NotesTreeReport {
  root: string;
  tree: NoteTreeNode;
}

export interface NoteLink {
  path: string;
  title: string;
}

export interface NoteFileReport {
  root: string;
  path: string;
  title: string;
  frontmatter: Record<string, unknown>;
  body: string;
  size: number;
  mtime: number;
  backlinks: NoteLink[];
  forward_links: NoteLink[];
  tags: string[];
  updated_at: string;
}

export interface NoteGraphNode {
  id: string;
  label: string;
  group: string;
  size: number;
  tags: string[];
}

export interface NoteGraphEdge {
  source: string;
  target: string;
}

export interface NotesGraphReport {
  root: string;
  nodes: NoteGraphNode[];
  edges: NoteGraphEdge[];
  tags: string[];
  updated_at: string;
}

export interface NoteSearchHit {
  path: string;
  title: string;
  snippet: string;
  tags: string[];
}

export interface NotesSearchReport {
  query: string;
  hits: NoteSearchHit[];
  total: number;
}

export interface NoteTagEntry {
  tag: string;
  count: number;
  paths: string[];
}

export interface NotesTagsReport {
  root: string;
  tags: NoteTagEntry[];
}

export function useNotesRootsQuery() {
  return useQuery({
    queryKey: keys.notesRoots,
    queryFn: () => api<NotesRootsReport>(endpoints.notesRoots()),
    staleTime: POLL.oneMin,
  });
}

export function useNotesTreeQuery(root: string | null) {
  return useQuery({
    queryKey: keys.notesTree(root ?? ''),
    queryFn: () => api<NotesTreeReport>(endpoints.notesTree(root as string)),
    enabled: !!root,
    staleTime: POLL.verySlow,
  });
}

export function useNoteFileQuery(root: string | null, path: string | null) {
  return useQuery({
    queryKey: keys.notesFile(root ?? '', path ?? ''),
    queryFn: () => api<NoteFileReport>(endpoints.notesFile(root as string, path as string)),
    enabled: !!root && !!path,
    staleTime: POLL.slow,
  });
}

export function useNotesGraphQuery(root: string | null) {
  return useQuery({
    queryKey: keys.notesGraph(root ?? ''),
    queryFn: () => api<NotesGraphReport>(endpoints.notesGraph(root as string)),
    enabled: !!root,
    staleTime: POLL.oneMin,
  });
}

export function useNotesSearchQuery(root: string | null, q: string) {
  return useQuery({
    queryKey: keys.notesSearch(root ?? '', q),
    queryFn: () => api<NotesSearchReport>(endpoints.notesSearch(root as string, q)),
    enabled: !!root && q.trim().length > 0,
    staleTime: POLL.normal,
  });
}

export function useNotesTagsQuery(root: string | null) {
  return useQuery({
    queryKey: keys.notesTags(root ?? ''),
    queryFn: () => api<NotesTagsReport>(endpoints.notesTags(root as string)),
    enabled: !!root,
    staleTime: POLL.oneMin,
  });
}

// ── Ops: errors / anomalies / deploys ───────────────────────────────────────
export interface ErrorEntry {
  ts?: string;
  crew?: string;
  error_type?: string;
  error_msg?: string;
  user_input?: string;
  context?: string;
  diagnosed?: boolean;
  fix_applied?: boolean;
}

export interface ErrorsReport {
  recent: ErrorEntry[];
  patterns: Record<string, number>;
  total_recent: number;
  updated_at: string;
  error?: string | null;
}

export interface AnomalyAlert {
  metric?: string;
  value?: number;
  mean?: number;
  sigma?: number;
  direction?: string;
  type?: string;
  ts?: string;
}

export interface AnomaliesReport {
  recent_alerts: AnomalyAlert[];
  total: number;
  updated_at: string;
  error?: string | null;
}

export interface DeployEntry {
  ts?: string;
  status?: string;               // success | blocked | rollback | auto_rollback
  reason?: string;
  files?: string[];
  error?: string;
  backup_dir?: string;
}

export interface DeploysReport {
  recent: DeployEntry[];
  auto_deploy_enabled?: boolean | null;
  updated_at: string;
  error?: string | null;
}

export function useErrorsQuery() {
  return useQuery({
    queryKey: keys.errors(20),
    queryFn: () => api<ErrorsReport>(endpoints.errors(20)),
    refetchInterval: POLL.slow,
  });
}

export function useAnomaliesQuery() {
  return useQuery({
    queryKey: keys.anomalies(20),
    queryFn: () => api<AnomaliesReport>(endpoints.anomalies(20)),
    refetchInterval: POLL.slow,
  });
}

export function useDeploysQuery() {
  return useQuery({
    queryKey: keys.deploys(20),
    queryFn: () => api<DeploysReport>(endpoints.deploys(20)),
    refetchInterval: POLL.slow,
  });
}

// ── Permanent Error Monitor ─────────────────────────────────────────────────
// Surfaces signature-grouped error patterns + open anomalies from the
// errors.jsonl analyzer (app/observability/error_monitor.py). Polls every
// 30 s; the underlying scan runs every 5 min server-side, so faster polling
// gives no fresher data.

export interface ErrorAuditSummary {
  total_24h: number;
  total_1h: number;
  hourly_avg_24h: number;
  trend: 'rising' | 'falling' | 'stable';
}

export interface ErrorAuditPattern {
  signature: string;
  sample: string;
  count: number;
  share_pct: number;
}

export interface ErrorAuditTrendPoint {
  hour: string;
  count: number;
}

export interface ErrorAuditAnomaly {
  id: string;
  signature: string;
  sample: string;
  type: 'new_pattern' | 'rate_spike' | 'total_rate';
  severity: 'info' | 'warning' | 'critical';
  hourly_rate: number;
  baseline_rate: number;
  detected_at: string | null;
}

export interface ErrorAuditReport {
  summary: ErrorAuditSummary;
  top_patterns_24h: ErrorAuditPattern[];
  trend_hourly: ErrorAuditTrendPoint[];
  active_anomalies: ErrorAuditAnomaly[];
  updated_at: string;
  error?: string | null;
}

export function useErrorAuditQuery() {
  return useQuery({
    queryKey: keys.errorAudit,
    queryFn: () => api<ErrorAuditReport>(endpoints.errorAudit()),
    refetchInterval: POLL.verySlow,
  });
}

export function useAcknowledgeAnomaly() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (anomalyId: string) =>
      api<{ ok: boolean; anomaly_id: string; error?: string }>(
        endpoints.acknowledgeAnomaly(anomalyId),
        { method: 'POST', body: JSON.stringify({}) },
      ),
    onSuccess: () => qc.invalidateQueries({ queryKey: keys.errorAudit }),
  });
}

// ── Tech radar ──────────────────────────────────────────────────────────────
export interface TechDiscovery {
  category: string;       // models | frameworks | research | tools | unknown
  title: string;
  summary?: string;
  action?: string;
}

export interface SearchBackendStatus {
  /** Tier that satisfied the most recent search (null = all tiers failed). */
  last_backend_used: 'brave' | 'searxng' | 'ddg' | null;
  /** Per-tier failure tags accumulated during the cascade. */
  last_failure_chain: string[];
  /** Epoch seconds; non-null = Brave is in 402-quota backoff. */
  brave_quota_blocked_until: number | null;
}

export interface TechRadarReport {
  discoveries: TechDiscovery[];
  updated_at: string;
  error?: string | null;
  search_status?: SearchBackendStatus;
}

export function useTechRadarQuery() {
  return useQuery({
    queryKey: keys.techRadar(20),
    queryFn: () => api<TechRadarReport>(endpoints.techRadar(20)),
    refetchInterval: POLL.oneMin,
  });
}

// ── LLM catalog / roles / discovery ─────────────────────────────────────────
export interface LlmModel {
  name: string;
  tier?: string;
  provider?: string;
  model_id?: string;
  context?: number;
  multimodal?: boolean;
  cost_input_per_m?: number;
  cost_output_per_m?: number;
  tool_use_reliability?: number;
  supports_tools?: boolean;
  description?: string;
  strengths?: Record<string, number>;
  [key: string]: unknown;
}

export interface LlmCatalogReport {
  models: LlmModel[];
  role_assignments: Record<string, string>;
  /** Unified runtime mode (free / budget / balanced / quality / insane / anthropic). */
  mode: string;
  /** Legacy alias for ``mode`` — kept for back-compat. Prefer reading ``mode``. */
  cost_mode?: string;
  /** Authoritative list of pinnable roles from app.llm_catalog.PUBLIC_ROLES. */
  roles?: string[];
  /** Authoritative list of runtime modes from app.llm_catalog.RUNTIME_MODES. */
  modes?: string[];
  /** Legacy alias for ``modes`` — kept for back-compat. */
  cost_modes?: string[];
  updated_at: string;
  error?: string | null;
}

export interface LlmRoleAssignment {
  role: string;
  /** Unified runtime mode (free / budget / balanced / quality / insane / anthropic). */
  mode?: string;
  /** Legacy alias for ``mode``. */
  cost_mode?: string;
  model: string;
  priority?: number;
  source?: string;
  reason?: string;
  assigned_by?: string;
  active?: boolean;
  created_at?: string;
}

export interface LlmRolesReport {
  assignments: LlmRoleAssignment[];
  updated_at: string;
  error?: string | null;
}

export interface DiscoveredModel {
  model_id: string;
  provider?: string;
  display_name?: string;
  context_window?: number;
  cost_input_per_m?: number;
  cost_output_per_m?: number;
  multimodal?: boolean;
  tool_calling?: boolean;
  benchmark_score?: number;
  benchmark_role?: string;
  per_role_scores?: Record<string, number>;
  status?: string;
  promoted_tier?: string;
  promoted_roles?: string[];
  created_at?: string;
  updated_at?: string;
  promoted_at?: string | null;
}

export interface LlmDiscoveryReport {
  discovered: DiscoveredModel[];
  updated_at: string;
  error?: string | null;
}

// ── LLM runtime mode ────────────────────────────────────────────────────────
// The 6-value unified runtime-mode vocabulary. The server also accepts the
// legacy aliases ``hybrid`` / ``local`` / ``cloud`` at ``set_mode`` time, but
// they are normalised away and never returned as ``mode`` in the API.
export type LlmMode =
  | 'free'
  | 'budget'
  | 'balanced'
  | 'quality'
  | 'insane'
  | 'anthropic';

export interface LlmModeReport {
  mode: LlmMode;
  valid_modes: LlmMode[];
}

export function useLlmModeQuery() {
  return useQuery({
    queryKey: keys.llmMode,
    queryFn: () => api<LlmModeReport>(endpoints.llmMode()),
    refetchInterval: POLL.slow,
  });
}

export function useSetLlmMode() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (mode: LlmMode) =>
      api<{ status: string; mode: LlmMode }>(endpoints.llmMode(), {
        method: 'POST',
        body: JSON.stringify({ mode }),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: keys.llmMode });
      qc.invalidateQueries({ queryKey: keys.llmCatalog });
    },
  });
}

export function useLlmCatalogQuery() {
  return useQuery({
    queryKey: keys.llmCatalog,
    queryFn: () => api<LlmCatalogReport>(endpoints.llmCatalog()),
    refetchInterval: POLL.oneMin,
  });
}

export function useLlmRolesQuery() {
  return useQuery({
    queryKey: keys.llmRoles,
    queryFn: () => api<LlmRolesReport>(endpoints.llmRoles()),
    refetchInterval: POLL.oneMin,
  });
}

export function useLlmDiscoveryQuery() {
  return useQuery({
    queryKey: keys.llmDiscovery(50),
    queryFn: () => api<LlmDiscoveryReport>(endpoints.llmDiscovery(50)),
    refetchInterval: POLL.oneMin,
  });
}

export function useRunLlmDiscovery() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (max_benchmarks: number) =>
      api<{ status: string; result: Record<string, unknown> }>(endpoints.llmDiscoveryRun(), {
        method: 'POST',
        body: JSON.stringify({ max_benchmarks }),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['llms'] });
    },
  });
}

// ── Promotions + hand pins ──────────────────────────────────────────────────

export interface PromotionRow {
  model: string;
  promoted_by: string;
  reason?: string | null;
  created_at?: string;
}

export interface PromotionsReport {
  promotions: PromotionRow[];
  updated_at?: string;
  error?: string | null;
}

export interface PinRow {
  role: string;
  /** Unified runtime mode. Canonical field. */
  mode?: string;
  /** Legacy alias for ``mode``; both are present in the payload for back-compat. */
  cost_mode?: string;
  model: string;
  priority?: number;
  source?: string;
  reason?: string | null;
  assigned_by?: string;
  created_at?: string;
}

export interface PinsReport {
  pins: PinRow[];
  updated_at?: string;
  error?: string | null;
}

export function useLlmPromotionsQuery() {
  return useQuery({
    queryKey: keys.llmPromotions,
    queryFn: () => api<PromotionsReport>(endpoints.llmPromotions()),
    refetchInterval: POLL.oneMin,
  });
}

export function useLlmPinsQuery() {
  return useQuery({
    queryKey: keys.llmPins,
    queryFn: () => api<PinsReport>(endpoints.llmPins()),
    refetchInterval: POLL.oneMin,
  });
}

export function usePromoteModel() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ model, reason }: { model: string; reason?: string }) =>
      api<{ status: string; model: string }>(endpoints.llmPromote(), {
        method: 'POST',
        body: JSON.stringify({ model, reason: reason ?? '' }),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['llms'] });
    },
  });
}

export function useDemoteModel() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ model }: { model: string }) =>
      api<{ status: string; model: string }>(endpoints.llmDemote(), {
        method: 'POST',
        body: JSON.stringify({ model }),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['llms'] });
    },
  });
}

export function usePinRole() {
  const qc = useQueryClient();
  return useMutation({
    // ``mode`` is the canonical field. The server still accepts legacy
    // ``cost_mode`` for back-compat if present.
    mutationFn: (
      body: { role: string; mode: string; model: string; reason?: string },
    ) =>
      api<{ status: string }>(endpoints.llmPin(), {
        method: 'POST',
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['llms'] });
    },
  });
}

export function useUnpinRole() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: { role: string; mode: string }) =>
      api<{ status: string; retired: number }>(endpoints.llmUnpin(), {
        method: 'POST',
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['llms'] });
    },
  });
}

// ── Cross-eval judges ───────────────────────────────────────────────────────

export interface JudgeRotationEntry {
  catalog_key: string;
  provider_family: string;
  tier?: string | null;
  provider?: string | null;
  reasoning_score?: number | null;
  pinned: boolean;
}

export interface JudgePin {
  provider_family: string;
  model: string;
  pinned_by: string;
  reason?: string | null;
  pinned_at: string;
}

export interface JudgeAgreement {
  evaluations: number;
  mean_std_dev: number | null;
  high_disagreement: number;
  fallback_fired: number;
  panel_size_avg: number | null;
}

export interface JudgesReport {
  rotation: JudgeRotationEntry[];
  pins: JudgePin[];
  agreement: JudgeAgreement;
  updated_at: string;
  error?: string | null;
}

export interface JudgeEvaluation {
  id: number;
  task_id?: string | null;
  candidate_model: string;
  judges: string[];
  scores: (number | null)[];
  used_fallback: boolean[];
  mean_score: number | null;
  std_dev: number | null;
  rubric?: string | null;
  task_description?: string | null;
  created_at: string;
}

export interface JudgeEvaluationsReport {
  evaluations: JudgeEvaluation[];
  updated_at?: string;
  error?: string | null;
}

export function useLlmJudgesQuery() {
  return useQuery({
    queryKey: ['llms', 'judges'],
    queryFn: () => api<JudgesReport>(endpoints.llmJudges()),
    refetchInterval: POLL.oneMin,
  });
}

export function useLlmJudgeEvaluationsQuery(limit = 50, candidateModel?: string) {
  return useQuery({
    queryKey: ['llms', 'judge-evaluations', limit, candidateModel ?? ''],
    queryFn: () =>
      api<JudgeEvaluationsReport>(endpoints.llmJudgeEvaluations(limit, candidateModel)),
    refetchInterval: POLL.oneMin,
  });
}

export function usePinJudge() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: { provider_family: string; model: string; reason?: string }) =>
      api<{ status: string }>(endpoints.llmJudgePin(), {
        method: 'POST',
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['llms', 'judges'] });
    },
  });
}

export function useUnpinJudge() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: { provider_family: string }) =>
      api<{ status: string; removed: boolean }>(endpoints.llmJudgeUnpin(), {
        method: 'POST',
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['llms', 'judges'] });
    },
  });
}

// ── Evolution variants / genealogy ──────────────────────────────────────────
export interface Variant {
  id: string;
  parent_id?: string;
  generation?: number;
  hypothesis?: string;
  change_type?: string;
  fitness_before?: number;
  fitness_after?: number;
  delta?: number;
  test_pass_rate?: number;
  status?: string;
  files_changed?: string[];
  mutation_summary?: string;
  timestamp?: string;
}

export interface VariantsReport {
  variants: Variant[];
  drift_score: number;
  error?: string | null;
}

export interface VariantLineageReport {
  lineage: Variant[];
  error?: string | null;
}

export function useEvolutionVariantsQuery(n = 30) {
  return useQuery({
    queryKey: keys.evolutionVariants(n),
    queryFn: () => api<VariantsReport>(endpoints.evolutionVariants(n)),
    refetchInterval: POLL.slow,
  });
}

export function useEvolutionVariantLineageQuery(id: string | null) {
  return useQuery({
    queryKey: keys.evolutionVariantLineage(id ?? ''),
    queryFn: () => api<VariantLineageReport>(endpoints.evolutionVariantLineage(id as string)),
    enabled: !!id,
  });
}

// ── Crew tasks (live execution) ─────────────────────────────────────────────
export type CrewTaskState = 'running' | 'completed' | 'failed' | string;

export interface CrewTask {
  id: string;
  crew: string;
  summary?: string;
  state: CrewTaskState;
  started_at?: string;
  completed_at?: string | null;
  eta?: string | null;
  model?: string;
  tokens_used?: number | null;
  cost_usd?: number | null;
  result_preview?: string | null;
  error?: string | null;
  parent_task_id?: string | null;
  is_sub_agent?: boolean;
  delegated_from?: string | null;
  delegated_to?: string | null;
  delegation_reason?: string | null;
  delegation_ts?: string | null;
  sub_agent_progress?: string | null;
  last_updated?: string;
}

export interface CrewStatus {
  name: string;
  state?: string;                // 'idle' | 'running' | 'unknown' | ...
  kind?: 'user' | 'internal';    // backend backfill classifies the roster
  current_task?: string | null;
  started_at?: string | null;
  eta?: string | null;
  [key: string]: unknown;
}

export interface CrewTasksReport {
  tasks: CrewTask[];
  crews: CrewStatus[];
  agents: OrgChartAgent[];
  updated_at: string;
  error?: string | null;
}

export function useCrewTasksQuery(limit = 20, projectId?: string) {
  return useQuery({
    queryKey: [...keys.crewTasks(limit), projectId ?? 'all'],
    queryFn: () => api<CrewTasksReport>(endpoints.tasks(limit, projectId)),
    refetchInterval: POLL.normal,
  });
}

// ── Task-flow drawer (fine-grained execution spans) ─────────────────────────

export type TaskSpanType = 'agent' | 'tool' | 'llm_call';
export type TaskSpanState = 'running' | 'completed' | 'failed';

export interface TaskSpan {
  id: number;
  task_id: string;
  parent_span_id: number | null;
  span_type: TaskSpanType;
  name: string;
  crewai_event_id?: string | null;
  started_at: string;
  completed_at?: string | null;
  state: TaskSpanState;
  detail?: Record<string, unknown>;
  error?: string | null;
  children: TaskSpan[];   // nested by the server-side tree builder
}

export interface TaskTimelineReport {
  task: CrewTask;
  spans: TaskSpan[];      // roots of the tree
  span_count: number;
  updated_at: string;
}

/**
 * Poll the task-flow timeline for a single crew task.
 *
 * Strategy:
 *   - While task.state === 'running', re-fetch every 2 seconds so the
 *     drawer reflects new agent / tool / LLM spans as they appear.
 *   - Once the task hits a terminal state (completed/failed), stop
 *     polling — the tree is frozen.
 *   - ``enabled`` toggles the query off when no task is selected (so
 *     closing the drawer cancels any in-flight poll).
 */
export function useTaskTimelineQuery(taskId: string | null | undefined) {
  return useQuery({
    queryKey: ['taskTimeline', taskId ?? ''],
    queryFn: () => api<TaskTimelineReport>(endpoints.taskTimeline(taskId!)),
    enabled: Boolean(taskId),
    refetchInterval: (query) => {
      const state = query.state.data?.task?.state;
      return state === 'running' ? 2000 : false;
    },
  });
}

// ── Consciousness indicators ────────────────────────────────────────────────
export interface ProbeResult {
  indicator: string;
  theory: string;
  score: number;
  evidence?: string;
  samples?: number;
}

export interface ConsciousnessLatest {
  report_id?: string;
  timestamp?: string;
  probes: ProbeResult[];
  composite_score?: number;
  summary?: string;
}

export interface ConsciousnessHistoryEntry {
  score: number;
  timestamp: string;
  probes: ProbeResult[];
}

export interface HomeostasisState {
  cognitive_energy?: number | null;
  frustration?: number | null;
  confidence?: number | null;
  curiosity?: number | null;
  tasks_since_rest?: number | null;
  consecutive_failures?: number | null;
  last_updated?: string | null;
}

export interface ConsciousnessReport {
  latest: ConsciousnessLatest;
  history: ConsciousnessHistoryEntry[];
  homeostasis?: HomeostasisState;
  updated_at?: string | null;
  error?: string;
}

// ── Token usage & projection ────────────────────────────────────────────────
export type TokenPeriod = 'hour' | 'day' | 'week' | 'month' | 'year';

export interface TokenStat {
  model: string;
  prompt_tokens: number;
  completion_tokens: number;
  total: number;
  cost_usd: number;
  calls: number;
}

export interface RequestCostStat {
  requests: number;
  total_cost_usd: number;
  avg_cost_usd: number;
  avg_calls: number;
  avg_tokens: number;
}

export interface CrewCostStat {
  crew: string;
  requests: number;
  total_cost_usd: number;
  avg_cost_usd: number;
  avg_tokens: number;
}

export interface TokenUsageReport {
  stats: Record<TokenPeriod, TokenStat[]>;
  request_costs: Record<'day' | 'week' | 'month', RequestCostStat>;
  by_crew: { day: CrewCostStat[] };
  projection: {
    day_cost_usd: number;
    mtd_cost_usd: number;
    projected_monthly_usd: number;
  };
  updated_at: string;
  error?: string;
}

export function useTokenUsageQuery(projectId?: string) {
  return useQuery({
    queryKey: [...keys.tokens, projectId ?? 'all'],
    queryFn: () => api<TokenUsageReport>(endpoints.tokens(projectId)),
    refetchInterval: POLL.verySlow,
  });
}

export function useConsciousnessQuery(historyLimit = 30) {
  return useQuery({
    queryKey: keys.consciousness(historyLimit),
    queryFn: () => api<ConsciousnessReport>(endpoints.consciousness(historyLimit)),
    refetchInterval: POLL.verySlow,
  });
}

export function useKbBusinessesQuery() {
  return useQuery({
    queryKey: keys.kbBusinesses,
    queryFn: () => api<{ businesses: BusinessKB[] }>(endpoints.kbBusinesses()),
    refetchInterval: POLL.verySlow,
  });
}

/* ──────────────────────────────────────────────────────────────────
 *  Observability snapshots
 *
 *  A ``snapshot`` is a typed point-in-time observation served by the
 *  gateway from Postgres (no Firebase dependency).  Adding a new
 *  publisher backend-side gives you its data here for free — no
 *  types to add, no collection to subscribe to, no Firebase config.
 *
 *  `useSnapshot("subia_state")`         — one shot of latest payload
 *  `useSnapshotHistory("heartbeat", n)` — most-recent N snapshots
 *  `useSnapshotKinds()`                 — registry of what exists
 *
 *  Each hook returns a TanStack Query result whose shape is
 *  `{ data, isLoading, error, refetch, ... }`.  ``data`` is `null`
 *  (from a 404) when no snapshot has been recorded for that kind yet.
 * ──────────────────────────────────────────────────────────────── */

export interface SnapshotLatest<TPayload = Record<string, unknown>> {
  ts: string;
  kind: string;
  payload: TPayload;
}

export interface SnapshotHistory<TPayload = Record<string, unknown>> {
  kind: string;
  count: number;
  items: Array<{ ts: string; payload: TPayload }>;
}

export interface SnapshotKind {
  kind: string;
  latest_ts: string | null;
  count: number;
}

/** Latest payload for a given snapshot kind.  Returns `null` when
 *  no snapshot has been recorded yet (gateway returns HTTP 404,
 *  which we swallow to avoid noisy error banners for freshly-added
 *  publishers). */
export function useSnapshot<TPayload = Record<string, unknown>>(
  kind: string,
  opts: { refetchMs?: number } = {},
) {
  return useQuery<SnapshotLatest<TPayload> | null>({
    queryKey: keys.snapshotLatest(kind),
    queryFn: async () => {
      try {
        return await api<SnapshotLatest<TPayload>>(
          endpoints.snapshotLatest(kind),
        );
      } catch (err) {
        // 404 = no snapshot yet for this kind; return null rather
        // than propagating into the consumer's isError branch.
        const status = (err as { status?: number } | undefined)?.status;
        if (status === 404) return null;
        throw err;
      }
    },
    refetchInterval: opts.refetchMs ?? POLL.oneMin,
  });
}

/** History of the last N snapshots for a given kind, newest first. */
export function useSnapshotHistory<TPayload = Record<string, unknown>>(
  kind: string,
  limit = 50,
  opts: { refetchMs?: number } = {},
) {
  return useQuery<SnapshotHistory<TPayload>>({
    queryKey: keys.snapshotRecent(kind, limit),
    queryFn: () => api<SnapshotHistory<TPayload>>(
      endpoints.snapshotRecent(kind, limit),
    ),
    refetchInterval: opts.refetchMs ?? POLL.oneMin,
  });
}

/** Catalog of recorded snapshot kinds — useful for debug / admin
 *  pages that want to discover what's being produced. */
export function useSnapshotKinds() {
  return useQuery<{ kinds: SnapshotKind[] }>({
    queryKey: keys.snapshotKinds,
    queryFn: () => api<{ kinds: SnapshotKind[] }>(endpoints.snapshotKinds()),
    refetchInterval: POLL.verySlow,
  });
}


/* ──────────────────────────────────────────────────────────────────
 *  Proposal actions (evolution approve / reject / rollback)
 *
 *  Replaces the legacy Firestore ``proposal_actions`` polling queue.
 *  Direct synchronous HTTP — the backend applies immediately and
 *  returns the result text.  Intended for future React proposal-
 *  review UI; exported now so the API surface is complete.
 * ──────────────────────────────────────────────────────────────── */

export type ProposalActionKind = 'approve' | 'reject' | 'rollback';

export interface ProposalActionResult {
  proposal_id: number;
  action: ProposalActionKind;
  result: string;
}

export function useApplyProposalAction() {
  return useMutation<
    ProposalActionResult,
    Error,
    { proposalId: number | string; action: ProposalActionKind }
  >({
    mutationFn: ({ proposalId, action }) =>
      api<ProposalActionResult>(endpoints.proposalAction(proposalId), {
        method: 'POST',
        body: JSON.stringify({ action }),
      }),
  });
}
