// Central registry of backend paths. All callers import from here so that
// a path change only needs to happen in one place.

const CP = '/api/cp';

export const endpoints = {
  // Control-plane (prefix /api/cp)
  projects: () => `${CP}/projects`,
  tickets: (projectId?: string) =>
    projectId ? `${CP}/tickets?project_id=${encodeURIComponent(projectId)}` : `${CP}/tickets`,
  ticketsBoard: (projectId?: string) =>
    projectId ? `${CP}/tickets/board?project_id=${encodeURIComponent(projectId)}` : `${CP}/tickets/board`,
  ticket: (id: string) => `${CP}/tickets/${id}`,
  ticketComments: (id: string) => `${CP}/tickets/${id}/comments`,
  budgets: (projectId?: string) =>
    projectId ? `${CP}/budgets?project_id=${encodeURIComponent(projectId)}` : `${CP}/budgets`,
  budgetsOverride: () => `${CP}/budgets/override`,
  budgetsPause: () => `${CP}/budgets/pause`,
  creditAlerts: () => `${CP}/credit-alerts`,
  creditAlertDismiss: () => `${CP}/credit-alerts/dismiss`,
  audit: (limit = 100, projectId?: string) =>
    projectId
      ? `${CP}/audit?limit=${limit}&project_id=${encodeURIComponent(projectId)}`
      : `${CP}/audit?limit=${limit}`,
  governancePending: (projectId?: string) =>
    projectId
      ? `${CP}/governance/pending?project_id=${encodeURIComponent(projectId)}`
      : `${CP}/governance/pending`,
  governanceApprove: (id: string) => `${CP}/governance/${id}/approve`,
  governanceReject: (id: string) => `${CP}/governance/${id}/reject`,
  orgChart: () => `${CP}/org-chart`,
  delegationSettings: () => `${CP}/delegation`,
  delegationCrew: (crew: string) => `${CP}/delegation/${encodeURIComponent(crew)}`,
  metaAgentSettings: () => `${CP}/meta-agent`,
  metaAgentCrew: (crew: string) => `${CP}/meta-agent/${encodeURIComponent(crew)}`,
  costsDaily: (days = 30, projectId?: string) =>
    projectId
      ? `${CP}/costs/daily?days=${days}&project_id=${encodeURIComponent(projectId)}`
      : `${CP}/costs/daily?days=${days}`,
  costsByAgent: (projectId?: string) =>
    projectId ? `${CP}/costs/by-agent?project_id=${encodeURIComponent(projectId)}` : `${CP}/costs/by-agent`,
  costsByCrew: (projectId?: string) =>
    projectId ? `${CP}/costs/by-crew?project_id=${encodeURIComponent(projectId)}` : `${CP}/costs/by-crew`,
  costsByInternalAgent: (projectId?: string) =>
    projectId ? `${CP}/costs/by-internal-agent?project_id=${encodeURIComponent(projectId)}` : `${CP}/costs/by-internal-agent`,
  health: () => `${CP}/health`,
  consciousness: (historyLimit = 30) => `${CP}/consciousness?history_limit=${historyLimit}`,
  tokens: (projectId?: string) =>
    projectId ? `${CP}/tokens?project_id=${encodeURIComponent(projectId)}` : `${CP}/tokens`,
  tasks: (limit = 20, projectId?: string) =>
    projectId
      ? `${CP}/tasks?limit=${limit}&project_id=${encodeURIComponent(projectId)}`
      : `${CP}/tasks?limit=${limit}`,
  taskTimeline: (taskId: string) => `${CP}/tasks/${encodeURIComponent(taskId)}/timeline`,

  // Ops (self-heal / anomaly / self-deploy)
  errors: (limit = 20) => `${CP}/errors?limit=${limit}`,
  anomalies: (limit = 20) => `${CP}/anomalies?limit=${limit}`,
  deploys: (limit = 20) => `${CP}/deploys?limit=${limit}`,

  // Permanent error monitor (errors.jsonl signature analyzer)
  errorAudit: () => `${CP}/error_audit`,
  acknowledgeAnomaly: (id: string) =>
    `${CP}/error_audit/anomaly/${encodeURIComponent(id)}/acknowledge`,

  // Tech radar
  techRadar: (limit = 20) => `${CP}/tech-radar?limit=${limit}`,

  // LLM runtime mode (lives on /config, not /api/cp)
  llmMode: () => '/config/llm_mode',

  // LLMs
  llmCatalog: () => `${CP}/llms/catalog`,
  llmRoles: () => `${CP}/llms/roles`,
  llmDiscovery: (limit = 50) => `${CP}/llms/discovery?limit=${limit}`,
  llmDiscoveryRun: () => `${CP}/llms/discovery/run`,
  llmPromotions: () => `${CP}/llms/promotions`,
  llmPromote: () => `${CP}/llms/promote`,
  llmDemote: () => `${CP}/llms/demote`,
  llmPins: () => `${CP}/llms/pins`,
  llmPin: () => `${CP}/llms/pin`,
  llmUnpin: () => `${CP}/llms/unpin`,

  // Cross-eval judges (rotation + pins + agreement telemetry)
  llmJudges: () => `${CP}/llms/judges`,
  llmJudgePin: () => `${CP}/llms/judges/pin`,
  llmJudgeUnpin: () => `${CP}/llms/judges/unpin`,
  llmJudgeEvaluations: (limit = 50, candidateModel?: string) =>
    candidateModel
      ? `${CP}/llms/judge-evaluations?limit=${limit}&candidate_model=${encodeURIComponent(candidateModel)}`
      : `${CP}/llms/judge-evaluations?limit=${limit}`,

  // Evolution genealogy (variants)
  evolutionVariants: (n = 30) => `${CP}/evolution/variants?n=${n}`,
  evolutionVariantLineage: (id: string) => `${CP}/evolution/variants/${encodeURIComponent(id)}/lineage`,

  // Notes viewer (Obsidian-style)
  notesRoots: () => `${CP}/notes/roots`,
  notesTree: (root: string) => `${CP}/notes/tree?root=${encodeURIComponent(root)}`,
  notesFile: (root: string, path: string) =>
    `${CP}/notes/file?root=${encodeURIComponent(root)}&path=${encodeURIComponent(path)}`,
  notesAttachment: (root: string, path: string) =>
    `${CP}/notes/attachment?root=${encodeURIComponent(root)}&path=${encodeURIComponent(path)}`,
  notesGraph: (root: string) => `${CP}/notes/graph?root=${encodeURIComponent(root)}`,
  notesSearch: (root: string, q: string, limit = 50) =>
    `${CP}/notes/search?root=${encodeURIComponent(root)}&q=${encodeURIComponent(q)}&limit=${limit}`,
  notesTags: (root: string) => `${CP}/notes/tags?root=${encodeURIComponent(root)}`,

  // Observability snapshots (Postgres-backed; migration target off Firestore).
  // These three endpoints cover every current + future snapshot kind —
  // adding a new publisher backend-side requires NO new endpoint entry
  // here, just pass the kind string to useSnapshot* below.
  snapshotKinds: () => `${CP}/observability/snapshots`,
  snapshotLatest: (kind: string) =>
    `${CP}/observability/snapshots/${encodeURIComponent(kind)}/latest`,
  snapshotRecent: (kind: string, limit = 50) =>
    `${CP}/observability/snapshots/${encodeURIComponent(kind)}/recent?limit=${limit}`,

  // Evolution-proposal actions (replaces the legacy Firestore
  // proposal_actions queue — now synchronous via HTTP).
  proposalAction: (proposalId: number | string) =>
    `${CP}/proposals/${encodeURIComponent(String(proposalId))}/action`,

  // Evolution (prefix /api/cp/evolution)
  evolutionSummary: () => `${CP}/evolution/summary`,
  evolutionResults: (params: { limit?: number; engine?: string; status?: string } = {}) => {
    const p = new URLSearchParams();
    p.set('limit', String(params.limit ?? 100));
    if (params.engine) p.set('engine', params.engine);
    if (params.status) p.set('status', params.status);
    return `${CP}/evolution/results?${p.toString()}`;
  },
  evolutionEngine: () => `${CP}/evolution/engine`,

  // Workspaces (prefix /api — NOT /api/cp)
  workspaces: () => `/api/workspaces`,
  workspaceItems: (projectId: string) => `/api/workspaces/${encodeURIComponent(projectId)}/items`,
  workspacesMeta: () => `/api/workspaces/meta`,
  workspaceCreate: () => `/api/workspaces`,

  // Creative mode (prefix /config — requires gateway secret on POST)
  creativeMode: () => `/config/creative_mode`,
  creativeRun: () => `/config/creative_run`,

  // Personal-agent runtime settings — voice mode, vision CU, concierge.
  // POST requires gateway secret (same as creative_mode).
  runtimeSettings: () => `/config/runtime_settings`,

  // Web Push (PWA notifications)
  vapidPublicKey: () => `/config/vapid_public_key`,
  webPushSubscribe: () => `/config/web_push/subscribe`,
  webPushUnsubscribe: () => `/config/web_push/unsubscribe`,
  webPushSubscriptions: () => `/config/web_push/subscriptions`,
  webPushTest: () => `/config/web_push/test`,

  // Skills (Hermes-style "save this workflow" registry)
  skills: () => `${CP}/skills`,
  skill: (name: string) => `${CP}/skills/${encodeURIComponent(name)}`,
  skillRun: (name: string) => `${CP}/skills/${encodeURIComponent(name)}/run`,

  // Files / artifacts (Generated docs, skill markdown, notes)
  files: () => `${CP}/files`,
  fileDownload: (path: string) => `${CP}/files/download?path=${encodeURIComponent(path)}`,
  fileSend: () => `${CP}/files/send`,

  // Knowledge bases (root-mounted prefixes)
  kbStatus: () => `/kb/status`,
  kbUpload: () => `/kb/upload`,
  kbDocuments: () => `/kb/documents`,
  kbBusinesses: () => `/kb/businesses`,
  kbBusinessUpload: (name: string) => `/kb/business/${encodeURIComponent(name)}/upload`,
  fictionStatus: () => `/fiction/status`,
  fictionUpload: () => `/fiction/upload`,
  fictionDocuments: () => `/fiction/documents`,
  philosophyStats: () => `/philosophy/stats`,
  philosophyUpload: () => `/philosophy/upload`,
  philosophyDocuments: () => `/philosophy/documents`,
  epistemeStats: () => `/episteme/stats`,
  epistemeUpload: () => `/episteme/upload`,
  epistemeDocuments: () => `/episteme/documents`,
  experientialStats: () => `/experiential/stats`,
  experientialUpload: () => `/experiential/upload`,
  experientialDocuments: () => `/experiential/documents`,
  aestheticsStats: () => `/aesthetics/stats`,
  aestheticsUpload: () => `/aesthetics/upload`,
  tensionsStats: () => `/tensions/stats`,
  tensionsUpload: () => `/tensions/upload`,

  // Affective layer (Phase 1 + Phase 2)
  affectNow: () => `/affect/now`,
  affectWelfareAudit: (limit = 100) => `/affect/welfare-audit?limit=${limit}`,
  affectReferencePanel: () => `/affect/reference-panel`,
  affectCalibration: () => `/affect/calibration`,
  affectCalibrationHistory: (limit = 50) => `/affect/calibration-history?limit=${limit}`,
  affectReflections: () => `/affect/reflections`,
  affectReflectionByDate: (date: string) => `/affect/reflections/${encodeURIComponent(date)}`,
  affectL9Snapshots: (days = 30) => `/affect/l9-snapshots?days=${days}`,
  affectAttachments: () => `/affect/attachments`,
  affectCheckInCandidates: (limit = 50) => `/affect/check-in-candidates?limit=${limit}`,
  affectCareLedger: (limit = 100) => `/affect/care-ledger?limit=${limit}`,
  affectEcological: () => `/affect/ecological`,
  affectConsciousnessIndicators: () => `/affect/consciousness-indicators`,
  affectPhase5Proposals: () => `/affect/phase5-proposals`,
  affectPhase5ProposalReview: (featureName: string) =>
    `/affect/phase5-proposals/${encodeURIComponent(featureName)}/review`,
  affectTrace: (hours = 24, maxPoints = 200) =>
    `/affect/trace?hours=${hours}&max_points=${maxPoints}`,
  affectWelfareConfig: () => `/affect/welfare-config`,
  affectSetpoints: () => `/affect/setpoints`,
  affectOverrideReset: () => `/affect/override-reset`,

  // Epistemic Integrity Layer
  // see crewai-team/docs/EPISTEMIC_INTEGRITY.md
  epistemicNow: (taskId?: string) =>
    taskId
      ? `/epistemic/now?task_id=${encodeURIComponent(taskId)}`
      : `/epistemic/now`,
  epistemicFeed: (windowMin = 60, limit = 200) =>
    `/epistemic/feed?window_min=${windowMin}&limit=${limit}`,
  epistemicClaim: (claimId: string) =>
    `/epistemic/claim/${encodeURIComponent(claimId)}`,
  epistemicBiases: () => `/epistemic/biases`,
  epistemicVerifiers: () => `/epistemic/verifiers`,
  epistemicPushbackStats: (windowMin = 1440) =>
    `/epistemic/pushback/stats?window_min=${windowMin}`,
  epistemicPushbackRecent: (windowMin = 1440, limit = 50) =>
    `/epistemic/pushback/recent?window_min=${windowMin}&limit=${limit}`,
  epistemicIncidents: (limit = 50) => `/epistemic/incidents?limit=${limit}`,
  epistemicIncident: (incidentId: string) =>
    `/epistemic/incidents/${encodeURIComponent(incidentId)}`,
  epistemicPeerReviewStats: (windowMin = 1440) =>
    `/epistemic/peer-reviews/stats?window_min=${windowMin}`,
  epistemicPeerReviewsRecent: (windowMin = 1440, limit = 50) =>
    `/epistemic/peer-reviews/recent?window_min=${windowMin}&limit=${limit}`,
  epistemicOverrideStats: (windowMin = 1440) =>
    `/epistemic/overrides/stats?window_min=${windowMin}`,
  epistemicOverridesRecent: (windowMin = 1440, limit = 50) =>
    `/epistemic/overrides/recent?window_min=${windowMin}&limit=${limit}`,
  epistemicRecordOverride: () => `/epistemic/overrides`,
  epistemicTuningProposals: (status: string | null = 'proposed', limit = 100) =>
    status === null
      ? `/epistemic/tuning/proposals?limit=${limit}`
      : `/epistemic/tuning/proposals?status=${encodeURIComponent(status)}&limit=${limit}`,
  epistemicTuningProposal: (proposalId: string) =>
    `/epistemic/tuning/proposals/${encodeURIComponent(proposalId)}`,
  epistemicTuningAccept: (proposalId: string) =>
    `/epistemic/tuning/proposals/${encodeURIComponent(proposalId)}/accept`,
  epistemicTuningReject: (proposalId: string) =>
    `/epistemic/tuning/proposals/${encodeURIComponent(proposalId)}/reject`,
  epistemicTuningRun: () => `/epistemic/tuning/run`,
} as const;
