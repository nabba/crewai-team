import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api } from './client';
import { endpoints } from './endpoints';
import type {
  AffectNowReport,
  WelfareAuditReport,
  ReferencePanelReport,
  CalibrationReport,
  CalibrationHistoryReport,
  ReflectionListReport,
  L9SnapshotsReport,
  AttachmentsReport,
  CheckInCandidatesReport,
  CareLedgerReport,
  EcologicalReport,
  ConsciousnessIndicatorsReport,
  Phase5ProposalsReport,
  TraceReport,
  WelfareConfigReport,
  ProposalReviewResult,
  SetpointsApplyResult,
  OverrideResetResult,
} from '../types/affect';

const POLL = {
  fast: 5_000,
  normal: 10_000,
  slow: 15_000,
  verySlow: 30_000,
} as const;

export const affectKeys = {
  now: ['affect', 'now'] as const,
  welfareAudit: (limit: number) => ['affect', 'welfare-audit', limit] as const,
  referencePanel: ['affect', 'reference-panel'] as const,
  calibration: ['affect', 'calibration'] as const,
  calibrationHistory: (limit: number) => ['affect', 'calibration-history', limit] as const,
  reflections: ['affect', 'reflections'] as const,
  reflectionByDate: (date: string) => ['affect', 'reflection', date] as const,
  l9Snapshots: (days: number) => ['affect', 'l9-snapshots', days] as const,
  attachments: ['affect', 'attachments'] as const,
  checkInCandidates: (limit: number) => ['affect', 'check-in-candidates', limit] as const,
  careLedger: (limit: number) => ['affect', 'care-ledger', limit] as const,
  ecological: ['affect', 'ecological'] as const,
  consciousnessIndicators: ['affect', 'consciousness-indicators'] as const,
  phase5Proposals: ['affect', 'phase5-proposals'] as const,
  trace: (hours: number) => ['affect', 'trace', hours] as const,
  welfareConfig: ['affect', 'welfare-config'] as const,
};

export function useAffectNowQuery(intervalMs: number = POLL.normal) {
  return useQuery({
    queryKey: affectKeys.now,
    queryFn: () => api<AffectNowReport>(endpoints.affectNow()),
    refetchInterval: intervalMs,
  });
}

export function useWelfareAuditQuery(limit: number = 100, intervalMs: number = POLL.slow) {
  return useQuery({
    queryKey: affectKeys.welfareAudit(limit),
    queryFn: () => api<WelfareAuditReport>(endpoints.affectWelfareAudit(limit)),
    refetchInterval: intervalMs,
  });
}

export function useReferencePanelQuery() {
  return useQuery({
    queryKey: affectKeys.referencePanel,
    queryFn: () => api<ReferencePanelReport>(endpoints.affectReferencePanel()),
    refetchInterval: POLL.verySlow,
  });
}

export function useCalibrationReportQuery() {
  return useQuery({
    queryKey: affectKeys.calibration,
    queryFn: () => api<CalibrationReport>(endpoints.affectCalibration()),
    refetchInterval: POLL.verySlow,
  });
}

export function useCalibrationHistoryQuery(limit: number = 50) {
  return useQuery({
    queryKey: affectKeys.calibrationHistory(limit),
    queryFn: () => api<CalibrationHistoryReport>(endpoints.affectCalibrationHistory(limit)),
    refetchInterval: POLL.verySlow,
  });
}

export function useReflectionsListQuery() {
  return useQuery({
    queryKey: affectKeys.reflections,
    queryFn: () => api<ReflectionListReport>(endpoints.affectReflections()),
    refetchInterval: POLL.verySlow,
  });
}

export function useReflectionByDateQuery(date: string | null) {
  return useQuery({
    queryKey: affectKeys.reflectionByDate(date ?? ''),
    queryFn: () => api<{ date: string; report: Record<string, unknown> }>(
      endpoints.affectReflectionByDate(date!)
    ),
    enabled: Boolean(date),
  });
}

export function useL9SnapshotsQuery(days: number = 30) {
  return useQuery({
    queryKey: affectKeys.l9Snapshots(days),
    queryFn: () => api<L9SnapshotsReport>(endpoints.affectL9Snapshots(days)),
    refetchInterval: POLL.verySlow,
  });
}

export function useAttachmentsQuery() {
  return useQuery({
    queryKey: affectKeys.attachments,
    queryFn: () => api<AttachmentsReport>(endpoints.affectAttachments()),
    refetchInterval: POLL.slow,
  });
}

export function useCheckInCandidatesQuery(limit: number = 50) {
  return useQuery({
    queryKey: affectKeys.checkInCandidates(limit),
    queryFn: () => api<CheckInCandidatesReport>(endpoints.affectCheckInCandidates(limit)),
    refetchInterval: POLL.slow,
  });
}

export function useCareLedgerQuery(limit: number = 100) {
  return useQuery({
    queryKey: affectKeys.careLedger(limit),
    queryFn: () => api<CareLedgerReport>(endpoints.affectCareLedger(limit)),
    refetchInterval: POLL.verySlow,
  });
}

export function useEcologicalQuery() {
  return useQuery({
    queryKey: affectKeys.ecological,
    queryFn: () => api<EcologicalReport>(endpoints.affectEcological()),
    refetchInterval: POLL.verySlow,
  });
}

export function useConsciousnessIndicatorsQuery() {
  return useQuery({
    queryKey: affectKeys.consciousnessIndicators,
    queryFn: () => api<ConsciousnessIndicatorsReport>(endpoints.affectConsciousnessIndicators()),
    refetchInterval: POLL.verySlow,
  });
}

export function usePhase5ProposalsQuery() {
  return useQuery({
    queryKey: affectKeys.phase5Proposals,
    queryFn: () => api<Phase5ProposalsReport>(endpoints.affectPhase5Proposals()),
    refetchInterval: POLL.verySlow,
  });
}

export function useTraceQuery(hours: number = 24, intervalMs: number = POLL.normal) {
  return useQuery({
    queryKey: affectKeys.trace(hours),
    queryFn: () => api<TraceReport>(endpoints.affectTrace(hours, 200)),
    refetchInterval: intervalMs,
  });
}

export function useWelfareConfigQuery() {
  return useQuery({
    queryKey: affectKeys.welfareConfig,
    queryFn: () => api<WelfareConfigReport>(endpoints.affectWelfareConfig()),
    refetchInterval: POLL.verySlow * 10, // 5 min
  });
}

// ── Mutations ──────────────────────────────────────────────────────────────

export function useReviewProposal() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({
      featureName,
      action,
      note,
    }: {
      featureName: string;
      action: 'approve' | 'defer' | 'reject';
      note?: string;
    }) =>
      api<ProposalReviewResult>(endpoints.affectPhase5ProposalReview(featureName), {
        method: 'POST',
        body: JSON.stringify({ action, note: note ?? '' }),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['affect', 'phase5-proposals'] });
      qc.invalidateQueries({ queryKey: ['affect', 'consciousness-indicators'] });
    },
  });
}

export function useApplySetpoints() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({
      setpoints,
      weights,
      overrideToken,
    }: {
      setpoints?: Record<string, number>;
      weights?: Record<string, number>;
      overrideToken: string;
    }) =>
      api<SetpointsApplyResult>(endpoints.affectSetpoints(), {
        method: 'POST',
        headers: { 'X-Override-Token': overrideToken, 'X-Override-Actor': 'user:dashboard' },
        body: JSON.stringify({ setpoints: setpoints ?? {}, weights: weights ?? {} }),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['affect', 'now'] });
      qc.invalidateQueries({ queryKey: ['affect', 'calibration-history'] });
    },
  });
}

export function useOverrideReset() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ overrideToken }: { overrideToken: string }) =>
      api<OverrideResetResult>(endpoints.affectOverrideReset(), {
        method: 'POST',
        headers: { 'X-Override-Token': overrideToken, 'X-Override-Actor': 'user:dashboard' },
        body: JSON.stringify({}),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['affect'] });
    },
  });
}
