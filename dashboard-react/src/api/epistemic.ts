import { useQuery } from '@tanstack/react-query';
import { api } from './client';
import { endpoints } from './endpoints';
import type {
  BiasFeedReport,
  BiasLibraryReport,
  ClaimDTO,
  EpistemicNowReport,
  IncidentDetailDTO,
  IncidentListReport,
  OverrideStatsReport,
  OverridesRecentReport,
  PeerReviewStatsReport,
  PeerReviewsRecentReport,
  PushbackRecentReport,
  PushbackStatsReport,
  VerifierRegistryReport,
} from '../types/epistemic';

const POLL = {
  fast: 5_000,
  normal: 10_000,
  slow: 30_000,
} as const;

export const epistemicKeys = {
  now: (taskId: string | undefined) => ['epistemic', 'now', taskId ?? null] as const,
  feed: (windowMin: number) => ['epistemic', 'feed', windowMin] as const,
  claim: (claimId: string) => ['epistemic', 'claim', claimId] as const,
  biases: ['epistemic', 'biases'] as const,
  verifiers: ['epistemic', 'verifiers'] as const,
  pushbackStats: (windowMin: number) =>
    ['epistemic', 'pushback', 'stats', windowMin] as const,
  pushbackRecent: (windowMin: number, limit: number) =>
    ['epistemic', 'pushback', 'recent', windowMin, limit] as const,
  incidents: (limit: number) => ['epistemic', 'incidents', limit] as const,
  incident: (id: string) => ['epistemic', 'incident', id] as const,
  peerReviewStats: (windowMin: number) =>
    ['epistemic', 'peer-reviews', 'stats', windowMin] as const,
  peerReviewsRecent: (windowMin: number, limit: number) =>
    ['epistemic', 'peer-reviews', 'recent', windowMin, limit] as const,
  overrideStats: (windowMin: number) =>
    ['epistemic', 'overrides', 'stats', windowMin] as const,
  overridesRecent: (windowMin: number, limit: number) =>
    ['epistemic', 'overrides', 'recent', windowMin, limit] as const,
};

export function useEpistemicNowQuery(
  taskId: string | undefined,
  intervalMs: number = POLL.normal,
) {
  return useQuery({
    queryKey: epistemicKeys.now(taskId),
    queryFn: () => api<EpistemicNowReport>(endpoints.epistemicNow(taskId)),
    refetchInterval: intervalMs,
  });
}

export function useBiasFeedQuery(windowMin: number = 60) {
  return useQuery({
    queryKey: epistemicKeys.feed(windowMin),
    queryFn: () => api<BiasFeedReport>(endpoints.epistemicFeed(windowMin)),
    refetchInterval: POLL.fast,
  });
}

export function useClaimQuery(claimId: string | null) {
  return useQuery({
    queryKey: claimId
      ? epistemicKeys.claim(claimId)
      : (['epistemic', 'claim', '__noop__'] as const),
    queryFn: () => api<ClaimDTO>(endpoints.epistemicClaim(claimId!)),
    enabled: !!claimId,
  });
}

export function useBiasLibraryQuery() {
  return useQuery({
    queryKey: epistemicKeys.biases,
    queryFn: () => api<BiasLibraryReport>(endpoints.epistemicBiases()),
    refetchInterval: POLL.slow,
  });
}

export function useVerifierRegistryQuery() {
  return useQuery({
    queryKey: epistemicKeys.verifiers,
    queryFn: () => api<VerifierRegistryReport>(endpoints.epistemicVerifiers()),
    refetchInterval: POLL.slow,
  });
}

export function usePushbackStatsQuery(windowMin: number = 1440) {
  return useQuery({
    queryKey: epistemicKeys.pushbackStats(windowMin),
    queryFn: () =>
      api<PushbackStatsReport>(endpoints.epistemicPushbackStats(windowMin)),
    refetchInterval: POLL.normal,
  });
}

export function usePushbackRecentQuery(
  windowMin: number = 1440,
  limit: number = 50,
) {
  return useQuery({
    queryKey: epistemicKeys.pushbackRecent(windowMin, limit),
    queryFn: () =>
      api<PushbackRecentReport>(
        endpoints.epistemicPushbackRecent(windowMin, limit),
      ),
    refetchInterval: POLL.normal,
  });
}

export function useIncidentsQuery(limit: number = 50) {
  return useQuery({
    queryKey: epistemicKeys.incidents(limit),
    queryFn: () =>
      api<IncidentListReport>(endpoints.epistemicIncidents(limit)),
    refetchInterval: POLL.slow,
  });
}

export function useIncidentDetailQuery(incidentId: string | null) {
  return useQuery({
    queryKey: incidentId
      ? epistemicKeys.incident(incidentId)
      : (['epistemic', 'incident', '__noop__'] as const),
    queryFn: () =>
      api<IncidentDetailDTO>(endpoints.epistemicIncident(incidentId!)),
    enabled: !!incidentId,
  });
}

export function usePeerReviewStatsQuery(windowMin: number = 1440) {
  return useQuery({
    queryKey: epistemicKeys.peerReviewStats(windowMin),
    queryFn: () =>
      api<PeerReviewStatsReport>(
        endpoints.epistemicPeerReviewStats(windowMin),
      ),
    refetchInterval: POLL.normal,
  });
}

export function usePeerReviewsRecentQuery(
  windowMin: number = 1440,
  limit: number = 50,
) {
  return useQuery({
    queryKey: epistemicKeys.peerReviewsRecent(windowMin, limit),
    queryFn: () =>
      api<PeerReviewsRecentReport>(
        endpoints.epistemicPeerReviewsRecent(windowMin, limit),
      ),
    refetchInterval: POLL.normal,
  });
}

export function useOverrideStatsQuery(windowMin: number = 1440) {
  return useQuery({
    queryKey: epistemicKeys.overrideStats(windowMin),
    queryFn: () =>
      api<OverrideStatsReport>(endpoints.epistemicOverrideStats(windowMin)),
    refetchInterval: POLL.normal,
  });
}

export function useOverridesRecentQuery(
  windowMin: number = 1440,
  limit: number = 50,
) {
  return useQuery({
    queryKey: epistemicKeys.overridesRecent(windowMin, limit),
    queryFn: () =>
      api<OverridesRecentReport>(
        endpoints.epistemicOverridesRecent(windowMin, limit),
      ),
    refetchInterval: POLL.normal,
  });
}
