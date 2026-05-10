// Proposals aggregator API hooks — read-only.

import { useQuery } from '@tanstack/react-query';
import { api } from './client';
import type {
  ProposalDetail,
  ProposalKind,
  ProposalsListResponse,
} from '../types/proposals';

const P = '/api/cp/proposals';

export const proposalsEndpoints = {
  list: (kind?: ProposalKind) =>
    kind ? `${P}?kind=${kind}` : P,
  detail: (kind: ProposalKind, name: string) =>
    `${P}/${kind}/${encodeURIComponent(name)}`,
};

export const proposalsKeys = {
  list: (kind?: ProposalKind) =>
    ['proposals', 'list', kind ?? 'all'] as const,
  detail: (kind: ProposalKind, name: string) =>
    ['proposals', 'detail', kind, name] as const,
};

export function useProposalsListQuery(kind?: ProposalKind) {
  return useQuery({
    queryKey: proposalsKeys.list(kind),
    queryFn: () =>
      api<ProposalsListResponse>(proposalsEndpoints.list(kind)),
    refetchInterval: 60_000,
  });
}

export function useProposalDetailQuery(
  kind: ProposalKind | undefined,
  name: string | undefined,
) {
  return useQuery({
    queryKey: proposalsKeys.detail(kind ?? 'capability', name ?? ''),
    queryFn: () =>
      api<ProposalDetail>(proposalsEndpoints.detail(
        kind as ProposalKind, name as string,
      )),
    enabled: Boolean(kind) && Boolean(name),
  });
}
