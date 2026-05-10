// Unified proposals listing — capability gaps + library adoption +
// recipe retirement.

export type ProposalKind = 'capability' | 'library' | 'recipe';

export interface ProposalSummary {
  kind: ProposalKind;
  name: string;
  title: string;
  size_bytes: number;
  modified_at: string;
  row?: Record<string, unknown>;  // present for recipe kind
}

export interface ProposalDetail extends ProposalSummary {
  body?: string;  // markdown body for capability/library
}

export interface ProposalsListResponse {
  count: number;
  proposals: ProposalSummary[];
  kinds: ProposalKind[];
}
