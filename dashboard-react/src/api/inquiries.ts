// Inquiry API hooks — read-only.

import { useQuery } from '@tanstack/react-query';
import { api } from './client';
import type {
  InquiryDetail,
  InquiryListResponse,
  InquiryQuestionsResponse,
} from '../types/inquiries';

const I = '/api/cp/inquiries';

export const inquiriesEndpoints = {
  list: (limit = 100) => `${I}?limit=${limit}`,
  questions: () => `${I}/questions`,
  detail: (filename: string) => `${I}/${encodeURIComponent(filename)}`,
};

export const inquiriesKeys = {
  list: () => ['inquiries', 'list'] as const,
  questions: () => ['inquiries', 'questions'] as const,
  detail: (filename: string) => ['inquiries', 'detail', filename] as const,
};

export function useInquiriesListQuery() {
  return useQuery({
    queryKey: inquiriesKeys.list(),
    queryFn: () => api<InquiryListResponse>(inquiriesEndpoints.list()),
    refetchInterval: 60_000,
  });
}

export function useInquiryQuestionsQuery() {
  return useQuery({
    queryKey: inquiriesKeys.questions(),
    queryFn: () =>
      api<InquiryQuestionsResponse>(inquiriesEndpoints.questions()),
    refetchInterval: 60_000,
  });
}

export function useInquiryDetailQuery(filename: string | undefined) {
  return useQuery({
    queryKey: inquiriesKeys.detail(filename ?? ''),
    queryFn: () =>
      api<InquiryDetail>(inquiriesEndpoints.detail(filename as string)),
    enabled: Boolean(filename),
  });
}
