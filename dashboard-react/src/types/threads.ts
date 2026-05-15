// Thread types — mirror app/threads/models.py

export type ThreadStatus =
  | 'open'
  | 'in_progress'
  | 'blocked'
  | 'resolved'
  | 'abandoned';

export interface SubQuestion {
  id: string;
  text: string;
  resolved: boolean;
  resolution: string;
  resolved_at?: string | null;
}

export interface Thread {
  id: string;
  created_at: string;
  title: string;
  description: string;
  status: ThreadStatus;
  sub_questions: SubQuestion[];
  blockers: string[];
  // Q8.1 (PROGRAM §46.1) — symmetric "what would unblock this" list.
  unblock_hints: string[];
  notes: string[];
  related_crew_task_ids: string[];
  related_inquiry_slugs: string[];
  last_touched_at: string;
  resolved_at?: string | null;
  abandoned_at?: string | null;
  abandon_reason?: string | null;
  // Q8.2 (PROGRAM §46.2) — populated on closure by distill_on_closure.
  approaches_summary?: string;
  is_terminal: boolean;
  open_subquestion_count: number;
  resolved_subquestion_count: number;
}

export interface ThreadListResponse {
  count: number;
  threads: Thread[];
}

export interface ThreadResponse {
  ok: boolean;
  thread: Thread;
}
