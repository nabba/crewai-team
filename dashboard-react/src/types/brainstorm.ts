// Types mirroring app/brainstorm/api.py response shapes.

export type SessionStatus = 'active' | 'paused' | 'complete' | 'cancelled';
export type SessionMode = 'solo' | 'team';

export interface TechniqueInfo {
  name: string;
  title: string;
  description: string;
  total_steps: number;
}

export interface AgentResponsePayload {
  role: string;
  text: string;
  duration_s: number;
  error: string | null;
}

export interface StepDeliveryPayload {
  prompt: string | null;
  seed: AgentResponsePayload[];
  react: AgentResponsePayload[];
}

export interface TranscriptTurn {
  role: 'user' | 'assistant' | 'system' | 'agent';
  content: string;
  ts: number;
  participant?: string;
  phase?: 'seed' | 'react';
}

export interface TechniqueResponse {
  step_id: string;
  prompt: string;
  response: string;
  ts: number;
}

export interface AgentRoundEntry {
  step_id: string;
  phase: 'seed' | 'react';
  ts: number;
  responses: AgentResponsePayload[];
}

export interface BrainstormSessionPayload {
  session_id: string;
  sender: string;
  topic: string;
  technique: string;
  technique_title: string;
  technique_description: string;
  step_index: number;
  total_steps: number | null;
  is_complete_state_machine: boolean;
  status: SessionStatus;
  mode: SessionMode;
  participants: string[];
  transcript: TranscriptTurn[];
  agent_rounds: AgentRoundEntry[];
  responses: TechniqueResponse[];
  created_at: number;
  updated_at: number;
  final_report_path: string | null;
  final_report: string | null;
}

export interface SessionResponse {
  session: BrainstormSessionPayload;
  delivery: StepDeliveryPayload | null;
  advanced: boolean | null;
}
