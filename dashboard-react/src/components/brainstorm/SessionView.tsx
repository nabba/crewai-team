import { useEffect, useMemo, useRef, useState } from 'react';
import {
  useCancelMutation,
  useFinishMutation,
  usePauseMutation,
  useRespondMutation,
  useSkipMutation,
} from '../../api/brainstorm';
import type {
  AgentResponsePayload,
  BrainstormSessionPayload,
  TranscriptTurn,
} from '../../types/brainstorm';
import { AgentRoundBlock } from './AgentRoundBlock';

interface Props {
  session: BrainstormSessionPayload;
  onAfterFinish?: () => void;
}

interface RenderableStep {
  stepIndex: number;
  stepId: string;
  prompt: string;
  userAnswer: string | null;
  seed: AgentResponsePayload[];
  react: AgentResponsePayload[];
}

function _findAssistantPrompts(transcript: TranscriptTurn[]): string[] {
  return transcript
    .filter((t) => t.role === 'assistant')
    .map((t) => t.content);
}

function _findUserAnswers(transcript: TranscriptTurn[]): string[] {
  return transcript
    .filter((t) => t.role === 'user')
    .map((t) => t.content);
}

function _agentRoundsByStep(session: BrainstormSessionPayload) {
  const seedByStep = new Map<string, AgentResponsePayload[]>();
  const reactByStep = new Map<string, AgentResponsePayload[]>();
  for (const r of session.agent_rounds) {
    if (r.phase === 'seed') seedByStep.set(r.step_id, r.responses);
    else if (r.phase === 'react') reactByStep.set(r.step_id, r.responses);
  }
  return { seedByStep, reactByStep };
}

function _buildSteps(session: BrainstormSessionPayload): RenderableStep[] {
  // Each completed step has a recorded response; the current (unanswered)
  // step shows the latest assistant prompt without a user answer.
  const prompts = _findAssistantPrompts(session.transcript);
  const userAnswers = _findUserAnswers(session.transcript);
  const { seedByStep, reactByStep } = _agentRoundsByStep(session);

  const out: RenderableStep[] = [];
  // Past completed steps
  session.responses.forEach((resp, i) => {
    out.push({
      stepIndex: i,
      stepId: resp.step_id,
      prompt: resp.prompt || prompts[i] || '',
      userAnswer: userAnswers[i] ?? resp.response,
      seed: seedByStep.get(resp.step_id) ?? [],
      react: reactByStep.get(resp.step_id) ?? [],
    });
  });
  // Current open step (if any) — last assistant prompt without a user answer
  if (
    !session.is_complete_state_machine &&
    session.status === 'active' &&
    prompts.length > session.responses.length
  ) {
    const i = session.responses.length;
    const currentPrompt = prompts[i] ?? '';
    // Match the current step's seed by step_id: the only seed round whose
    // step_id has not yet been recorded in responses.
    const answeredStepIds = new Set(session.responses.map((r) => r.step_id));
    const pendingSeedEntry = session.agent_rounds.find(
      (r) => r.phase === 'seed' && !answeredStepIds.has(r.step_id),
    );
    out.push({
      stepIndex: i,
      stepId: pendingSeedEntry?.step_id ?? `step_${i}`,
      prompt: currentPrompt,
      userAnswer: null,
      seed: pendingSeedEntry?.responses ?? [],
      react: [],
    });
  }
  return out;
}

export function SessionView({ session, onAfterFinish }: Props) {
  const respondMut = useRespondMutation();
  const skipMut = useSkipMutation();
  const pauseMut = usePauseMutation();
  const cancelMut = useCancelMutation();
  const finishMut = useFinishMutation();

  const [draft, setDraft] = useState('');
  const transcriptEndRef = useRef<HTMLDivElement | null>(null);

  const steps = useMemo(() => _buildSteps(session), [session]);
  const isLastStep =
    session.total_steps !== null &&
    session.responses.length === session.total_steps;
  const isComplete = session.is_complete_state_machine;
  const busy =
    respondMut.isPending ||
    skipMut.isPending ||
    pauseMut.isPending ||
    cancelMut.isPending ||
    finishMut.isPending;

  // Auto-scroll on transcript growth.
  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [session.transcript.length, session.agent_rounds.length]);

  const handleSend = () => {
    const text = draft.trim();
    if (!text || busy) return;
    respondMut.mutate(
      { sessionId: session.session_id, message: text },
      {
        onSuccess: () => setDraft(''),
      },
    );
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex flex-col h-full max-h-[calc(100vh-12rem)]">
      {/* Header */}
      <div className="flex items-start justify-between gap-4 pb-3 border-b border-[#1e2738]">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <h2 className="text-lg font-semibold text-[#e2e8f0] truncate">
              {session.technique_title}
            </h2>
            <span className="text-xs px-2 py-0.5 rounded-full bg-[#1e2738] text-[#7a8599]">
              {session.responses.length}/{session.total_steps ?? '?'}
            </span>
            {session.mode === 'team' && (
              <span className="text-xs px-2 py-0.5 rounded-full bg-[#60a5fa]/10 text-[#60a5fa] border border-[#60a5fa]/20">
                team · {session.participants.join(', ')}
              </span>
            )}
          </div>
          <div className="text-sm text-[#7a8599] mt-1 truncate">
            {session.topic}
          </div>
        </div>
        <div className="flex items-center gap-1 flex-shrink-0">
          <button
            type="button"
            disabled={busy || isComplete}
            onClick={() => skipMut.mutate({ sessionId: session.session_id })}
            className="px-2 py-1 rounded-md text-xs bg-[#0c1118] border border-[#1e2738] text-[#7a8599] hover:text-[#e2e8f0] hover:border-[#3a4a5f] disabled:opacity-40 disabled:cursor-not-allowed"
          >
            Skip
          </button>
          <button
            type="button"
            disabled={busy}
            onClick={() => pauseMut.mutate({ sessionId: session.session_id })}
            className="px-2 py-1 rounded-md text-xs bg-[#0c1118] border border-[#1e2738] text-[#7a8599] hover:text-[#e2e8f0] hover:border-[#3a4a5f] disabled:opacity-40 disabled:cursor-not-allowed"
          >
            Pause
          </button>
          <button
            type="button"
            disabled={busy}
            onClick={() => {
              if (window.confirm('Cancel this brainstorm session?')) {
                cancelMut.mutate({ sessionId: session.session_id });
              }
            }}
            className="px-2 py-1 rounded-md text-xs bg-[#0c1118] border border-[#1e2738] text-[#7a8599] hover:text-[#f87171] hover:border-[#f87171]/40 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            Cancel
          </button>
          <button
            type="button"
            disabled={busy}
            onClick={() => {
              finishMut.mutate(
                { sessionId: session.session_id },
                {
                  onSuccess: () => onAfterFinish?.(),
                },
              );
            }}
            className="px-2 py-1 rounded-md text-xs bg-[#60a5fa] text-[#0a0e14] hover:bg-[#60a5fa]/80 disabled:opacity-40 disabled:cursor-not-allowed font-semibold"
          >
            {finishMut.isPending ? 'Finishing…' : 'Finish'}
          </button>
        </div>
      </div>

      {/* Transcript */}
      <div className="flex-1 overflow-y-auto py-4 space-y-6">
        {steps.map((step) => (
          <div key={`${step.stepIndex}-${step.stepId}`} className="space-y-3">
            <div className="text-[11px] uppercase tracking-wider text-[#7a8599] font-semibold">
              Step {step.stepIndex + 1}
              {session.total_steps ? ` / ${session.total_steps}` : ''} —{' '}
              {step.stepId}
            </div>
            <div className="rounded-md bg-[#0c1118] border border-[#1e2738] p-3 text-sm text-[#e2e8f0] whitespace-pre-wrap">
              {step.prompt}
            </div>
            {step.seed.length > 0 && (
              <AgentRoundBlock label="Agents seed" responses={step.seed} />
            )}
            {step.userAnswer !== null && (
              <div className="rounded-md bg-[#1e3a5f]/30 border border-[#60a5fa]/30 p-3 text-sm">
                <div className="font-semibold text-[#60a5fa] mb-1">You</div>
                <div className="text-[#e2e8f0] whitespace-pre-wrap leading-snug">
                  {step.userAnswer}
                </div>
              </div>
            )}
            {step.react.length > 0 && (
              <AgentRoundBlock label="Agents react" responses={step.react} />
            )}
          </div>
        ))}
        {isComplete && (
          <div className="rounded-md bg-[#2f4a3a]/30 border border-[#4ade80]/30 p-3 text-sm text-[#4ade80]">
            All steps complete. Click <strong>Finish</strong> to generate the
            final report.
          </div>
        )}
        <div ref={transcriptEndRef} />
      </div>

      {/* Input */}
      {!isComplete && (
        <div className="pt-3 border-t border-[#1e2738] space-y-2">
          <textarea
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              busy
                ? session.mode === 'team'
                  ? 'Agents are thinking…'
                  : 'Saving…'
                : 'Your answer…  (⌘/Ctrl+Enter to send)'
            }
            rows={3}
            disabled={busy}
            className="w-full rounded-md bg-[#0c1118] border border-[#1e2738] px-3 py-2 text-sm text-[#e2e8f0] placeholder:text-[#7a8599] focus:outline-none focus:border-[#60a5fa]/40 resize-y disabled:opacity-50"
          />
          <div className="flex items-center justify-between gap-2">
            <div className="text-[11px] text-[#7a8599]">
              {session.mode === 'team' &&
                'Each round dispatches all agents in parallel; replies may take 10–60s.'}
              {isLastStep && session.mode === 'solo' && 'Last step.'}
            </div>
            <button
              type="button"
              disabled={!draft.trim() || busy}
              onClick={handleSend}
              className={`px-4 py-1.5 rounded-md text-sm font-semibold transition-colors ${
                !draft.trim() || busy
                  ? 'bg-[#1e2738] text-[#7a8599] cursor-not-allowed'
                  : 'bg-[#60a5fa] text-[#0a0e14] hover:bg-[#60a5fa]/80'
              }`}
            >
              {respondMut.isPending ? 'Sending…' : 'Send'}
            </button>
          </div>
          {respondMut.isError && (
            <div className="text-xs text-[#f87171]">
              {(respondMut.error as Error).message}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
