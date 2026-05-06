import { useState } from 'react';
import { useStartSession, useTechniquesQuery } from '../../api/brainstorm';
import type { TechniqueInfo } from '../../types/brainstorm';

const MAX_AGENTS = 4;
const ROSTER = ['researcher', 'writer', 'coder', 'critic'];

interface Props {
  onStarted?: () => void;
}

export function StartPanel({ onStarted }: Props) {
  const techniquesQuery = useTechniquesQuery();
  const startMut = useStartSession();
  const [technique, setTechnique] = useState<string>('');
  const [topic, setTopic] = useState<string>('');
  const [withAgents, setWithAgents] = useState<number>(0);

  const techniques: TechniqueInfo[] = techniquesQuery.data ?? [];
  const selected = techniques.find((t) => t.name === technique) ?? null;

  const canStart =
    !!technique && topic.trim().length > 0 && !startMut.isPending;

  const handleStart = () => {
    if (!canStart) return;
    startMut.mutate(
      { technique, topic: topic.trim(), withAgents },
      {
        onSuccess: () => {
          setTopic('');
          onStarted?.();
        },
      },
    );
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold text-[#e2e8f0] mb-1">
          Start a brainstorm
        </h2>
        <p className="text-sm text-[#7a8599]">
          Pick a technique, give it a topic, and optionally invite up to{' '}
          {MAX_AGENTS} high-creativity agents to brainstorm with you.
        </p>
      </div>

      {/* Technique grid */}
      <div className="space-y-2">
        <label className="block text-xs uppercase tracking-wider text-[#7a8599] font-semibold">
          Technique
        </label>
        {techniquesQuery.isLoading ? (
          <div className="text-sm text-[#7a8599]">Loading techniques…</div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
            {techniques.map((t) => (
              <button
                key={t.name}
                type="button"
                onClick={() => setTechnique(t.name)}
                className={`text-left rounded-md border p-3 transition-colors ${
                  technique === t.name
                    ? 'bg-[#60a5fa]/10 border-[#60a5fa]/40'
                    : 'bg-[#0c1118] border-[#1e2738] hover:border-[#3a4a5f]'
                }`}
              >
                <div
                  className={`text-sm font-semibold ${
                    technique === t.name ? 'text-[#60a5fa]' : 'text-[#e2e8f0]'
                  }`}
                >
                  {t.title}
                </div>
                <div className="text-xs text-[#7a8599] mt-1 leading-snug">
                  {t.description}
                </div>
                <div className="text-[10px] text-[#7a8599] mt-2">
                  {t.total_steps} steps
                </div>
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Topic */}
      <div className="space-y-2">
        <label className="block text-xs uppercase tracking-wider text-[#7a8599] font-semibold">
          Topic
        </label>
        <textarea
          value={topic}
          onChange={(e) => setTopic(e.target.value)}
          placeholder={
            selected
              ? `What should we brainstorm using ${selected.title}?`
              : 'What should we brainstorm?'
          }
          rows={3}
          className="w-full rounded-md bg-[#0c1118] border border-[#1e2738] px-3 py-2 text-sm text-[#e2e8f0] placeholder:text-[#7a8599] focus:outline-none focus:border-[#60a5fa]/40 resize-y"
        />
      </div>

      {/* With-agents control */}
      <div className="space-y-2">
        <div className="flex items-baseline justify-between">
          <label className="block text-xs uppercase tracking-wider text-[#7a8599] font-semibold">
            Mode
          </label>
          <span className="text-xs text-[#7a8599]">
            {withAgents === 0
              ? 'Solo (just you)'
              : `Team — ${withAgents} agent${withAgents > 1 ? 's' : ''}`}
          </span>
        </div>
        <div className="flex items-center gap-1">
          {[0, 1, 2, 3, 4].map((n) => (
            <button
              key={n}
              type="button"
              onClick={() => setWithAgents(n)}
              className={`flex-1 rounded-md border px-3 py-2 text-sm transition-colors ${
                withAgents === n
                  ? 'bg-[#60a5fa]/10 border-[#60a5fa]/40 text-[#60a5fa]'
                  : 'bg-[#0c1118] border-[#1e2738] text-[#7a8599] hover:border-[#3a4a5f] hover:text-[#e2e8f0]'
              }`}
            >
              {n === 0 ? 'Solo' : `+${n}`}
            </button>
          ))}
        </div>
        {withAgents > 0 && (
          <div className="text-[11px] text-[#7a8599] leading-snug">
            Inviting:{' '}
            <span className="text-[#e2e8f0]">
              {ROSTER.slice(0, withAgents).join(', ')}
            </span>
            . They will seed before each step and react after your answer.
            Multi-agent rounds take 10–60s per step.
          </div>
        )}
      </div>

      {/* Start button */}
      <div>
        <button
          type="button"
          disabled={!canStart}
          onClick={handleStart}
          className={`px-4 py-2 rounded-md text-sm font-semibold transition-colors ${
            canStart
              ? 'bg-[#60a5fa] text-[#0a0e14] hover:bg-[#60a5fa]/80'
              : 'bg-[#1e2738] text-[#7a8599] cursor-not-allowed'
          }`}
        >
          {startMut.isPending ? 'Starting…' : 'Start session'}
        </button>
        {startMut.isError && (
          <div className="mt-2 text-xs text-[#f87171]">
            {(startMut.error as Error).message}
          </div>
        )}
      </div>
    </div>
  );
}
