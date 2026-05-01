import { useState } from 'react';
import {
  useBiasFeedQuery,
  useBiasLibraryQuery,
  useEpistemicNowQuery,
  useVerifierRegistryQuery,
} from '../api/epistemic';
import { BiasFeed } from './epistemic/BiasFeed';
import { CalibrationSummary } from './epistemic/CalibrationSummary';
import { IncidentsPanel } from './epistemic/IncidentsPanel';
import { NowLedger } from './epistemic/NowLedger';
import { OverridesPanel } from './epistemic/OverridesPanel';
import { PeerReviewsPanel } from './epistemic/PeerReviewsPanel';
import { PushbackPanel } from './epistemic/PushbackPanel';
import { TuningProposalsPanel } from './epistemic/TuningProposalsPanel';

const STORAGE_KEY = 'epistemic.taskId';

export function EpistemicPage() {
  const [taskId, setTaskId] = useState<string>(
    () => localStorage.getItem(STORAGE_KEY) ?? '',
  );
  const trimmed = taskId.trim();
  const effectiveTaskId = trimmed === '' ? undefined : trimmed;

  const nowQuery = useEpistemicNowQuery(effectiveTaskId);
  const feedQuery = useBiasFeedQuery(60);
  const biasesQuery = useBiasLibraryQuery();
  const verifiersQuery = useVerifierRegistryQuery();

  function commitTaskId(next: string) {
    setTaskId(next);
    if (next.trim() === '') localStorage.removeItem(STORAGE_KEY);
    else localStorage.setItem(STORAGE_KEY, next.trim());
  }

  return (
    <div className="space-y-6 max-w-6xl">
      <header>
        <h1 className="text-2xl font-semibold text-[#e2e8f0]">
          Epistemic Integrity
        </h1>
        <p className="text-sm text-[#7a8599]">
          Provenance, calibration, and post-mortem analysis of the agent's
          reasoning. Read the design doc:{' '}
          <code>crewai-team/docs/EPISTEMIC_INTEGRITY.md</code>
        </p>
      </header>

      <TaskPicker
        value={taskId}
        onChange={setTaskId}
        onCommit={commitTaskId}
      />

      {nowQuery.isLoading && <Skeleton lines={4} />}
      {nowQuery.isError && (
        <ErrorBlock>
          Failed to load /epistemic/now: {String(nowQuery.error)}
        </ErrorBlock>
      )}
      {nowQuery.data && <CalibrationSummary report={nowQuery.data} />}

      {nowQuery.data?.ledger && (
        <NowLedger claims={nowQuery.data.ledger} />
      )}
      {nowQuery.data && nowQuery.data.task_id === null && (
        <NoTaskHint />
      )}

      {feedQuery.isLoading && <Skeleton lines={3} />}
      {feedQuery.isError && (
        <ErrorBlock>
          Failed to load /epistemic/feed: {String(feedQuery.error)}
        </ErrorBlock>
      )}
      {feedQuery.data && (
        <BiasFeed
          matches={feedQuery.data.matches}
          windowMinutes={feedQuery.data.window_minutes}
        />
      )}

      <PushbackPanel />

      <PeerReviewsPanel />

      <OverridesPanel />

      <IncidentsPanel />

      <TuningProposalsPanel />

      <BiasLibrarySection
        loading={biasesQuery.isLoading}
        error={biasesQuery.error}
        data={biasesQuery.data}
      />

      <VerifierRegistrySection
        loading={verifiersQuery.isLoading}
        error={verifiersQuery.error}
        data={verifiersQuery.data}
      />

      <footer className="text-xs text-[#7a8599] pt-4 border-t border-[#1e2738]">
        Phase 1 (this commit): provenance + ``inference_as_fact`` realtime
        detector + warn-mode calibration. Phases 2–7 wire the remaining
        biases, the pushback handler, post-mortem, peer review, and the
        Phase-7 blocking-mode toggle.
      </footer>
    </div>
  );
}

function TaskPicker({
  value,
  onChange,
  onCommit,
}: {
  value: string;
  onChange: (v: string) => void;
  onCommit: (v: string) => void;
}) {
  return (
    <section className="rounded-lg bg-[#111820] border border-[#1e2738] p-4">
      <div className="flex items-center gap-3">
        <label
          htmlFor="task-id"
          className="text-xs text-[#7a8599] uppercase tracking-wide"
        >
          Task ID
        </label>
        <input
          id="task-id"
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onBlur={(e) => onCommit(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') onCommit((e.target as HTMLInputElement).value);
          }}
          placeholder="e.g. task_abc — leave empty for global view"
          className="flex-1 bg-[#0a0e14] border border-[#1e2738] text-[#e2e8f0] text-sm rounded px-2 py-1 placeholder-[#7a8599]/60 focus:outline-none focus:border-[#60a5fa]"
        />
        {value && (
          <button
            onClick={() => onCommit('')}
            className="text-xs text-[#7a8599] hover:text-[#e2e8f0]"
          >
            clear
          </button>
        )}
      </div>
    </section>
  );
}

function NoTaskHint() {
  return (
    <section className="rounded-lg bg-[#111820] border border-[#1e2738] p-4">
      <p className="text-sm text-[#7a8599]">
        No task selected. Enter a <code>task_id</code> above to view that
        task's claim ledger and bias matches. The feed below shows recent
        firings across all tasks.
      </p>
    </section>
  );
}

function BiasLibrarySection({
  loading,
  error,
  data,
}: {
  loading: boolean;
  error: unknown;
  data: { biases: Array<{ id: string; name: string; description: string; severity: string; phase: string; corrective_action: string | null; blocking: boolean }> } | undefined;
}) {
  return (
    <section className="rounded-lg bg-[#111820] border border-[#1e2738] p-4">
      <header className="mb-3">
        <h2 className="text-lg font-medium text-[#e2e8f0]">Bias library</h2>
        <p className="text-xs text-[#7a8599]">
          Named cognitive failure modes the system can detect
        </p>
      </header>
      {loading && <Skeleton lines={2} />}
      {error && <ErrorBlock>{String(error)}</ErrorBlock>}
      {data && (
        <ul className="divide-y divide-[#1e2738]">
          {data.biases.map((b) => (
            <li key={b.id} className="py-3">
              <div className="flex items-baseline gap-2">
                <code className="text-[#22d3ee] text-sm">{b.id}</code>
                <span className="text-xs text-[#fb923c]">{b.severity}</span>
                <span className="text-xs text-[#7a8599]">{b.phase}</span>
                {b.blocking && (
                  <span className="text-xs text-[#f87171]">blocking</span>
                )}
              </div>
              <p className="text-sm text-[#e2e8f0] mt-1">{b.name}</p>
              <p className="text-xs text-[#7a8599] mt-1 whitespace-pre-wrap">
                {b.description}
              </p>
              {b.corrective_action && (
                <p className="text-xs text-[#a78bfa] mt-1">
                  → {b.corrective_action}
                </p>
              )}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

function VerifierRegistrySection({
  loading,
  error,
  data,
}: {
  loading: boolean;
  error: unknown;
  data: { verifiers: Array<{ id: string; tool: string; expected_signal: string; estimated_seconds: number; tags_any: string[] }> } | undefined;
}) {
  return (
    <section className="rounded-lg bg-[#111820] border border-[#1e2738] p-4">
      <header className="mb-3">
        <h2 className="text-lg font-medium text-[#e2e8f0]">
          Verifier registry
        </h2>
        <p className="text-xs text-[#7a8599]">
          Cheap exact-answer commands that settle claim shapes
        </p>
      </header>
      {loading && <Skeleton lines={2} />}
      {error && <ErrorBlock>{String(error)}</ErrorBlock>}
      {data && (
        <ul className="divide-y divide-[#1e2738]">
          {data.verifiers.map((v) => (
            <li key={v.id} className="py-2.5 flex items-baseline gap-3">
              <code className="text-[#22d3ee] text-sm w-56 shrink-0 truncate">
                {v.id}
              </code>
              <code className="text-[#e2e8f0] text-xs flex-1 min-w-0 truncate">
                {v.tool}
              </code>
              <span className="text-xs text-[#7a8599] whitespace-nowrap">
                ~{v.estimated_seconds}s
              </span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

function Skeleton({ lines = 3 }: { lines?: number }) {
  return (
    <div className="rounded-lg bg-[#111820] border border-[#1e2738] p-4 space-y-2">
      {Array.from({ length: lines }).map((_, i) => (
        <div key={i} className="h-4 bg-[#1e2738] rounded animate-pulse" />
      ))}
    </div>
  );
}

function ErrorBlock({ children }: { children: React.ReactNode }) {
  return (
    <div className="rounded-lg bg-[#1a0e0e] border border-[#f87171]/40 p-4 text-sm text-[#f87171]">
      {children}
    </div>
  );
}
