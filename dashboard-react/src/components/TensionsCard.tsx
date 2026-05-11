// Companion tensions UI — open questions Andrus left with the system,
// tracked on his behalf.
//
// PROGRAM §41 (2026-05-11) — Q4#16. Operator can:
//   * List OPEN / DORMANT / RESOLVED tensions
//   * File a new tension manually
//   * Resolve an open tension with a short note
//
// Freshness bar visualizes decay (1.0 fresh → 0.0 stale). Status
// transitions are operator-initiated except DORMANT (auto-aged at 90d).

import { useState } from 'react';
import {
  useCompanionTensions,
  useCreateCompanionTension,
  useResolveCompanionTension,
  type Tension,
} from '../api/queries';

const TEXT_DIM = '#7a8599';
const TEXT_BRIGHT = '#e2e8f0';
const PANEL_BG = '#111820';
const PANEL_BORDER = '#1e2738';

type StatusFilter = 'OPEN' | 'DORMANT' | 'RESOLVED';

export function TensionsCard() {
  const [status, setStatus] = useState<StatusFilter>('OPEN');
  const tensionsQ = useCompanionTensions(status, 0.0);
  const tensions = tensionsQ.data?.tensions ?? [];

  return (
    <div className="space-y-4">
      <div
        className="rounded-lg p-4 border"
        style={{ background: PANEL_BG, borderColor: PANEL_BORDER }}
      >
        <div className="flex items-baseline justify-between mb-1">
          <h2 className="text-sm font-medium" style={{ color: TEXT_BRIGHT }}>
            Open questions you left with me
          </h2>
          <span className="text-[10px]" style={{ color: TEXT_DIM }}>
            Q4#16 · companion tensions store
          </span>
        </div>
        <p className="text-[10px] mb-3" style={{ color: TEXT_DIM }}>
          Tracked across time, accumulating material. Freshness decays after
          ~30 days; OPEN auto-transitions to DORMANT at 90 days untouched.
        </p>

        <div className="flex gap-1 mb-3 text-xs">
          {(['OPEN', 'DORMANT', 'RESOLVED'] as StatusFilter[]).map((s) => (
            <button
              key={s}
              onClick={() => setStatus(s)}
              className="px-2 py-0.5 rounded transition-colors"
              style={
                status === s
                  ? { background: '#60a5fa1f', color: '#60a5fa' }
                  : { color: TEXT_DIM }
              }
            >
              {s}
            </button>
          ))}
        </div>

        <NewTensionForm />

        {tensionsQ.isLoading && (
          <div className="text-sm mt-3" style={{ color: TEXT_DIM }}>
            Loading…
          </div>
        )}
        {!tensionsQ.isLoading && tensions.length === 0 && (
          <div
            className="text-sm italic mt-3"
            style={{ color: TEXT_DIM }}
          >
            {status === 'OPEN'
              ? 'No open tensions. The system surfaces them when you express open questions in conversation.'
              : `No ${status.toLowerCase()} tensions.`}
          </div>
        )}

        <div className="space-y-2 mt-3">
          {tensions.map((t) => (
            <TensionRow key={t.id} tension={t} />
          ))}
        </div>
      </div>
    </div>
  );
}

function NewTensionForm() {
  const [question, setQuestion] = useState('');
  const create = useCreateCompanionTension();

  return (
    <div className="flex gap-2 mb-2">
      <input
        type="text"
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        placeholder="What's an open question you want me to track?"
        className="flex-1 bg-[#0c1118] border border-[#1e2738] rounded px-2 py-1 text-sm"
        style={{ color: TEXT_BRIGHT }}
      />
      <button
        onClick={() => {
          const q = question.trim();
          if (q.length < 8) return;
          create.mutate(
            { question: q },
            { onSuccess: () => setQuestion('') },
          );
        }}
        disabled={create.isPending || question.trim().length < 8}
        className="text-sm px-3 py-1 rounded border disabled:opacity-50"
        style={{
          background: '#60a5fa1f',
          color: '#60a5fa',
          borderColor: '#60a5fa66',
        }}
      >
        {create.isPending ? 'Filing…' : 'File tension'}
      </button>
    </div>
  );
}

function TensionRow({ tension }: { tension: Tension }) {
  const [resolveText, setResolveText] = useState('');
  const [expanded, setExpanded] = useState(false);
  const resolve = useResolveCompanionTension();

  const freshnessColor =
    tension.freshness >= 0.7
      ? '#34d399'
      : tension.freshness >= 0.3
        ? '#fbbf24'
        : '#7a8599';

  return (
    <div
      className="rounded border p-3"
      style={{ background: PANEL_BG, borderColor: PANEL_BORDER }}
    >
      <div className="flex items-center justify-between gap-2 mb-1">
        <code className="text-[10px]" style={{ color: TEXT_DIM }}>
          {tension.id}
        </code>
        <div className="flex items-center gap-2 text-[10px]">
          <span style={{ color: TEXT_DIM }}>
            freshness {tension.freshness.toFixed(2)}
          </span>
          <span
            className="inline-block w-12 h-1 rounded-full bg-[#1e2738] overflow-hidden"
            aria-label={`freshness ${tension.freshness.toFixed(2)}`}
          >
            <span
              className="inline-block h-full transition-all"
              style={{
                width: `${Math.max(0, Math.min(100, tension.freshness * 100))}%`,
                background: freshnessColor,
              }}
            />
          </span>
        </div>
      </div>
      <div className="text-sm whitespace-pre-wrap" style={{ color: TEXT_BRIGHT }}>
        {tension.question}
      </div>
      <div
        className="flex flex-wrap items-center gap-3 text-[10px] mt-1"
        style={{ color: TEXT_DIM }}
      >
        <span>created {tension.created_at.slice(0, 10)}</span>
        <span>touched {tension.last_touched_at.slice(0, 10)}</span>
        <span>via {tension.detection_source}</span>
        {tension.sources.length > 0 && (
          <button
            className="underline"
            onClick={() => setExpanded((v) => !v)}
            style={{ color: '#60a5fa' }}
          >
            {tension.sources.length} note
            {tension.sources.length === 1 ? '' : 's'}{' '}
            {expanded ? '▾' : '▸'}
          </button>
        )}
      </div>
      {expanded && tension.sources.length > 0 && (
        <div className="mt-2 space-y-1">
          {tension.sources.slice(-5).map((s, i) => (
            <div
              key={i}
              className="text-[11px] pl-2 border-l-2"
              style={{
                color: TEXT_DIM,
                borderColor: '#1e2738',
              }}
            >
              <span style={{ color: TEXT_BRIGHT }}>[{s.kind}]</span>{' '}
              <span>{s.ts.slice(0, 16)}</span>
              {s.snippet && (
                <div className="italic ml-2">{s.snippet}</div>
              )}
            </div>
          ))}
        </div>
      )}

      {tension.status === 'OPEN' && (
        <div className="flex gap-2 mt-2">
          <input
            type="text"
            value={resolveText}
            onChange={(e) => setResolveText(e.target.value)}
            placeholder="How did you resolve it?"
            className="flex-1 bg-[#0c1118] border border-[#1e2738] rounded px-2 py-1 text-xs"
            style={{ color: TEXT_BRIGHT }}
          />
          <button
            onClick={() => {
              const r = resolveText.trim();
              if (!r) return;
              resolve.mutate(
                { tid: tension.id, resolution: r },
                { onSuccess: () => setResolveText('') },
              );
            }}
            disabled={resolve.isPending || !resolveText.trim()}
            className="text-xs px-2 py-0.5 rounded border disabled:opacity-50"
            style={{
              background: '#34d3991f',
              color: '#34d399',
              borderColor: '#34d39966',
            }}
          >
            Resolve
          </button>
        </div>
      )}
      {tension.status === 'RESOLVED' && tension.resolution && (
        <div
          className="mt-2 text-xs italic pl-2 border-l-2"
          style={{ color: '#34d399', borderColor: '#34d39966' }}
        >
          Resolved: {tension.resolution}
        </div>
      )}
    </div>
  );
}
