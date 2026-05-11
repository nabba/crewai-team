// Cross-modal pattern detector UI — proactive insights from the
// companion noticing convergence across input modalities.
//
// PROGRAM §41 (2026-05-11) — Q4#15. Operator-facing read-only surface
// for "topics crossing modalities at unusual rates."

import {
  useCrossModalPatterns,
  type CrossModalPattern,
} from '../api/queries';

const TEXT_DIM = '#7a8599';
const TEXT_BRIGHT = '#e2e8f0';
const PANEL_BG = '#111820';
const PANEL_BORDER = '#1e2738';

const MODALITY_COLORS: Record<string, string> = {
  convs: '#60a5fa',
  emails: '#34d399',
  events: '#fbbf24',
  feedback: '#a78bfa',
  affect: '#f472b6',
  tickets: '#fb923c',
};

export function CrossModalPatternsCard() {
  const q = useCrossModalPatterns(20, 0.7);
  const patterns = q.data?.patterns ?? [];

  return (
    <div className="space-y-4">
      <div
        className="rounded-lg p-4 border"
        style={{ background: PANEL_BG, borderColor: PANEL_BORDER }}
      >
        <div className="flex items-baseline justify-between mb-1">
          <h2 className="text-sm font-medium" style={{ color: TEXT_BRIGHT }}>
            Proactive insights — cross-modal convergence
          </h2>
          <span className="text-[10px]" style={{ color: TEXT_DIM }}>
            Q4#15 · 6h cadence
          </span>
        </div>
        <p className="text-[10px] mb-3" style={{ color: TEXT_DIM }}>
          Topics that appeared in ≥3 modalities at high strength in the
          last 21 days. The detector reads interest_model topics + ticket
          subjects + the existing per-source counts (conversations,
          emails, calendar, feedback, affect). Strength = modality_factor
          × volume_factor (log-scaled).
        </p>

        {q.isLoading && (
          <div className="text-sm" style={{ color: TEXT_DIM }}>
            Loading…
          </div>
        )}
        {!q.isLoading && patterns.length === 0 && (
          <div
            className="text-sm italic"
            style={{ color: TEXT_DIM }}
          >
            No convergence patterns detected yet. The threshold is
            conservative; nothing has crossed ≥3 modalities at high
            strength in the current window.
          </div>
        )}

        <div className="space-y-2">
          {patterns.map((p, i) => (
            <PatternRow key={`${p.topic}-${p.detected_at}-${i}`} pattern={p} />
          ))}
        </div>
      </div>
    </div>
  );
}

function PatternRow({ pattern }: { pattern: CrossModalPattern }) {
  const strengthColor =
    pattern.strength >= 0.85
      ? '#34d399'
      : pattern.strength >= 0.7
        ? '#fbbf24'
        : '#7a8599';
  return (
    <div
      className="rounded border p-3"
      style={{ background: PANEL_BG, borderColor: PANEL_BORDER }}
    >
      <div className="flex items-start justify-between gap-2 mb-1">
        <code className="text-sm font-mono" style={{ color: TEXT_BRIGHT }}>
          {pattern.topic}
        </code>
        <div className="flex items-center gap-2 text-[10px]" style={{ color: TEXT_DIM }}>
          <span style={{ color: strengthColor }}>
            strength {pattern.strength.toFixed(2)}
          </span>
          <span>•</span>
          <span>{pattern.occurrences_total} hits / {pattern.window_days}d</span>
        </div>
      </div>
      <div className="flex flex-wrap gap-1.5 mb-1">
        {pattern.modalities.map((m) => (
          <span
            key={m}
            className="px-1.5 py-0.5 rounded text-[10px] font-medium"
            style={{
              background: `${MODALITY_COLORS[m] ?? '#7a8599'}22`,
              color: MODALITY_COLORS[m] ?? '#7a8599',
            }}
          >
            {m} × {pattern.occurrences_per_modality[m] ?? 0}
          </span>
        ))}
      </div>
      <div
        className="flex items-center gap-3 text-[10px]"
        style={{ color: TEXT_DIM }}
      >
        <span>detected {pattern.detected_at.slice(0, 16).replace('T', ' ')}</span>
        {pattern.triggered_tension_boost > 0 && (
          <span style={{ color: '#34d399' }}>
            boosted {pattern.triggered_tension_boost} tension
            {pattern.triggered_tension_boost === 1 ? '' : 's'}
          </span>
        )}
      </div>
    </div>
  );
}
