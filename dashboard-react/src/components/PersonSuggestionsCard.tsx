// L3 + L4.4 suggestions review surface — Q4.2.
//
// Shows recent emitted suggestions so operator can see what the
// system has been nudging about + which categories are firing most.

import { useCompanionPersonSuggestions } from '../api/queries';

const TEXT_DIM = '#7a8599';
const TEXT_BRIGHT = '#e2e8f0';
const PANEL = '#111820';
const BORDER = '#1e2738';

const CATEGORY_COLORS: Record<string, string> = {
  dormancy_nudge: '#fbbf24',
  responsiveness_nudge: '#60a5fa',
  cluster_dormancy: '#a78bfa',
  bridge_maintenance: '#f472b6',
  weak_tie_dormant: '#fb923c',
};

export function PersonSuggestionsCard() {
  const q = useCompanionPersonSuggestions(100);
  const suggestions = q.data?.suggestions || [];

  return (
    <div className="space-y-2">
      <div className="text-sm font-medium" style={{ color: TEXT_BRIGHT }}>
        Recent person-suggestions ({suggestions.length})
      </div>
      <p className="text-[10px]" style={{ color: TEXT_DIM }}>
        Operator-facing nudges from L3 (dormancy / responsiveness) and L4.4
        (graph-driven). Capped at 3 per briefing total. Per-category opt-in.
      </p>

      {suggestions.length === 0 && (
        <div
          className="rounded p-3 border text-xs italic"
          style={{ background: PANEL, borderColor: BORDER, color: TEXT_DIM }}
        >
          No suggestions emitted yet. They appear when one of the enabled
          categories has fired.
        </div>
      )}

      <div className="space-y-1">
        {suggestions.map((s, i) => (
          <div
            key={`${s.person_id}-${s.detected_at}-${i}`}
            className="rounded p-2 border"
            style={{ background: PANEL, borderColor: BORDER }}
          >
            <div className="flex items-center justify-between gap-2 mb-1">
              <span
                className="text-[10px] px-1.5 py-0.5 rounded font-mono"
                style={{
                  background: `${CATEGORY_COLORS[s.category] || '#7a8599'}22`,
                  color: CATEGORY_COLORS[s.category] || TEXT_DIM,
                }}
              >
                {s.category}
              </span>
              <span className="text-[10px]" style={{ color: TEXT_DIM }}>
                {s.detected_at.slice(0, 16).replace('T', ' ')}
              </span>
            </div>
            <div className="text-xs" style={{ color: TEXT_BRIGHT }}>
              {s.text}
            </div>
            {s.display_name && s.display_name !== s.person_id && (
              <div className="text-[10px] mt-0.5" style={{ color: TEXT_DIM }}>
                → {s.display_name}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
