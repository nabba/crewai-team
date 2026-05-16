// Analogy-index populator master switch — PROGRAM §46.18 Q11.1.
//
// Controls the HEAVY weekly LLM pass that walks wiki + episteme and
// extracts abstract structural patterns into the analogy index. The
// brainstorm subsystem queries the index on session open and folds
// cross-domain analogues into the agent prompts.
//
// Cost discipline: capped at 5 new entries per pass, ~$0.05/entry =
// ~$0.25 weekly worst case. Operator turns OFF here if they want
// to pause spend without rebuilding the gateway.

import { useState } from 'react';
import { api } from '../api/client';
import type { RuntimeSettings } from '../api/queries';

const TEXT_DIM = '#7a8599';
const TEXT_BRIGHT = '#e2e8f0';
const WARN = '#f87171';
const WARN_BG = '#7f1d1d22';

export function AnalogyIndexCard({
  settings,
  onSettingsChange,
}: {
  settings: RuntimeSettings | Partial<RuntimeSettings>;
  onSettingsChange: () => void;
}) {
  const [error, setError] = useState<string | null>(null);

  const update = async (patch: Record<string, unknown>) => {
    setError(null);
    try {
      await api('/api/cp/settings', {
        method: 'POST',
        body: JSON.stringify(patch),
      });
      onSettingsChange();
    } catch (e) {
      setError(String(e));
    }
  };

  const enabled = settings.analogy_index_populator_enabled !== false;

  return (
    <div
      className="rounded-lg p-4 border space-y-3"
      style={{ background: '#111820', borderColor: '#1e2738' }}
    >
      <div>
        <h2 className="text-sm font-medium" style={{ color: TEXT_BRIGHT }}>
          Analogy-index populator (PROGRAM §46.18)
        </h2>
        <p className="text-[10px] mt-1" style={{ color: TEXT_DIM }}>
          HEAVY weekly LLM pass over wiki + episteme that extracts
          abstract structural patterns into the analogy index. The
          brainstorm subsystem queries the index on session open and
          folds cross-domain analogues into the agent prompts. Cap of
          5 new entries per pass; ~$0.25 weekly worst-case spend.
          Queries to the index keep working even when the populator
          is OFF — turning this off just stops growing the index.
        </p>
      </div>

      {error && (
        <div
          className="text-xs p-2 rounded"
          style={{ color: WARN, background: WARN_BG }}
        >
          {error}
        </div>
      )}

      <Toggle
        label="Analogy populator (HEAVY weekly)"
        checked={enabled}
        onChange={(v) => update({ analogy_index_populator_enabled: v })}
        caveat="Default ON. OFF = no new analogy entries; existing entries still queryable. The brainstorm subsystem degrades gracefully on an empty index — no analogues, just the bare technique prompts."
      />
    </div>
  );
}

function Toggle({
  label,
  checked,
  onChange,
  caveat,
}: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
  caveat?: string;
}) {
  return (
    <div>
      <label className="flex items-center gap-2 text-xs cursor-pointer">
        <input
          type="checkbox"
          checked={checked}
          onChange={(e) => onChange(e.target.checked)}
        />
        <span style={{ color: TEXT_BRIGHT, fontWeight: 500 }}>{label}</span>
      </label>
      {caveat && (
        <p className="text-[10px] ml-5 mt-0.5" style={{ color: TEXT_DIM }}>
          {caveat}
        </p>
      )}
    </div>
  );
}
