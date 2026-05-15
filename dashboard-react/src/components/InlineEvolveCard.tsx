// Inline ShinkaEvolve master switch — PROGRAM §45.4 Q7.4.
//
// Gates `app.coding_session.evolution_bridge.evolve_in_session`,
// the per-coding-session ShinkaEvolve wire-in callable via the
// `coding_session_evolve_solution` agent tool. The bulk subsystem
// (`app.shinka_engine`) — which the EvolutionMonitor page surfaces —
// is gated separately and not affected by this switch.
//
// Default ON. When OFF the bridge returns status="disabled" without
// invoking the runner; the agent gets a clear refusal.

import { api } from '../api/client';
import type { RuntimeSettings } from '../api/queries';
import { useState } from 'react';

const TEXT_DIM = '#7a8599';
const TEXT_BRIGHT = '#e2e8f0';
const WARN = '#f87171';
const WARN_BG = '#7f1d1d22';

export function InlineEvolveCard({
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

  const enabled = settings.shinka_inline_evolve_enabled !== false;

  return (
    <div
      className="rounded-lg p-4 border space-y-3"
      style={{ background: '#111820', borderColor: '#1e2738' }}
    >
      <div>
        <h2 className="text-sm font-medium" style={{ color: TEXT_BRIGHT }}>
          Inline ShinkaEvolve per coding session (PROGRAM §45.4)
        </h2>
        <p className="text-[10px] mt-1" style={{ color: TEXT_DIM }}>
          Lets the coder agent run a population-based ShinkaEvolve
          search scoped to ONE file in an active coding session, against
          ONE evaluator script. Hard caps: 20 generations, 3 islands, $5
          per run. Diffs are NOT auto-applied — the agent must write +
          submit through the normal change-request gate. Refused for
          TIER_IMMUTABLE files, anything under <code>app/subia/</code>,
          and <code>app/affect/goal_emitter.py</code>. Audit at{' '}
          <code>workspace/coding_sessions/&lt;id&gt;/evolution_audit.jsonl</code>{' '}
          survives session cleanup; visible in{' '}
          <code>/cp/coding-sessions</code>.
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
        label="Inline evolution (master)"
        checked={enabled}
        onChange={(v) => update({ shinka_inline_evolve_enabled: v })}
        caveat="When OFF, the bridge returns status=disabled instead of invoking the runner. The agent tool still exists but every call returns a refusal until you re-enable. Independent of the bulk ShinkaEvolve subsystem (EvolutionMonitor page)."
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
