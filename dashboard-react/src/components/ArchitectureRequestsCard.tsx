// Architecture-request primitive toggles — PROGRAM §45.1 Q7.1.
//
// Top-level master + adoption-monitor switch. Both default ON. The
// adoption monitor (29th healing monitor) runs daily and proposes
// rollback CRs for subsystems that have been COMPLETED for 30+ days
// but show zero/low adoption signal. The monitor NEVER auto-applies
// rollback — operator gate intact through the normal CR review flow.

import { api } from '../api/client';
import type { RuntimeSettings } from '../api/queries';
import { useState } from 'react';

const TEXT_DIM = '#7a8599';
const TEXT_BRIGHT = '#e2e8f0';
const WARN = '#f87171';
const WARN_BG = '#7f1d1d22';

export function ArchitectureRequestsCard({
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

  const master = !!settings.architecture_requests_enabled;

  return (
    <div
      className="rounded-lg p-4 border space-y-3"
      style={{ background: '#111820', borderColor: '#1e2738' }}
    >
      <div>
        <h2 className="text-sm font-medium" style={{ color: TEXT_BRIGHT }}>
          Architecture-request primitive (PROGRAM §45)
        </h2>
        <p className="text-[10px] mt-1" style={{ color: TEXT_DIM }}>
          Agent-driven proposal of new SUBSYSTEMS (package-level), not just
          file edits. Goes through Signal + REST + React operator gates,
          scaffolds via coding sessions, then the adoption monitor checks
          for actual usage after 30 days. See{' '}
          <code>app/architecture_requests/</code>.
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
        label="Architecture requests (master)"
        checked={master}
        onChange={(v) => update({ architecture_requests_enabled: v })}
        caveat="When OFF, the lifecycle refuses new proposals (ProtocolDisabled). GET endpoints remain accessible for auditing historical requests."
      />

      {master && (
        <div className="pl-4 space-y-3 border-l-2" style={{ borderColor: '#1e2738' }}>
          <Toggle
            label="Auto-rollback adoption monitor"
            checked={!!settings.architecture_adoption_monitor_enabled}
            onChange={(v) =>
              update({ architecture_adoption_monitor_enabled: v })
            }
            caveat="Daily probe of COMPLETED requests aged 30-60 days. When adoption score < 0.2 (imports + idle-runs + outputs + operator interactions), files a rollback CR for operator review. NEVER auto-applies rollback — operator decides via normal CR gate."
          />
        </div>
      )}
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
