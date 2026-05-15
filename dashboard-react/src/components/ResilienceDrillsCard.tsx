// Resilience drills toggles — PROGRAM §44 Q6.3.
//
// Master + per-drill switches. kill_the_gateway has additional
// warning text because it is the only DISRUPTIVE drill. Toggling
// it ON enables scheduler notifications; actual execution requires
// the operator to run the external script with a typed-phrase
// confirmation.

import { api } from '../api/client';
import type { RuntimeSettings } from '../api/queries';
import { useState } from 'react';

const TEXT_DIM = '#7a8599';
const TEXT_BRIGHT = '#e2e8f0';
const WARN = '#f87171';
const WARN_BG = '#7f1d1d22';

export function ResilienceDrillsCard({
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

  const master = !!settings.resilience_drills_enabled;

  return (
    <div
      className="rounded-lg p-4 border space-y-3"
      style={{ background: '#111820', borderColor: '#1e2738' }}
    >
      <div>
        <h2 className="text-sm font-medium" style={{ color: TEXT_BRIGHT }}>
          Resilience drills (PROGRAM §44)
        </h2>
        <p className="text-[10px] mt-1" style={{ color: TEXT_DIM }}>
          Quarterly exercises that verify recovery procedures work. Posture
          decision: <em>identity is data, not uptime</em> — no HA, only
          verified-fast-recovery. Read{' '}
          <code>docs/RESILIENCE_POSTURE.md</code> +{' '}
          <code>docs/RESILIENCE_DRILLS.md</code> before disabling any drill.
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
        label="Resilience drills (master)"
        checked={master}
        onChange={(v) => update({ resilience_drills_enabled: v })}
        caveat="When OFF, the scheduler stops emitting drill-due notifications and the four per-drill switches become moot."
      />

      {master && (
        <div className="pl-4 space-y-3 border-l-2" style={{ borderColor: '#1e2738' }}>
          <Toggle
            label="backup_restore drill"
            checked={!!settings.drill_backup_restore_enabled}
            onChange={(v) => update({ drill_backup_restore_enabled: v })}
            caveat="Quarterly. LOW risk — operates on a tarball + ephemeral target directory; never touches live workspace."
          />

          <Toggle
            label="embedding_migration drill"
            checked={!!settings.drill_embedding_migration_enabled}
            onChange={(v) => update({ drill_embedding_migration_enabled: v })}
            caveat="Quarterly. LOW risk — exercises the dry-run state machine against an isolated sandbox collection."
          />

          <Toggle
            label="secret_rotation drill"
            checked={!!settings.drill_secret_rotation_enabled}
            onChange={(v) => update({ drill_secret_rotation_enabled: v })}
            caveat="Quarterly. LOW risk — DRY-RUN ONLY. Verifies the rotation procedure works without rotating any production secret."
          />

          {/* kill_the_gateway — the disruptive one */}
          <div>
            <div className="text-xs font-medium" style={{ color: TEXT_BRIGHT }}>
              kill_the_gateway drill
            </div>
            <div
              className="text-[10px] mt-0.5 p-2 rounded"
              style={{ color: WARN, background: WARN_BG }}
            >
              ⚠️ DISRUPTIVE. This is the only drill that actually stops the
              gateway container. Toggling ON enables scheduler "due"
              notifications — execution still requires the operator to run
              <code> scripts/drills/kill_the_gateway.sh "EXECUTE KILL DRILL"</code>{' '}
              externally during a maintenance window. The gateway cannot
              kill itself, by design.
            </div>
            <label className="flex items-center gap-2 text-xs cursor-pointer mt-2">
              <input
                type="checkbox"
                checked={!!settings.drill_kill_the_gateway_enabled}
                onChange={(e) =>
                  update({ drill_kill_the_gateway_enabled: e.target.checked })
                }
              />
              <span style={{ color: TEXT_BRIGHT, fontWeight: 500 }}>
                Enable scheduler notifications for kill_the_gateway
              </span>
            </label>
          </div>

          <Toggle
            label="drill_staleness monitor"
            checked={!!settings.drill_staleness_monitor_enabled}
            onChange={(v) => update({ drill_staleness_monitor_enabled: v })}
            caveat="Daily probe that alerts when any drill is past its cadence + grace window. Catches drift if the operator stops running drills."
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
