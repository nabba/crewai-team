// Resilience drills toggles — PROGRAM §44 Q6.3.
//
// Master + per-drill switches. kill_the_gateway has additional
// warning text because it is the only DISRUPTIVE drill. Toggling
// it ON enables scheduler notifications; actual execution requires
// the operator to run the external script with a typed-phrase
// confirmation.

import { api } from '../api/client';
import { useDrillsRegistryQuery } from '../api/queries';
import type { RuntimeSettings } from '../api/queries';
import { useState } from 'react';

const TEXT_DIM = '#7a8599';
const TEXT_BRIGHT = '#e2e8f0';
const WARN = '#f87171';
const WARN_BG = '#7f1d1d22';
const PASS_GREEN = '#4ade80';
const STALE_AMBER = '#fbbf24';

export function ResilienceDrillsCard({
  settings,
  onSettingsChange,
}: {
  settings: RuntimeSettings | Partial<RuntimeSettings>;
  onSettingsChange: () => void;
}) {
  const [error, setError] = useState<string | null>(null);
  const drillsQ = useDrillsRegistryQuery();

  const update = async (patch: Record<string, unknown>) => {
    setError(null);
    try {
      await api('/api/cp/settings', {
        method: 'POST',
        body: JSON.stringify(patch),
      });
      onSettingsChange();
      drillsQ.refetch();
    } catch (e) {
      setError(String(e));
    }
  };

  const master = !!settings.resilience_drills_enabled;

  // Q6.4 P2#8 — display drill state alongside toggles. Map by name
  // for O(1) lookup; tolerate query failures (empty map → no state
  // info but toggles still work).
  const drillState = new Map<string, ReturnType<typeof drillStateFor>>();
  if (drillsQ.data?.drills) {
    for (const d of drillsQ.data.drills) {
      drillState.set(d.name, drillStateFor(d));
    }
  }

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
            state={drillState.get('backup_restore')}
          />

          <Toggle
            label="embedding_migration drill"
            checked={!!settings.drill_embedding_migration_enabled}
            onChange={(v) => update({ drill_embedding_migration_enabled: v })}
            caveat="Quarterly. LOW risk — exercises the dry-run state machine against an isolated sandbox collection."
            state={drillState.get('embedding_migration')}
          />

          <Toggle
            label="secret_rotation drill"
            checked={!!settings.drill_secret_rotation_enabled}
            onChange={(v) => update({ drill_secret_rotation_enabled: v })}
            caveat="Quarterly. LOW risk — DRY-RUN ONLY. Verifies the rotation procedure works without rotating any production secret."
            state={drillState.get('secret_rotation')}
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
              {drillState.get('kill_the_gateway') && (() => {
                const s = drillState.get('kill_the_gateway')!;
                return (
                  <span
                    className="text-[10px] ml-2 px-1.5 py-0.5 rounded"
                    style={{
                      color: s.color, background: s.bg,
                      border: `1px solid ${s.color}33`,
                    }}
                    title={s.tooltip}
                  >
                    {s.label}
                  </span>
                );
              })()}
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
  state,
}: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
  caveat?: string;
  state?: ReturnType<typeof drillStateFor>;
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
        {state && (
          <span
            className="text-[10px] ml-2 px-1.5 py-0.5 rounded"
            style={{
              color: state.color,
              background: state.bg,
              border: `1px solid ${state.color}33`,
            }}
            title={state.tooltip}
          >
            {state.label}
          </span>
        )}
      </label>
      {caveat && (
        <p className="text-[10px] ml-5 mt-0.5" style={{ color: TEXT_DIM }}>
          {caveat}
        </p>
      )}
    </div>
  );
}


function drillStateFor(d: {
  name: string;
  last_status: string | null;
  last_run_at: string | null;
  days_since_last_success: number | null;
  cadence_days: number;
  grace_days: number;
}) {
  // Never run — neutral.
  if (d.last_run_at === null) {
    return {
      label: 'never run',
      color: TEXT_DIM,
      bg: '#1e273866',
      tooltip: `Drill ${d.name} has not been run yet`,
    };
  }
  const days = d.days_since_last_success;
  const stale = days === null || days > d.cadence_days + d.grace_days;
  const last = d.last_status || 'unknown';
  if (last === 'pass' && !stale) {
    return {
      label: `${days?.toFixed(0) ?? '?'}d ago`,
      color: PASS_GREEN,
      bg: '#16653422',
      tooltip: `Last pass: ${d.last_run_at}`,
    };
  }
  if (stale) {
    return {
      label: 'STALE',
      color: STALE_AMBER,
      bg: '#78350f22',
      tooltip: `Past cadence (${d.cadence_days}d) + grace (${d.grace_days}d)`,
    };
  }
  if (last === 'fail' || last === 'error') {
    return {
      label: last.toUpperCase(),
      color: WARN,
      bg: WARN_BG,
      tooltip: `Last run ${last}: ${d.last_run_at}`,
    };
  }
  if (last === 'skipped') {
    return {
      label: 'skipped',
      color: TEXT_DIM,
      bg: '#1e273866',
      tooltip: 'Last run was skipped (master switch off or in-flight)',
    };
  }
  return {
    label: last,
    color: TEXT_DIM,
    bg: '#1e273866',
    tooltip: `Last status: ${last}`,
  };
}
