import { useEffect, useState } from 'react';
import { ErrorPanel } from './ui/ErrorPanel';
import {
  useLifeCompanionQuery,
  useUpdateLifeCompanionFeature,
  type LifeCompanionFeature,
  type LifeCompanionTunable,
} from '../api/queries';

// Note: POST to /config/life_companion requires a gateway bearer secret.
// The dashboard server (server.mjs) injects `Authorization: Bearer
// $GATEWAY_SECRET` on outbound requests when the env var is set; the Vite
// dev server does the same. Without that, the save buttons return 401.

export function LifeCompanionPage() {
  const stateQ = useLifeCompanionQuery();

  if (stateQ.isLoading) {
    return (
      <div className="bg-[#111820] border border-[#1e2738] rounded-xl p-4">
        <div className="text-[#7a8599] text-sm">Loading life-companion state…</div>
      </div>
    );
  }
  if (stateQ.error) {
    return <ErrorPanel error={stateQ.error} onRetry={stateQ.refetch} />;
  }
  if (!stateQ.data) return null;

  const { master_enabled, features } = stateQ.data;

  return (
    <div className="space-y-4 max-w-4xl">
      <div>
        <h1 className="text-xl font-semibold text-[#e2e8f0]">
          Life Companion control panel
        </h1>
        <p className="text-xs text-[#7a8599] mt-1">
          Per-job on/off toggles + tunable env vars. Changes persist
          across restarts and are audited as{' '}
          <code className="text-[#60a5fa]">life_companion_settings_change</code>{' '}
          events.
        </p>
        {!master_enabled && (
          <div className="mt-2 text-xs px-3 py-2 rounded bg-[#fbbf24]/10 border border-[#fbbf24]/30 text-[#fbbf24]">
            ⚠ Master switch <code>LIFE_COMPANION_ENABLED=false</code> — all
            jobs are off regardless of per-feature toggles below. Flip the
            env var on the gateway to re-enable the subsystem.
          </div>
        )}
      </div>

      {features.map((feat) => (
        <FeatureCard key={feat.key} feature={feat} />
      ))}
    </div>
  );
}

// ── Feature card ───────────────────────────────────────────────────────────

function FeatureCard({ feature }: { feature: LifeCompanionFeature }) {
  const update = useUpdateLifeCompanionFeature();

  // Tunable edit buffer: mirror current values; user edits land here
  // before "Save" persists.  Reset whenever the underlying feature
  // state changes (e.g. after a successful save returns the new value).
  const [tunBuf, setTunBuf] = useState<Record<string, string>>(() =>
    Object.fromEntries(feature.tunables.map((t) => [t.env_key, t.current_value])),
  );
  useEffect(() => {
    setTunBuf(
      Object.fromEntries(
        feature.tunables.map((t) => [t.env_key, t.current_value]),
      ),
    );
  }, [feature.tunables]);

  const dirty = feature.tunables.some(
    (t) => tunBuf[t.env_key] !== t.current_value,
  );

  const flipEnabled = async () => {
    await update.mutateAsync({
      feature_key: feature.key,
      enabled: !feature.enabled,
    });
  };

  const saveTunables = async () => {
    // Only send keys that actually changed.  Empty string == clear
    // override (server-side semantics).
    const changed: Record<string, string> = {};
    for (const t of feature.tunables) {
      if (tunBuf[t.env_key] !== t.current_value) {
        changed[t.env_key] = tunBuf[t.env_key] ?? '';
      }
    }
    if (Object.keys(changed).length === 0) return;
    await update.mutateAsync({
      feature_key: feature.key,
      tunables: changed,
    });
  };

  const resetEnabledOverride = async () => {
    await update.mutateAsync({
      feature_key: feature.key,
      enabled: null,
    });
  };

  return (
    <div className="bg-[#111820] border border-[#1e2738] rounded-xl p-4 space-y-3">
      {/* Header row: name + status pill + toggle */}
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 flex-wrap">
            <h2 className="text-base font-semibold text-[#e2e8f0]">
              {feature.name}
            </h2>
            <SourcePill source={feature.enabled_source} />
          </div>
          <p className="text-xs text-[#7a8599] mt-1">{feature.description}</p>
          <div className="text-[10px] text-[#7a8599] mt-1 font-mono">
            job: {feature.job_name} · env: {feature.feature_env_key}
          </div>
        </div>
        <div className="flex items-center gap-2 flex-shrink-0">
          <Toggle
            on={feature.enabled}
            disabled={update.isPending}
            onClick={flipEnabled}
          />
          {feature.enabled_source === 'override' && (
            <button
              onClick={resetEnabledOverride}
              disabled={update.isPending}
              className="text-[10px] px-1.5 py-0.5 rounded text-[#7a8599] hover:text-[#e2e8f0] hover:bg-[#1e2738]"
              title="Clear override; revert to env / default"
            >
              reset
            </button>
          )}
        </div>
      </div>

      {/* Tunables */}
      {feature.tunables.length > 0 && (
        <div className="border-t border-[#1e2738] pt-3 space-y-2">
          {feature.tunables.map((t) => (
            <TunableRow
              key={t.env_key}
              tunable={t}
              value={tunBuf[t.env_key] ?? ''}
              onChange={(v) =>
                setTunBuf((prev) => ({ ...prev, [t.env_key]: v }))
              }
              disabled={update.isPending}
            />
          ))}
          {dirty && (
            <div className="flex items-center justify-end gap-2 pt-1">
              <button
                onClick={() =>
                  setTunBuf(
                    Object.fromEntries(
                      feature.tunables.map((t) => [t.env_key, t.current_value]),
                    ),
                  )
                }
                disabled={update.isPending}
                className="text-xs px-3 py-1 rounded text-[#7a8599] hover:text-[#e2e8f0] hover:bg-[#1e2738]"
              >
                Discard
              </button>
              <button
                onClick={saveTunables}
                disabled={update.isPending}
                className="text-xs px-3 py-1 rounded bg-[#60a5fa] text-[#0a0e14] hover:bg-[#60a5fa]/90 disabled:opacity-50"
              >
                {update.isPending ? 'Saving…' : 'Save tunables'}
              </button>
            </div>
          )}
        </div>
      )}

      {/* Error from last mutation */}
      {update.error && (
        <div className="text-xs text-[#f87171] bg-[#f87171]/10 border border-[#f87171]/30 rounded px-2 py-1">
          {(update.error as Error).message}
        </div>
      )}
    </div>
  );
}

// ── Tunable row ────────────────────────────────────────────────────────────

function TunableRow({
  tunable,
  value,
  onChange,
  disabled,
}: {
  tunable: LifeCompanionTunable;
  value: string;
  onChange: (v: string) => void;
  disabled: boolean;
}) {
  const isNumeric =
    tunable.type === 'int' || tunable.type === 'float' ||
    tunable.type === 'minutes' || tunable.type === 'hours' ||
    tunable.type === 'secs';

  const showRange =
    isNumeric && (tunable.min !== null || tunable.max !== null);

  return (
    <div className="grid grid-cols-[1fr_auto] gap-3 items-center">
      <div className="min-w-0">
        <div className="flex items-center gap-2 flex-wrap">
          <label className="text-xs font-medium text-[#e2e8f0]">
            {tunable.label}
          </label>
          <SourcePill source={tunable.value_source} />
          <span className="text-[10px] text-[#7a8599] font-mono">
            {tunable.env_key}
          </span>
        </div>
        <p className="text-[11px] text-[#7a8599] mt-0.5">
          {tunable.description}
          {showRange && (
            <span className="ml-1 text-[#7a8599]">
              (range:{' '}
              {tunable.min !== null ? tunable.min : '∞-'}
              {' / '}
              {tunable.max !== null ? tunable.max : '+∞'}
              ; default: {String(tunable.default)})
            </span>
          )}
        </p>
      </div>
      {tunable.options.length > 0 ? (
        <select
          value={value}
          disabled={disabled}
          onChange={(e) => onChange(e.target.value)}
          className="text-xs px-2 py-1 bg-[#0a0e14] border border-[#1e2738] rounded text-[#e2e8f0] focus:outline-none focus:border-[#60a5fa] w-32"
        >
          {tunable.options.map((opt) => (
            <option key={opt} value={opt}>
              {opt}
            </option>
          ))}
        </select>
      ) : (
        <input
          type={isNumeric ? 'number' : 'text'}
          value={value}
          disabled={disabled}
          min={tunable.min ?? undefined}
          max={tunable.max ?? undefined}
          step={tunable.type === 'float' ? 0.1 : 1}
          onChange={(e) => onChange(e.target.value)}
          className="text-xs px-2 py-1 bg-[#0a0e14] border border-[#1e2738] rounded text-[#e2e8f0] focus:outline-none focus:border-[#60a5fa] w-32"
        />
      )}
    </div>
  );
}

// ── Tiny UI bits ───────────────────────────────────────────────────────────

function Toggle({
  on,
  disabled,
  onClick,
}: {
  on: boolean;
  disabled: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      role="switch"
      aria-checked={on}
      className={`w-10 h-6 rounded-full transition-colors flex items-center px-0.5 ${
        on ? 'bg-[#34d399]' : 'bg-[#1e2738]'
      } disabled:opacity-50`}
    >
      <span
        className={`w-5 h-5 rounded-full bg-white transition-transform ${
          on ? 'translate-x-4' : 'translate-x-0'
        }`}
      />
    </button>
  );
}

function SourcePill({
  source,
}: {
  source: 'override' | 'env' | 'default';
}) {
  const styles = {
    override: 'bg-[#60a5fa]/15 text-[#60a5fa] border-[#60a5fa]/30',
    env: 'bg-[#7a8599]/15 text-[#7a8599] border-[#7a8599]/30',
    default: 'bg-[#7a8599]/10 text-[#7a8599] border-[#1e2738]',
  };
  const labels = {
    override: 'override',
    env: 'env',
    default: 'default',
  };
  return (
    <span
      className={`text-[10px] px-1.5 py-0.5 rounded border font-medium ${styles[source]}`}
      title={
        source === 'override'
          ? 'Set via this control panel — persisted in workspace/runtime_settings.json'
          : source === 'env'
          ? 'Set via gateway environment variable (boot default)'
          : 'Registry default — no env var, no override'
      }
    >
      {labels[source]}
    </span>
  );
}
