// Resilience drills v2 — operator surface for the Q18 state machine
// + baseline-ratification flow (PROGRAM §57).
//
// Replaces the legacy /cp/settings ResilienceDrillsCard list-only view
// with a dedicated page that exposes:
//   * Per-drill state badge (HEALTHY / WATCH / DEGRADED / QUARANTINED
//     / WARMING_UP / MUTED) with consecutive-failure count + next-
//     attempt time.
//   * Drill detail drawer: state, baseline (or "Ratify baseline" form),
//     recent observations, recent results, last traceback (when
//     QUARANTINED), state transitions.
//   * Per-drill actions: Run now / Unquarantine / Mute / Unmute /
//     Ratify baseline.
//
// Backend at /api/cp/drills/* — see
// app/control_plane/dashboard_routes_sentience_drills.py.

import { useState } from 'react';
import { Skeleton } from './ui/Skeleton';
import {
  useDrillsRegistryQuery,
  useDrillDetailQuery,
  useDrillRunMutation,
  useRatifyBaselineMutation,
  useUnquarantineDrillMutation,
  useMuteDrillMutation,
  useUnmuteDrillMutation,
  type DrillRegistryEntry,
  type DrillBaseline,
  type DrillObservation,
} from '../api/queries';

const STATE_COLORS: Record<string, { bg: string; text: string; border: string }> = {
  healthy: { bg: 'bg-green-50', text: 'text-green-800', border: 'border-green-200' },
  warming_up: { bg: 'bg-blue-50', text: 'text-blue-800', border: 'border-blue-200' },
  watch: { bg: 'bg-yellow-50', text: 'text-yellow-800', border: 'border-yellow-200' },
  degraded: { bg: 'bg-orange-50', text: 'text-orange-800', border: 'border-orange-200' },
  quarantined: { bg: 'bg-red-50', text: 'text-red-800', border: 'border-red-200' },
  muted: { bg: 'bg-gray-50', text: 'text-gray-700', border: 'border-gray-200' },
};

function StateBadge({ state }: { state: string }) {
  const c = STATE_COLORS[state] ?? STATE_COLORS.healthy;
  return (
    <span
      className={`inline-flex items-center rounded px-2 py-0.5 text-xs font-medium ${c.bg} ${c.text} border ${c.border}`}
    >
      {state}
    </span>
  );
}

function timeAgo(iso: string | null | undefined): string {
  if (!iso) return '—';
  const then = new Date(iso).getTime();
  const now = Date.now();
  const diff = now - then;
  if (diff < 0) {
    const fwd = -diff;
    if (fwd < 3600_000) return `in ${Math.round(fwd / 60_000)}m`;
    if (fwd < 86_400_000) return `in ${Math.round(fwd / 3_600_000)}h`;
    return `in ${Math.round(fwd / 86_400_000)}d`;
  }
  if (diff < 60_000) return 'just now';
  if (diff < 3600_000) return `${Math.round(diff / 60_000)}m ago`;
  if (diff < 86_400_000) return `${Math.round(diff / 3_600_000)}h ago`;
  return `${Math.round(diff / 86_400_000)}d ago`;
}

function DrillRow({
  drill,
  onSelect,
  isSelected,
}: {
  drill: DrillRegistryEntry;
  onSelect: () => void;
  isSelected: boolean;
}) {
  return (
    <button
      onClick={onSelect}
      className={`w-full text-left rounded border px-3 py-2 hover:bg-gray-50 ${
        isSelected ? 'border-blue-400 bg-blue-50/30' : 'border-gray-200'
      }`}
    >
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <span className="font-mono text-sm truncate">{drill.name}</span>
          <StateBadge state={drill.state} />
          {drill.risk === 'high' && (
            <span className="text-xs rounded bg-red-100 text-red-800 px-1.5">
              HIGH
            </span>
          )}
          {drill.has_baseline ? (
            <span className="text-xs rounded bg-green-100 text-green-700 px-1.5">
              baseline ✓
            </span>
          ) : (
            <span className="text-xs rounded bg-amber-100 text-amber-800 px-1.5">
              no baseline
            </span>
          )}
        </div>
        <div className="text-xs text-gray-500 shrink-0">
          {drill.consecutive_failures > 0 && (
            <span className="mr-2 text-orange-600">
              {drill.consecutive_failures} fail
              {drill.consecutive_failures !== 1 ? 's' : ''}
            </span>
          )}
          last: {timeAgo(drill.last_run_at)}
          {drill.next_attempt_after && (
            <span className="ml-2">next: {timeAgo(drill.next_attempt_after)}</span>
          )}
        </div>
      </div>
      {drill.last_failure_summary && drill.state !== 'healthy' && (
        <div className="mt-1 text-xs text-gray-600 truncate">
          {drill.last_failure_class}: {drill.last_failure_summary}
        </div>
      )}
    </button>
  );
}

function MeasurementsTable({
  measurements,
}: {
  measurements: Record<string, unknown>;
}) {
  const keys = Object.keys(measurements);
  if (keys.length === 0) return <p className="text-sm text-gray-500">(empty)</p>;
  return (
    <table className="w-full text-xs">
      <tbody>
        {keys.map((k) => (
          <tr key={k} className="border-b border-gray-100 last:border-0">
            <td className="py-1 pr-3 font-mono text-gray-600">{k}</td>
            <td className="py-1 font-mono">
              {JSON.stringify(measurements[k])}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function RatifyBaselineForm({
  drillName,
  recentObservations,
  onClose,
}: {
  drillName: string;
  recentObservations: DrillObservation[];
  onClose: () => void;
}) {
  const ratify = useRatifyBaselineMutation();
  const [notes, setNotes] = useState('');
  // Per-key tolerance picker: default 'exact'; operator can change to
  // 'min', 'max', 'subset_of', 'superset_of' via the dropdown.
  const [tolerances, setTolerances] = useState<
    Record<string, { rule: string; value?: unknown }>
  >({});

  const latest = recentObservations[0];
  if (!latest) {
    return (
      <div className="rounded border border-amber-200 bg-amber-50 p-3 text-sm">
        No observations yet. Run the drill at least once to produce an
        observation, then ratify.
      </div>
    );
  }

  const keys = Object.keys(latest.measurements ?? {});

  return (
    <div className="space-y-3 rounded border border-blue-200 bg-blue-50/40 p-3">
      <div>
        <h3 className="text-sm font-medium">
          Ratify baseline from {timeAgo(latest.observed_at)} observation
        </h3>
        <p className="text-xs text-gray-600 mt-1">
          Future runs will compare to this baseline. Pick the tolerance
          rule for each measurement.
        </p>
      </div>
      <div className="rounded border bg-white p-2 max-h-64 overflow-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="text-gray-500">
              <th className="text-left pb-1">Key</th>
              <th className="text-left pb-1">Observed</th>
              <th className="text-left pb-1">Rule</th>
            </tr>
          </thead>
          <tbody>
            {keys.map((k) => {
              const v = latest.measurements[k];
              const t = tolerances[k]?.rule ?? 'exact';
              return (
                <tr key={k} className="border-t border-gray-100">
                  <td className="py-1 pr-3 font-mono">{k}</td>
                  <td className="py-1 pr-3 font-mono">{JSON.stringify(v)}</td>
                  <td className="py-1">
                    <select
                      value={t}
                      onChange={(e) =>
                        setTolerances((prev) => ({
                          ...prev,
                          [k]: { rule: e.target.value, value: v },
                        }))
                      }
                      className="text-xs border rounded px-1 py-0.5"
                    >
                      <option value="exact">exact</option>
                      <option value="min">min</option>
                      <option value="max">max</option>
                      <option value="subset_of">subset_of</option>
                      <option value="superset_of">superset_of</option>
                    </select>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div>
        <label className="block text-xs text-gray-600 mb-1">
          Notes (why is this acceptable?)
        </label>
        <textarea
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          className="w-full text-sm border rounded p-2"
          rows={2}
          placeholder="e.g. single fallback fine for our posture"
        />
      </div>
      <div className="flex gap-2">
        <button
          onClick={() => {
            const tolerancesPayload: Record<string, Record<string, unknown>> = {};
            for (const k of Object.keys(tolerances)) {
              const t = tolerances[k];
              if (t.rule === 'min' || t.rule === 'max') {
                tolerancesPayload[k] = { rule: t.rule, value: t.value };
              } else if (t.rule === 'subset_of' || t.rule === 'superset_of') {
                tolerancesPayload[k] = { rule: t.rule, value: t.value };
              } else {
                tolerancesPayload[k] = { rule: 'exact' };
              }
            }
            ratify.mutate(
              {
                name: drillName,
                payload: {
                  observation_at: latest.observed_at,
                  tolerances: tolerancesPayload,
                  notes,
                  operator: 'operator-react',
                },
              },
              { onSuccess: onClose },
            );
          }}
          disabled={ratify.isPending}
          className="px-3 py-1.5 bg-blue-600 text-white text-sm rounded hover:bg-blue-700 disabled:opacity-50"
        >
          {ratify.isPending ? 'Ratifying…' : 'Ratify baseline'}
        </button>
        <button
          onClick={onClose}
          className="px-3 py-1.5 border text-sm rounded hover:bg-gray-50"
        >
          Cancel
        </button>
      </div>
    </div>
  );
}

function DrillDetailDrawer({
  drillName,
  onClose,
}: {
  drillName: string;
  onClose: () => void;
}) {
  const detail = useDrillDetailQuery(drillName);
  const run = useDrillRunMutation();
  const unquarantine = useUnquarantineDrillMutation();
  const mute = useMuteDrillMutation();
  const unmute = useUnmuteDrillMutation();
  const [ratifying, setRatifying] = useState(false);

  if (detail.isLoading) return <Skeleton />;
  if (detail.error || !detail.data) {
    return (
      <div className="text-sm text-red-700">
        Failed to load drill detail.
      </div>
    );
  }
  const d = detail.data;
  const state = d.state.state;
  const baseline: DrillBaseline | null = d.baseline;
  const tracebackVisible = state === 'quarantined' && d.state.last_traceback;

  return (
    <div className="rounded border border-gray-200 bg-white p-4 space-y-3">
      <div className="flex items-start justify-between gap-2">
        <div>
          <h2 className="text-lg font-medium flex items-center gap-2">
            {d.spec.name}
            <StateBadge state={state} />
          </h2>
          <p className="text-sm text-gray-600 mt-1">{d.spec.description}</p>
        </div>
        <button
          onClick={onClose}
          className="text-gray-400 hover:text-gray-700 text-sm"
        >
          Close ✕
        </button>
      </div>

      {/* State details */}
      <div className="grid grid-cols-2 gap-3 text-sm">
        <div>
          <div className="text-xs text-gray-500">Last run</div>
          <div>{timeAgo(d.state.last_run_at)}</div>
        </div>
        <div>
          <div className="text-xs text-gray-500">Last success</div>
          <div>{timeAgo(d.state.last_success_at)}</div>
        </div>
        <div>
          <div className="text-xs text-gray-500">Consecutive failures</div>
          <div>{d.state.consecutive_failures}</div>
        </div>
        <div>
          <div className="text-xs text-gray-500">Next attempt</div>
          <div>
            {d.is_runnable_now
              ? 'now'
              : `${timeAgo(d.state.next_attempt_after)} (${d.runnable_reason})`}
          </div>
        </div>
      </div>

      {/* Actions */}
      <div className="flex flex-wrap gap-2">
        <button
          onClick={() => run.mutate(drillName)}
          disabled={run.isPending}
          className="px-3 py-1 bg-blue-600 text-white text-sm rounded hover:bg-blue-700 disabled:opacity-50"
        >
          {run.isPending ? 'Running…' : 'Run now'}
        </button>
        {state === 'quarantined' && (
          <button
            onClick={() =>
              unquarantine.mutate({
                name: drillName,
                reason: 'operator unquarantine via /cp/drills',
              })
            }
            disabled={unquarantine.isPending}
            className="px-3 py-1 bg-red-600 text-white text-sm rounded hover:bg-red-700 disabled:opacity-50"
          >
            Unquarantine
          </button>
        )}
        {state !== 'muted' ? (
          <button
            onClick={() => {
              const r = window.prompt('Mute reason?', '');
              if (r != null)
                mute.mutate({ name: drillName, reason: r });
            }}
            className="px-3 py-1 border text-sm rounded hover:bg-gray-50"
          >
            Mute
          </button>
        ) : (
          <button
            onClick={() => unmute.mutate(drillName)}
            className="px-3 py-1 border text-sm rounded hover:bg-gray-50"
          >
            Unmute
          </button>
        )}
        {!baseline && d.recent_observations.length > 0 && (
          <button
            onClick={() => setRatifying(true)}
            className="px-3 py-1 border border-blue-400 text-blue-700 text-sm rounded hover:bg-blue-50"
          >
            Ratify baseline
          </button>
        )}
      </div>

      {/* Traceback (quarantined) */}
      {tracebackVisible && (
        <details className="rounded border border-red-200 bg-red-50 p-2">
          <summary className="text-sm font-medium text-red-800 cursor-pointer">
            Last traceback ({d.state.last_failure_class})
          </summary>
          <pre className="mt-2 text-xs overflow-auto whitespace-pre-wrap">
            {d.state.last_traceback}
          </pre>
        </details>
      )}

      {/* Baseline */}
      <div>
        <h3 className="text-sm font-medium mb-1">Baseline</h3>
        {ratifying ? (
          <RatifyBaselineForm
            drillName={drillName}
            recentObservations={d.recent_observations}
            onClose={() => setRatifying(false)}
          />
        ) : baseline ? (
          <div className="rounded border bg-gray-50 p-2">
            <div className="text-xs text-gray-500 mb-2">
              Ratified {timeAgo(baseline.ratified_at)} by{' '}
              {baseline.ratified_by}
              {baseline.notes && <> — {baseline.notes}</>}
            </div>
            <MeasurementsTable measurements={baseline.measurements} />
          </div>
        ) : (
          <p className="text-sm text-gray-500">
            No baseline ratified. Drill is in {state === 'warming_up' ? 'warmup — observations are accumulating' : 'pre-baseline state'}.
          </p>
        )}
      </div>

      {/* Recent observations */}
      <div>
        <h3 className="text-sm font-medium mb-1">
          Recent observations ({d.recent_observations.length})
        </h3>
        {d.recent_observations.length === 0 ? (
          <p className="text-sm text-gray-500">No observations recorded.</p>
        ) : (
          <div className="space-y-2">
            {d.recent_observations.slice(0, 5).map((obs, i) => (
              <details key={i} className="rounded border p-2">
                <summary className="text-xs cursor-pointer flex items-center gap-2">
                  <span className="text-gray-500">{timeAgo(obs.observed_at)}</span>
                  <span className="font-mono">{obs.status}</span>
                  {obs.failure_class && (
                    <span className="text-orange-600">
                      {obs.failure_class}
                    </span>
                  )}
                </summary>
                <div className="mt-2">
                  <MeasurementsTable measurements={obs.measurements} />
                </div>
              </details>
            ))}
          </div>
        )}
      </div>

      {/* State transitions */}
      {d.state.transitions.length > 0 && (
        <details className="rounded border p-2">
          <summary className="text-sm font-medium cursor-pointer">
            State transitions ({d.state.transitions.length})
          </summary>
          <ul className="mt-2 space-y-1 text-xs">
            {[...d.state.transitions].reverse().map((t, i) => (
              <li key={i} className="text-gray-600">
                <span className="text-gray-400">{timeAgo(t.at)}</span>{' '}
                <span className="font-mono">
                  {t.from_state} → {t.to_state}
                </span>{' '}
                <span className="text-gray-500">({t.triggered_by})</span>
                {t.reason && <> — {t.reason}</>}
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}

export function DrillsPage() {
  const registry = useDrillsRegistryQuery();
  const [selected, setSelected] = useState<string | null>(null);

  if (registry.isLoading) return <Skeleton />;
  if (registry.error || !registry.data) {
    return (
      <div className="p-4 text-red-700">Failed to load drill registry.</div>
    );
  }
  if (registry.data.error) {
    return (
      <div className="p-4 text-red-700">
        Drill registry error: {registry.data.error}
      </div>
    );
  }

  const drills = registry.data.drills;
  const byState = {
    quarantined: drills.filter((d) => d.state === 'quarantined'),
    degraded: drills.filter((d) => d.state === 'degraded'),
    watch: drills.filter((d) => d.state === 'watch'),
    warming_up: drills.filter((d) => d.state === 'warming_up'),
    healthy: drills.filter((d) => d.state === 'healthy'),
    muted: drills.filter((d) => d.state === 'muted'),
  };

  return (
    <div className="p-4 max-w-6xl mx-auto">
      <header className="mb-4">
        <h1 className="text-2xl font-medium">Resilience drills</h1>
        <p className="text-sm text-gray-600 mt-1">
          State machine + baselines (PROGRAM §57). Each drill is an
          observer — operator-ratified baselines define what counts as a
          regression.
        </p>
      </header>

      <div className="grid lg:grid-cols-2 gap-4">
        <div className="space-y-4">
          {([
            ['Needs attention', [...byState.quarantined, ...byState.degraded, ...byState.watch]],
            ['Warming up', byState.warming_up],
            ['Healthy', byState.healthy],
            ['Muted', byState.muted],
          ] as const).map(([label, items]) =>
            items.length === 0 ? null : (
              <section key={label}>
                <h2 className="text-sm font-medium text-gray-700 mb-2">
                  {label} ({items.length})
                </h2>
                <div className="space-y-2">
                  {items.map((d) => (
                    <DrillRow
                      key={d.name}
                      drill={d}
                      onSelect={() => setSelected(d.name)}
                      isSelected={selected === d.name}
                    />
                  ))}
                </div>
              </section>
            ),
          )}
        </div>
        <div>
          {selected ? (
            <DrillDetailDrawer
              drillName={selected}
              onClose={() => setSelected(null)}
            />
          ) : (
            <div className="rounded border border-dashed border-gray-300 p-6 text-center text-sm text-gray-500">
              Select a drill to see state, baseline, and actions.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
