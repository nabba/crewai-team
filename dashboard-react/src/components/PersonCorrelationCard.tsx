// Person correlation settings — PROGRAM §42 Q4.2.
//
// Progressive disclosure: each level visible only when the previous
// is enabled. L4 + L4.4 require typed-phrase confirmation.

import { useState } from 'react';
import { api } from '../api/client';
import type { RuntimeSettings } from '../api/queries';

const TEXT_DIM = '#7a8599';
const TEXT_BRIGHT = '#e2e8f0';
const WARN = '#f87171';
const WARN_BG = '#7f1d1d22';

export function PersonCorrelationCard({
  settings,
  onSettingsChange,
}: {
  settings: RuntimeSettings | Partial<RuntimeSettings>;
  onSettingsChange: () => void;
}) {
  const [sgPhrase, setSgPhrase] = useState('');
  const [gsPhrase, setGsPhrase] = useState('');
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

  const L1 = !!settings.person_correlation_enabled;
  const L4 = !!settings.person_correlation_social_graph_enabled;
  const L44 = !!settings.graph_suggestions_enabled;

  return (
    <div
      className="rounded-lg p-4 border space-y-3"
      style={{ background: '#111820', borderColor: '#1e2738' }}
    >
      <div>
        <h2 className="text-sm font-medium" style={{ color: TEXT_BRIGHT }}>
          Person correlation (PROGRAM §42)
        </h2>
        <p className="text-[10px] mt-1" style={{ color: TEXT_DIM }}>
          Tracks people who appear in your inputs. Four levels of opt-in
          with progressively higher Goodhart risk. Read{' '}
          <code>docs/PERSON_CORRELATION.md</code> before enabling beyond
          Level 1.
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

      {/* L1 — Presence */}
      <Toggle
        label="Level 1 — Presence (counts only)"
        checked={L1}
        onChange={(v) => update({ person_correlation_enabled: v })}
        caveat="Begins tracking sender/attendee emails. No body parsing. You can mute or forget any person at any time. Disable to stop all data collection."
      />

      {L1 && (
        <div className="pl-4 space-y-3 border-l-2" style={{ borderColor: '#1e2738' }}>
          {/* L2 — Centrality */}
          <Toggle
            label="Level 2 — Centrality scores"
            checked={!!settings.person_centrality_enabled}
            onChange={(v) => update({ person_centrality_enabled: v })}
            caveat="Computes per-person score. ⚠️ Goodhart risk: scores can become targets. Three formulas you choose between."
          />
          {settings.person_centrality_enabled && (
            <div className="ml-4 text-xs space-y-1">
              <label style={{ color: TEXT_DIM }}>Formula:</label>
              <select
                value={settings.person_centrality_formula || 'frequency'}
                onChange={(e) =>
                  update({ person_centrality_formula: e.target.value })
                }
                className="bg-[#0c1118] border border-[#1e2738] rounded px-2 py-0.5"
                style={{ color: TEXT_BRIGHT }}
              >
                <option value="frequency">frequency</option>
                <option value="recency_weighted">recency_weighted</option>
                <option value="cross_modal">cross_modal</option>
              </select>
            </div>
          )}

          {/* L3 — Suggestions */}
          <Toggle
            label="Level 3 — Automatic suggestions"
            checked={!!settings.person_suggestions_enabled}
            onChange={(v) => update({ person_suggestions_enabled: v })}
            caveat="⚠️ PRESCRIPTIVE — system surfaces ≤3 social nudges per briefing. Always phrased as questions. Per-category opt-in."
          />
          {settings.person_suggestions_enabled && (
            <div className="ml-4 text-xs space-y-1">
              <SubToggle
                label="Dormancy nudge"
                checked={!!settings.person_suggestions_dormancy_enabled}
                onChange={(v) =>
                  update({ person_suggestions_dormancy_enabled: v })
                }
              />
              <SubToggle
                label="Responsiveness nudge"
                checked={!!settings.person_suggestions_responsiveness_enabled}
                onChange={(v) =>
                  update({ person_suggestions_responsiveness_enabled: v })
                }
              />
            </div>
          )}

          {/* L4 — Social graph */}
          <div>
            <div className="text-xs font-medium" style={{ color: TEXT_BRIGHT }}>
              Level 4 — Social graph
            </div>
            <div
              className="text-[10px] mt-0.5 p-2 rounded"
              style={{ color: WARN, background: WARN_BG }}
            >
              ⚠️ ⚠️ ⚠️ MOST INVASIVE. Builds a structural map of
              relationships from observation. Excluded from DR backups by
              default.
            </div>
            {!L4 ? (
              <div className="flex gap-2 mt-2">
                <input
                  type="text"
                  value={sgPhrase}
                  onChange={(e) => setSgPhrase(e.target.value)}
                  placeholder="Type: ENABLE SOCIAL GRAPH"
                  className="flex-1 bg-[#0c1118] border border-[#1e2738] rounded px-2 py-1 text-xs font-mono"
                  style={{ color: TEXT_BRIGHT }}
                />
                <button
                  onClick={() => {
                    update({
                      person_correlation_social_graph_enabled: true,
                      social_graph_confirm_phrase: sgPhrase,
                    });
                    setSgPhrase('');
                  }}
                  disabled={sgPhrase !== 'ENABLE SOCIAL GRAPH'}
                  className="text-xs px-3 py-1 rounded border disabled:opacity-30"
                  style={{ background: '#7f1d1d22', color: WARN, borderColor: WARN }}
                >
                  Enable
                </button>
              </div>
            ) : (
              <button
                onClick={() =>
                  update({ person_correlation_social_graph_enabled: false })
                }
                className="text-xs px-3 py-1 rounded mt-1 border"
                style={{ color: TEXT_DIM, borderColor: '#1e2738' }}
              >
                Disable
              </button>
            )}
          </div>

          {L4 && (
            <div className="ml-4 space-y-2" style={{ borderColor: '#1e2738' }}>
              <SubToggle
                label="L4.1 — Shortest-path queries (operator-initiated only)"
                checked={!!settings.graph_shortest_path_enabled}
                onChange={(v) => update({ graph_shortest_path_enabled: v })}
              />
              <SubToggle
                label="L4.2 — Community detection"
                checked={!!settings.graph_communities_enabled}
                onChange={(v) => update({ graph_communities_enabled: v })}
              />
              <SubToggle
                label="L4.3 — Bridge / cut-vertex identification"
                checked={!!settings.graph_bridges_enabled}
                onChange={(v) => update({ graph_bridges_enabled: v })}
              />

              {/* L4.4 — second typed-phrase gate */}
              <div>
                <div className="text-xs font-medium" style={{ color: TEXT_BRIGHT }}>
                  L4.4 — Graph-driven suggestions
                </div>
                <div
                  className="text-[10px] mt-0.5 p-2 rounded"
                  style={{ color: WARN, background: WARN_BG }}
                >
                  ⚠️ ⚠️ PRESCRIPTIVE. The system pushes suggestions based on
                  graph topology. Shares L3 rate limit (3/briefing total).
                </div>
                {!L44 ? (
                  <div className="flex gap-2 mt-2">
                    <input
                      type="text"
                      value={gsPhrase}
                      onChange={(e) => setGsPhrase(e.target.value)}
                      placeholder="Type: ENABLE GRAPH-DRIVEN SUGGESTIONS"
                      className="flex-1 bg-[#0c1118] border border-[#1e2738] rounded px-2 py-1 text-xs font-mono"
                      style={{ color: TEXT_BRIGHT }}
                    />
                    <button
                      onClick={() => {
                        update({
                          graph_suggestions_enabled: true,
                          graph_suggestions_confirm_phrase: gsPhrase,
                        });
                        setGsPhrase('');
                      }}
                      disabled={gsPhrase !== 'ENABLE GRAPH-DRIVEN SUGGESTIONS'}
                      className="text-xs px-3 py-1 rounded border disabled:opacity-30"
                      style={{ background: '#7f1d1d22', color: WARN, borderColor: WARN }}
                    >
                      Enable
                    </button>
                  </div>
                ) : (
                  <button
                    onClick={() => update({ graph_suggestions_enabled: false })}
                    className="text-xs px-3 py-1 rounded mt-1 border"
                    style={{ color: TEXT_DIM, borderColor: '#1e2738' }}
                  >
                    Disable
                  </button>
                )}

                {L44 && (
                  <div className="ml-4 mt-2 space-y-1">
                    <SubToggle
                      label="Cluster dormancy"
                      checked={!!settings.graph_suggestions_cluster_dormancy_enabled}
                      onChange={(v) =>
                        update({ graph_suggestions_cluster_dormancy_enabled: v })
                      }
                    />
                    <SubToggle
                      label="Bridge maintenance"
                      checked={
                        !!settings.graph_suggestions_bridge_maintenance_enabled
                      }
                      onChange={(v) =>
                        update({ graph_suggestions_bridge_maintenance_enabled: v })
                      }
                    />
                    <SubToggle
                      label="Weak-tie dormant"
                      checked={!!settings.graph_suggestions_weak_tie_enabled}
                      onChange={(v) => update({ graph_suggestions_weak_tie_enabled: v })}
                    />
                  </div>
                )}
              </div>
            </div>
          )}
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

function SubToggle({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <label className="flex items-center gap-2 text-xs cursor-pointer">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
      />
      <span style={{ color: TEXT_BRIGHT }}>{label}</span>
    </label>
  );
}
