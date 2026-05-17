// Source ledger health + switches — PROGRAM §56 iter-3.
//
// Surfaces per-KB ledger row counts, chain-verify status, compaction
// history, and off-host upload state. Eight master switches below.
// Endpoint: GET /api/cp/source-ledger/state (read-only, polled).

import { useEffect, useState } from 'react';
import { api } from '../api/client';
import type { RuntimeSettings } from '../api/queries';

const TEXT_DIM = '#7a8599';
const TEXT_BRIGHT = '#e2e8f0';
const WARN = '#f87171';
const WARN_BG = '#7f1d1d22';
const OK = '#4ade80';

type KBState = {
  name: string;
  ledger_rows?: number;
  ledger_bytes?: number;
  ledger_age_s?: number | null;
  chain_ok?: boolean | null;
  chain_first_bad_row?: number;
  chain_first_bad_reason?: string;
  history_count?: number;
  history_bytes?: number;
  last_compaction_at?: number | null;
  offhost?: { [key: string]: { last_upload_ts?: number; last_object_key?: string } | null };
  error?: string;
};

type LedgerState = {
  kbs: KBState[];
  switches: { [key: string]: boolean };
  as_of?: string;
  error?: string;
};

function formatBytes(n: number | undefined): string {
  if (!n) return '—';
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

function formatAge(s: number | null | undefined): string {
  if (s === null || s === undefined) return '—';
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

function formatTs(ts: number | null | undefined): string {
  if (!ts) return 'never';
  return formatAge(Math.floor(Date.now() / 1000 - ts));
}

export function SourceLedgerCard({
  settings,
  onSettingsChange,
}: {
  settings: RuntimeSettings | Partial<RuntimeSettings>;
  onSettingsChange: () => void;
}) {
  const [state, setState] = useState<LedgerState | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = async () => {
    try {
      const data = await api<LedgerState>('/api/cp/source-ledger/state');
      setState(data);
      setError(null);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void refresh();
    const t = setInterval(() => void refresh(), 30_000);
    return () => clearInterval(t);
  }, []);

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

  return (
    <div
      className="rounded-lg p-4 border space-y-3"
      style={{ background: '#111820', borderColor: '#1e2738' }}
    >
      <div>
        <h2 className="text-sm font-medium" style={{ color: TEXT_BRIGHT }}>
          Source ledger health (PROGRAM §56)
        </h2>
        <p className="text-[10px] mt-1" style={{ color: TEXT_DIM }}>
          Per-KB append-only canonical source of truth that makes
          chromadb files purely cacheable. Replay reconstructs any KB
          from its ledger with the current embed model. See{' '}
          <code>docs/SOURCE_LEDGER.md</code>.
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

      {loading && !state ? (
        <div className="text-xs" style={{ color: TEXT_DIM }}>
          Loading…
        </div>
      ) : state?.kbs && state.kbs.length > 0 ? (
        <div className="text-xs">
          <table className="w-full" style={{ borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ color: TEXT_DIM, borderBottom: `1px solid #1e2738` }}>
                <th className="text-left py-1 pr-2">KB</th>
                <th className="text-right py-1 pr-2">Rows</th>
                <th className="text-right py-1 pr-2">Size</th>
                <th className="text-right py-1 pr-2">Last write</th>
                <th className="text-left py-1 pr-2">Chain</th>
                <th className="text-right py-1 pr-2">History</th>
                <th className="text-right py-1 pr-2">Compacted</th>
                <th className="text-left py-1">Off-host</th>
              </tr>
            </thead>
            <tbody>
              {state.kbs.map((kb) => (
                <tr key={kb.name} style={{ color: TEXT_BRIGHT }}>
                  <td className="py-1 pr-2 font-mono">{kb.name}</td>
                  <td className="py-1 pr-2 text-right">{kb.ledger_rows ?? '—'}</td>
                  <td className="py-1 pr-2 text-right">{formatBytes(kb.ledger_bytes)}</td>
                  <td className="py-1 pr-2 text-right" style={{ color: TEXT_DIM }}>
                    {formatAge(kb.ledger_age_s)}
                  </td>
                  <td className="py-1 pr-2">
                    {kb.chain_ok === true ? (
                      <span style={{ color: OK }}>ok</span>
                    ) : kb.chain_ok === false ? (
                      <span style={{ color: WARN }} title={kb.chain_first_bad_reason}>
                        broken @ {kb.chain_first_bad_row}
                      </span>
                    ) : (
                      <span style={{ color: TEXT_DIM }}>skipped</span>
                    )}
                  </td>
                  <td className="py-1 pr-2 text-right" style={{ color: TEXT_DIM }}>
                    {kb.history_count ?? 0}
                    {kb.history_bytes ? ` (${formatBytes(kb.history_bytes)})` : ''}
                  </td>
                  <td className="py-1 pr-2 text-right" style={{ color: TEXT_DIM }}>
                    {formatTs(kb.last_compaction_at)}
                  </td>
                  <td className="py-1" style={{ color: TEXT_DIM }}>
                    {kb.offhost?.s3 ? `s3:${formatTs(kb.offhost.s3.last_upload_ts)}` : ''}
                    {kb.offhost?.s3 && kb.offhost?.gdrive ? ' · ' : ''}
                    {kb.offhost?.gdrive ? `gd:${formatTs(kb.offhost.gdrive.last_upload_ts)}` : ''}
                    {!kb.offhost?.s3 && !kb.offhost?.gdrive ? '—' : ''}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="text-xs" style={{ color: TEXT_DIM }}>
          No KBs found yet — daemon bootstraps ~5 min after gateway start.
        </div>
      )}

      <div
        className="pt-3 grid grid-cols-2 gap-2"
        style={{ borderTop: `1px solid #1e2738` }}
      >
        <Toggle
          label="Source ledger (master)"
          checked={settings.chromadb_source_ledger_enabled !== false}
          onChange={(v) => update({ chromadb_source_ledger_enabled: v })}
        />
        <Toggle
          label="Daily bootstrap back-fill"
          checked={settings.chromadb_ledger_bootstrap_enabled !== false}
          onChange={(v) => update({ chromadb_ledger_bootstrap_enabled: v })}
        />
        <Toggle
          label="Auto-replay on drift"
          checked={settings.chromadb_ledger_drift_replay_enabled !== false}
          onChange={(v) => update({ chromadb_ledger_drift_replay_enabled: v })}
        />
        <Toggle
          label="Weekly compaction"
          checked={settings.chromadb_ledger_compaction_enabled !== false}
          onChange={(v) => update({ chromadb_ledger_compaction_enabled: v })}
        />
        <Toggle
          label="S3 off-host upload"
          checked={settings.chromadb_ledger_s3_upload_enabled === true}
          onChange={(v) => update({ chromadb_ledger_s3_upload_enabled: v })}
          caveat="Requires LEDGER_S3_BUCKET + credentials in .env"
        />
        <Toggle
          label="Google Drive off-host"
          checked={settings.chromadb_ledger_gdrive_upload_enabled === true}
          onChange={(v) => update({ chromadb_ledger_gdrive_upload_enabled: v })}
          caveat="Uses existing Google Workspace OAuth"
        />
        <Toggle
          label="Quarterly replay drill"
          checked={settings.drill_source_ledger_replay_enabled !== false}
          onChange={(v) => update({ drill_source_ledger_replay_enabled: v })}
        />
        <Toggle
          label="Embed-rotation drill"
          checked={settings.drill_embedding_rotation_enabled !== false}
          onChange={(v) => update({ drill_embedding_rotation_enabled: v })}
        />
      </div>
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
        <span style={{ color: TEXT_BRIGHT }}>{label}</span>
      </label>
      {caveat && (
        <p className="text-[10px] ml-5 mt-0.5" style={{ color: TEXT_DIM }}>
          {caveat}
        </p>
      )}
    </div>
  );
}
