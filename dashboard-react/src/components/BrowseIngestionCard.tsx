// Browser-history ingestion card — PROGRAM §50 Q15 (Phase B+).
//
// Reads from /api/cp/browse/* directly (the master switch is env-only —
// BROWSE_INGESTION_ENABLED — not a runtime_settings flag). When the
// switch is OFF the card renders an explanatory block; when ON it
// shows stats, top topic clusters from the daily LLM batch, the
// blocklist, and the mute / forget actions.

import { useEffect, useState } from 'react';
import { api } from '../api/client';

const TEXT_DIM = '#7a8599';
const TEXT_BRIGHT = '#e2e8f0';
const WARN = '#f87171';
const WARN_BG = '#7f1d1d22';

type BrowseState = {
  enabled: boolean;
  stats: {
    total: number;
    window_days: number;
    by_browser: Record<string, number>;
    by_domain_top: Array<{ domain: string; count: number }>;
  };
};

type Category = {
  label: string;
  count: number;
  sample_titles: string[];
};

type Categories = {
  categories: Category[];
  window_days: number;
  days_with_data: string[];
};

type Blocklist = {
  seeded: string[];
  operator: string[];
};

export function BrowseIngestionCard() {
  const [state, setState] = useState<BrowseState | null>(null);
  const [categories, setCategories] = useState<Categories | null>(null);
  const [blocklist, setBlocklist] = useState<Blocklist | null>(null);
  const [muteInput, setMuteInput] = useState('');
  const [forgetPhrase, setForgetPhrase] = useState('');
  const [error, setError] = useState<string | null>(null);

  async function refresh() {
    try {
      const [s, c, b] = await Promise.all([
        api<BrowseState>('/api/cp/browse/state'),
        api<Categories>('/api/cp/browse/categories?days=7'),
        api<Blocklist>('/api/cp/browse/blocklist'),
      ]);
      setState(s);
      setCategories(c);
      setBlocklist(b);
      setError(null);
    } catch (e) {
      setError(String(e));
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  async function mute() {
    const d = muteInput.trim().toLowerCase();
    if (!d) return;
    try {
      await api('/api/cp/browse/mute', {
        method: 'POST',
        body: JSON.stringify({ domain: d }),
      });
      setMuteInput('');
      refresh();
    } catch (e) {
      setError(String(e));
    }
  }

  async function forgetAll() {
    try {
      await api('/api/cp/browse/forget', {
        method: 'POST',
        body: JSON.stringify({ scope: 'all' }),
      });
      setForgetPhrase('');
      refresh();
    } catch (e) {
      setError(String(e));
    }
  }

  return (
    <div
      className="rounded-lg p-4 border space-y-3"
      style={{ background: '#111820', borderColor: '#1e2738' }}
    >
      <div>
        <h2 className="text-sm font-medium" style={{ color: TEXT_BRIGHT }}>
          Browser-history ingestion (PROGRAM §50)
        </h2>
        <p className="text-[10px] mt-1" style={{ color: TEXT_DIM }}>
          Reads Safari + Chromium + Firefox history at idle, drops
          blocklisted domains, clusters titles into themes via a daily LLM
          batch, and feeds the 6th source into the interest model. Master
          switch is env-only (<code>BROWSE_INGESTION_ENABLED</code>).
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

      {state && !state.enabled && (
        <div
          className="text-xs p-2 rounded"
          style={{ color: TEXT_DIM, background: '#0c1118', border: '1px dashed #1e2738' }}
        >
          Browse ingestion is OFF. Set <code>BROWSE_INGESTION_ENABLED=true</code>{' '}
          in the gateway env and restart to opt in. Private/incognito browsing
          is automatically excluded by every browser.
        </div>
      )}

      {state && state.enabled && (
        <>
          <div className="grid grid-cols-3 gap-3 text-xs">
            <Stat label="Events (7d)" value={state.stats.total} />
            <Stat
              label="Browsers"
              value={Object.keys(state.stats.by_browser).length}
            />
            <Stat
              label="Top domains"
              value={state.stats.by_domain_top.length}
            />
          </div>

          {categories && categories.categories.length > 0 && (
            <div>
              <div className="text-xs font-medium" style={{ color: TEXT_BRIGHT }}>
                Top themes (7d, from daily LLM batch)
              </div>
              <ul className="mt-1 space-y-0.5">
                {categories.categories.slice(0, 8).map((c) => (
                  <li
                    key={c.label}
                    className="text-[11px]"
                    style={{ color: TEXT_DIM }}
                  >
                    <span style={{ color: TEXT_BRIGHT }}>{c.label}</span>
                    {' '}× {c.count}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {state.stats.by_domain_top.length > 0 && (
            <div>
              <div className="text-xs font-medium" style={{ color: TEXT_BRIGHT }}>
                Top domains (7d, raw)
              </div>
              <ul className="mt-1 space-y-0.5">
                {state.stats.by_domain_top.slice(0, 8).map((d) => (
                  <li
                    key={d.domain}
                    className="text-[11px]"
                    style={{ color: TEXT_DIM }}
                  >
                    {d.domain} × {d.count}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Mute */}
          <div>
            <div className="text-xs font-medium" style={{ color: TEXT_BRIGHT }}>
              Mute a domain
            </div>
            <div className="flex gap-2 mt-1">
              <input
                type="text"
                value={muteInput}
                onChange={(e) => setMuteInput(e.target.value)}
                placeholder="example.com"
                className="flex-1 bg-[#0c1118] border border-[#1e2738] rounded px-2 py-1 text-xs font-mono"
                style={{ color: TEXT_BRIGHT }}
              />
              <button
                onClick={mute}
                disabled={!muteInput.trim()}
                className="text-xs px-3 py-1 rounded border disabled:opacity-30"
                style={{ color: TEXT_BRIGHT, borderColor: '#1e2738' }}
              >
                Mute
              </button>
            </div>
            <p className="text-[10px] mt-1" style={{ color: TEXT_DIM }}>
              Future visits to this domain skip ingestion. Use{' '}
              <code>/browse forget {'<domain>'}</code> in Signal to also
              clear past entries.
            </p>
          </div>

          {/* Blocklist */}
          {blocklist && (
            <details>
              <summary
                className="text-xs cursor-pointer"
                style={{ color: TEXT_DIM }}
              >
                Blocklist ({blocklist.seeded.length} seeded +{' '}
                {blocklist.operator.length} operator)
              </summary>
              <ul className="mt-2 space-y-0.5">
                {blocklist.operator.map((d) => (
                  <li
                    key={`op-${d}`}
                    className="text-[11px]"
                    style={{ color: TEXT_BRIGHT }}
                  >
                    {d} <span style={{ color: TEXT_DIM }}>(operator)</span>
                  </li>
                ))}
                {blocklist.seeded.map((d) => (
                  <li
                    key={`seed-${d}`}
                    className="text-[11px]"
                    style={{ color: TEXT_DIM }}
                  >
                    {d} (seeded)
                  </li>
                ))}
              </ul>
            </details>
          )}

          {/* Forget all */}
          <div
            className="p-2 rounded"
            style={{ background: WARN_BG, border: `1px solid ${WARN}` }}
          >
            <div className="text-xs font-medium" style={{ color: WARN }}>
              ⚠️  Forget all browse history
            </div>
            <p className="text-[10px] mt-1" style={{ color: TEXT_DIM }}>
              Deletes every event + cursor state. Blocklist is preserved.
              Type <code>FORGET BROWSE HISTORY</code> below to confirm.
            </p>
            <div className="flex gap-2 mt-2">
              <input
                type="text"
                value={forgetPhrase}
                onChange={(e) => setForgetPhrase(e.target.value)}
                placeholder="FORGET BROWSE HISTORY"
                className="flex-1 bg-[#0c1118] border border-[#1e2738] rounded px-2 py-1 text-xs font-mono"
                style={{ color: TEXT_BRIGHT }}
              />
              <button
                onClick={forgetAll}
                disabled={forgetPhrase !== 'FORGET BROWSE HISTORY'}
                className="text-xs px-3 py-1 rounded border disabled:opacity-30"
                style={{ background: WARN_BG, color: WARN, borderColor: WARN }}
              >
                Forget
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: number | string }) {
  return (
    <div
      className="p-2 rounded"
      style={{ background: '#0c1118', border: '1px solid #1e2738' }}
    >
      <div className="text-[10px]" style={{ color: TEXT_DIM }}>
        {label}
      </div>
      <div className="text-base font-semibold" style={{ color: TEXT_BRIGHT }}>
        {value}
      </div>
    </div>
  );
}
