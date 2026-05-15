// Travel monitor settings — PROGRAM §46.6 Q9.3.
//
// Two inputs:
//
//   1. TripIt iCal URL — copy from TripIt account UI:
//      Settings → Calendar Sync → "Copy to your calendar".
//      Looks like: https://www.tripit.com/feed/ical/...
//
//   2. Aviationstack API key (optional) — for live flight-status
//      delays/gates. Free tier 100 calls/month is plenty for
//      personal use; the travel runner caps at 3 calls/cycle.
//
// Both persist via runtime_settings (no gateway restart). Empty
// values disable the corresponding source; the travel module
// degrades gracefully (TripIt absent = no upcoming-trip data;
// API key absent = no live flight status, but trip data still
// surfaces).

import { useState, useEffect } from 'react';
import { api } from '../api/client';
import type { RuntimeSettings } from '../api/queries';

const TEXT_DIM = '#7a8599';
const TEXT_BRIGHT = '#e2e8f0';
const WARN = '#f87171';
const WARN_BG = '#7f1d1d22';
const OK = '#34d399';

export function TravelCard({
  settings,
  onSettingsChange,
}: {
  settings: RuntimeSettings | Partial<RuntimeSettings>;
  onSettingsChange: () => void;
}) {
  const [url, setUrl] = useState<string>('');
  const [apiKey, setApiKey] = useState<string>('');
  const [showKey, setShowKey] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [savedAt, setSavedAt] = useState<number | null>(null);

  // Hydrate from server snapshot when it changes
  useEffect(() => {
    setUrl(settings.tripit_ical_url ?? '');
    setApiKey(settings.aviationstack_api_key ?? '');
  }, [settings.tripit_ical_url, settings.aviationstack_api_key]);

  const save = async (patch: Record<string, string>) => {
    setError(null);
    try {
      await api('/api/cp/settings', {
        method: 'POST',
        body: JSON.stringify(patch),
      });
      setSavedAt(Date.now());
      onSettingsChange();
    } catch (e) {
      setError(String(e));
    }
  };

  const tripitConfigured = Boolean(url && url.trim());
  const keyConfigured = Boolean(apiKey && apiKey.trim());

  return (
    <div
      className="rounded-lg p-4 border space-y-4"
      style={{ background: '#111820', borderColor: '#1e2738' }}
    >
      <div>
        <h2 className="text-sm font-medium" style={{ color: TEXT_BRIGHT }}>
          Travel monitor (PROGRAM §46.6)
        </h2>
        <p className="text-[10px] mt-1" style={{ color: TEXT_DIM }}>
          Watches your upcoming trips via TripIt iCal feed; cross-references
          with calendar/email; surfaces in daily briefing's "🛫 Travel"
          section. Both inputs persist via{' '}
          <code>workspace/runtime_settings.json</code> — no gateway restart.
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
      {savedAt && Date.now() - savedAt < 4000 && (
        <div
          className="text-xs p-1.5 rounded"
          style={{ color: OK, background: '#14532d22' }}
        >
          ✓ Saved
        </div>
      )}

      {/* TripIt iCal URL */}
      <div>
        <label
          className="text-xs block mb-1"
          style={{ color: TEXT_BRIGHT, fontWeight: 500 }}
        >
          TripIt iCal feed URL
          {tripitConfigured && (
            <span className="ml-2 text-[10px]" style={{ color: OK }}>
              ● configured
            </span>
          )}
        </label>
        <input
          type="url"
          placeholder="https://www.tripit.com/feed/ical/your-token-here.ics"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          className="w-full px-2 py-1 text-xs bg-[#0a0e14] border border-[#1e2738] rounded font-mono text-[#cbd5e1]"
        />
        <p className="text-[10px] mt-1" style={{ color: TEXT_DIM }}>
          TripIt → Settings → Calendar Sync → "Copy to your calendar" →
          paste here. URL must start with <code>https://</code> and contain
          <code> tripit</code> in the hostname.
        </p>
        <div className="flex gap-2 mt-2">
          <button
            onClick={() => save({ tripit_ical_url: url })}
            className="text-xs px-3 py-1 rounded bg-[#60a5fa]/15 text-[#60a5fa] border border-[#60a5fa]/30 hover:bg-[#60a5fa]/25"
          >
            Save URL
          </button>
          {tripitConfigured && (
            <button
              onClick={() => {
                setUrl('');
                save({ tripit_ical_url: '' });
              }}
              className="text-xs px-3 py-1 rounded text-[#7a8599] hover:text-[#e2e8f0] hover:bg-[#1e2738] border border-[#1e2738]"
            >
              Clear
            </button>
          )}
        </div>
      </div>

      {/* Aviationstack key */}
      <div>
        <label
          className="text-xs block mb-1"
          style={{ color: TEXT_BRIGHT, fontWeight: 500 }}
        >
          Aviationstack API key{' '}
          <span className="text-[10px]" style={{ color: TEXT_DIM }}>
            (optional — live flight status)
          </span>
          {keyConfigured && (
            <span className="ml-2 text-[10px]" style={{ color: OK }}>
              ● configured
            </span>
          )}
        </label>
        <div className="flex gap-1">
          <input
            type={showKey ? 'text' : 'password'}
            placeholder="32-char API key from aviationstack.com"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            className="flex-1 px-2 py-1 text-xs bg-[#0a0e14] border border-[#1e2738] rounded font-mono text-[#cbd5e1]"
          />
          <button
            onClick={() => setShowKey(!showKey)}
            className="text-xs px-2 rounded text-[#7a8599] hover:text-[#e2e8f0] hover:bg-[#1e2738] border border-[#1e2738]"
          >
            {showKey ? 'hide' : 'show'}
          </button>
        </div>
        <p className="text-[10px] mt-1" style={{ color: TEXT_DIM }}>
          Free tier: 100 calls/month. The travel runner caps at 3
          calls/cycle (every 6 hours). Empty = TripIt segments still
          surface, just without live status.
        </p>
        <div className="flex gap-2 mt-2">
          <button
            onClick={() => save({ aviationstack_api_key: apiKey })}
            className="text-xs px-3 py-1 rounded bg-[#60a5fa]/15 text-[#60a5fa] border border-[#60a5fa]/30 hover:bg-[#60a5fa]/25"
          >
            Save key
          </button>
          {keyConfigured && (
            <button
              onClick={() => {
                setApiKey('');
                save({ aviationstack_api_key: '' });
              }}
              className="text-xs px-3 py-1 rounded text-[#7a8599] hover:text-[#e2e8f0] hover:bg-[#1e2738] border border-[#1e2738]"
            >
              Clear
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
