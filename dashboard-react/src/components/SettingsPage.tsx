import { useEffect, useState } from 'react';
import { ErrorPanel } from './ui/ErrorPanel';
import {
  useRuntimeSettingsQuery,
  useUpdateRuntimeSettings,
  useVapidPublicKeyQuery,
  useWebPushSubscriptionsQuery,
  useWebPushSubscribe,
  useWebPushUnsubscribe,
  useWebPushTest,
  type RuntimeSettings,
  type VoiceMode,
} from '../api/queries';
import {
  getPushSubscription,
  subscribeToPush,
  unsubscribeFromPush,
} from '../api/pwa';

// Note: POST to /config/runtime_settings requires a gateway bearer secret.
// The dashboard server (server.mjs) injects `Authorization: Bearer
// $GATEWAY_SECRET` on outbound requests when the env var is set; the Vite
// dev server does the same. Without that, the save buttons return 401.

export function SettingsPage() {
  const settingsQ = useRuntimeSettingsQuery();

  if (settingsQ.isLoading) {
    return (
      <div className="bg-[#111820] border border-[#1e2738] rounded-xl p-4">
        <div className="text-[#7a8599] text-sm">Loading runtime settings…</div>
      </div>
    );
  }
  if (settingsQ.error) {
    return <ErrorPanel error={settingsQ.error} onRetry={settingsQ.refetch} />;
  }
  if (!settingsQ.data) return null;

  return (
    <div className="space-y-4 max-w-3xl">
      <div>
        <h1 className="text-xl font-semibold text-[#e2e8f0]">Settings</h1>
        <p className="text-xs text-[#7a8599] mt-1">
          Personal-agent surface toggles. Changes persist across restarts and are
          audited as <code className="text-[#60a5fa]">runtime_settings_change</code> events.
        </p>
      </div>

      <VoiceModeCard settings={settingsQ.data} />
      <VisionComputerUseCard settings={settingsQ.data} />
      <ConciergePersonaCard settings={settingsQ.data} />
      <WebPushCard />
    </div>
  );
}

// ── Web Push (PWA notifications) ──────────────────────────────────────────

function WebPushCard() {
  const vapidQ = useVapidPublicKeyQuery();
  const subsQ = useWebPushSubscriptionsQuery();
  const subscribe = useWebPushSubscribe();
  const unsubscribe = useWebPushUnsubscribe();
  const test = useWebPushTest();
  const [thisDeviceSubscribed, setThisDeviceSubscribed] = useState<boolean | null>(null);
  const [feedback, setFeedback] = useState('');

  // Reflect the current browser's subscription state.
  useEffect(() => {
    let alive = true;
    getPushSubscription().then((s) => {
      if (alive) setThisDeviceSubscribed(s !== null);
    });
    return () => { alive = false; };
  }, [subsQ.data?.count]);

  const supported =
    typeof navigator !== 'undefined' &&
    'serviceWorker' in navigator &&
    'PushManager' in (typeof window !== 'undefined' ? window : ({} as Window));

  const vapid = vapidQ.data?.public_key ?? '';
  const configured = subsQ.data?.configured ?? false;

  const enable = async () => {
    setFeedback('');
    if (!vapid) return;
    const payload = await subscribeToPush(vapid);
    if (!payload) {
      setFeedback('Permission denied or push unsupported.');
      return;
    }
    try {
      await subscribe.mutateAsync(payload);
      setThisDeviceSubscribed(true);
      setFeedback('Notifications enabled on this device.');
      setTimeout(() => setFeedback(''), 3000);
    } catch (exc) {
      setFeedback(`Subscribe failed: ${exc instanceof Error ? exc.message : exc}`);
    }
  };

  const disable = async () => {
    setFeedback('');
    const sub = await getPushSubscription();
    const endpoint = sub?.endpoint ?? '';
    await unsubscribeFromPush();
    if (endpoint) {
      try {
        await unsubscribe.mutateAsync({ endpoint });
      } catch {/* ignored: client-side unsub already happened */}
    }
    setThisDeviceSubscribed(false);
    setFeedback('Notifications disabled on this device.');
    setTimeout(() => setFeedback(''), 3000);
  };

  const sendTest = async () => {
    setFeedback('');
    try {
      const res = await test.mutateAsync();
      setFeedback(`Test sent → ${res.delivered} device(s).`);
      setTimeout(() => setFeedback(''), 3000);
    } catch (exc) {
      setFeedback(`Test failed: ${exc instanceof Error ? exc.message : exc}`);
    }
  };

  return (
    <div className="bg-[#111820] border border-[#1e2738] rounded-xl p-4 space-y-3">
      <div>
        <h2 className="text-base font-semibold text-[#e2e8f0]">PWA notifications</h2>
        <p className="text-xs text-[#7a8599] mt-1">
          Add the dashboard to your iPhone Home Screen (Safari → Share → Add to Home Screen)
          to install it as an app, then enable Web Push so completed scheduled tasks
          and Signal events ping this device. Independent from Signal — you'll get both.
        </p>
      </div>

      {!supported && (
        <div className="text-xs text-[#fbbf24] bg-[#fbbf24]/10 border border-[#fbbf24]/30 rounded p-2">
          This browser doesn't support Web Push. Install the PWA and try again.
        </div>
      )}
      {supported && !configured && (
        <div className="text-xs text-[#fbbf24] bg-[#fbbf24]/10 border border-[#fbbf24]/30 rounded p-2">
          VAPID keys not configured on the server. Run <code>python -m app.web_push.bootstrap</code> to generate them, then restart the gateway.
        </div>
      )}
      {supported && configured && (
        <div className="flex flex-wrap items-center gap-3">
          {thisDeviceSubscribed ? (
            <button
              onClick={disable}
              disabled={unsubscribe.isPending}
              className="px-3 py-2 bg-[#1e2738] hover:bg-[#2a3548] disabled:opacity-50 rounded text-[#e2e8f0] text-sm"
            >
              Disable on this device
            </button>
          ) : (
            <button
              onClick={enable}
              disabled={subscribe.isPending || !vapid}
              className="px-3 py-2 bg-[#2563eb] hover:bg-[#1d4ed8] disabled:opacity-50 rounded text-white text-sm"
            >
              {subscribe.isPending ? 'Enabling…' : 'Enable on this device'}
            </button>
          )}
          <button
            onClick={sendTest}
            disabled={test.isPending || (subsQ.data?.count ?? 0) === 0}
            className="px-3 py-2 bg-[#0a0f18] border border-[#1e2738] hover:border-[#3b4659] disabled:opacity-50 rounded text-[#7a8599] hover:text-[#e2e8f0] text-sm"
          >
            {test.isPending ? 'Sending…' : 'Send test'}
          </button>
          <span className="text-xs text-[#7a8599]">
            {subsQ.data ? `${subsQ.data.count} device${subsQ.data.count === 1 ? '' : 's'} registered` : ''}
          </span>
        </div>
      )}

      {subsQ.data && subsQ.data.devices.length > 0 && (
        <ul className="text-xs text-[#7a8599] space-y-1 mt-2">
          {subsQ.data.devices.map((d) => (
            <li key={d.added_at + d.user_agent} className="flex items-center justify-between gap-3">
              <span className="truncate flex-1">{d.user_agent || '<no UA>'}</span>
              <span className="text-[#94a3b8]">{d.endpoint_host}</span>
            </li>
          ))}
        </ul>
      )}

      {feedback && <div className="text-xs text-[#34d399]">{feedback}</div>}
    </div>
  );
}

// ── Voice mode ────────────────────────────────────────────────────────────

const VOICE_OPTIONS: Array<{
  value: VoiceMode;
  label: string;
  detail: string;
}> = [
  {
    value: 'off',
    label: 'Off',
    detail: 'No speech-to-text or text-to-speech. Voice notes ignored.',
  },
  {
    value: 'local',
    label: 'Local (whisper.cpp + Piper)',
    detail:
      'Runs on the host. Zero per-message cost, no audio leaves the box. Requires the binaries installed via host_bridge/install_voice.sh.',
  },
  {
    value: 'cloud',
    label: 'Cloud (Groq Whisper + Google Neural2)',
    detail:
      'Faster, multilingual coverage. Needs GROQ_API_KEY and GOOGLE_CLOUD_TTS_KEY in .env. Falls back to local if either key is empty.',
  },
];

function VoiceModeCard({ settings }: { settings: RuntimeSettings }) {
  const update = useUpdateRuntimeSettings();
  const [pending, setPending] = useState<VoiceMode | null>(null);
  const [success, setSuccess] = useState('');

  const choose = async (mode: VoiceMode) => {
    if (mode === settings.voice_mode || update.isPending) return;
    setPending(mode);
    setSuccess('');
    try {
      await update.mutateAsync({ voice_mode: mode });
      setSuccess(`Voice mode → ${mode}.`);
      setTimeout(() => setSuccess(''), 2500);
    } finally {
      setPending(null);
    }
  };

  const error = update.error instanceof Error ? update.error.message : '';

  return (
    <div className="bg-[#111820] border border-[#1e2738] rounded-xl p-4 space-y-3">
      <div>
        <h2 className="text-base font-semibold text-[#e2e8f0]">Voice mode</h2>
        <p className="text-xs text-[#7a8599] mt-1">
          Controls how AndrusAI handles incoming Signal voice notes and whether
          replies are spoken back. Switch any time — the change applies to the
          next inbound message.
        </p>
      </div>

      <div className="space-y-2">
        {VOICE_OPTIONS.map((opt) => {
          const active = settings.voice_mode === opt.value;
          const isPending = pending === opt.value;
          return (
            <button
              key={opt.value}
              onClick={() => choose(opt.value)}
              disabled={update.isPending}
              className={`w-full text-left px-3 py-2.5 rounded-lg border transition-colors ${
                active
                  ? 'bg-[#60a5fa]/10 border-[#60a5fa]/40 text-[#e2e8f0]'
                  : 'bg-[#0a0f18] border-[#1e2738] text-[#7a8599] hover:text-[#e2e8f0] hover:border-[#3b4659]'
              } disabled:opacity-50`}
            >
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">{opt.label}</span>
                {active && (
                  <span className="text-[10px] uppercase tracking-wider text-[#34d399]">
                    Active
                  </span>
                )}
                {isPending && (
                  <span className="text-[10px] uppercase tracking-wider text-[#fbbf24]">
                    Saving…
                  </span>
                )}
              </div>
              <div className="text-xs text-[#7a8599] mt-1">{opt.detail}</div>
            </button>
          );
        })}
      </div>

      {error && <div className="text-[#f87171] text-sm">{error}</div>}
      {success && <div className="text-[#34d399] text-sm">{success}</div>}
    </div>
  );
}

// ── Vision-driven computer use ────────────────────────────────────────────

function VisionComputerUseCard({ settings }: { settings: RuntimeSettings }) {
  const update = useUpdateRuntimeSettings();
  const [enabled, setEnabled] = useState(settings.vision_cu_enabled);
  const [capInput, setCapInput] = useState(
    settings.vision_cu_monthly_cap_usd.toFixed(2),
  );
  const [success, setSuccess] = useState('');

  // Re-sync when server data lands.
  useEffect(() => {
    setEnabled(settings.vision_cu_enabled);
    setCapInput(settings.vision_cu_monthly_cap_usd.toFixed(2));
  }, [settings.vision_cu_enabled, settings.vision_cu_monthly_cap_usd]);

  const save = async () => {
    setSuccess('');
    const cap = parseFloat(capInput);
    if (isNaN(cap) || cap < 0) return;
    await update.mutateAsync({
      vision_cu_enabled: enabled,
      vision_cu_monthly_cap_usd: cap,
    });
    setSuccess('Saved.');
    setTimeout(() => setSuccess(''), 2500);
  };

  const dirty =
    enabled !== settings.vision_cu_enabled ||
    parseFloat(capInput) !== settings.vision_cu_monthly_cap_usd;
  const error = update.error instanceof Error ? update.error.message : '';

  return (
    <div className="bg-[#111820] border border-[#1e2738] rounded-xl p-4 space-y-3">
      <div>
        <h2 className="text-base font-semibold text-[#e2e8f0]">
          Vision-driven computer use
        </h2>
        <p className="text-xs text-[#7a8599] mt-1">
          Last-resort UI control via Anthropic Haiku 4.5 with screenshot input.
          Used only when Playwright + AppleScript can't accomplish the task.
          Hard cap stops new tasks for the rest of the calendar month when
          reached.
        </p>
      </div>

      <label className="flex items-center gap-3 cursor-pointer">
        <input
          type="checkbox"
          checked={enabled}
          onChange={(e) => setEnabled(e.target.checked)}
          className="w-4 h-4 accent-[#60a5fa]"
        />
        <span className="text-sm text-[#e2e8f0]">Enable vision computer use</span>
      </label>

      <label className="block">
        <span className="text-sm text-[#e2e8f0]">Monthly USD cap</span>
        <input
          type="number"
          step="0.50"
          min="0"
          max="1000"
          value={capInput}
          onChange={(e) => setCapInput(e.target.value)}
          className="mt-1 w-full bg-[#0a0f18] border border-[#1e2738] rounded px-3 py-2 text-[#e2e8f0] text-sm"
        />
        <span className="text-xs text-[#7a8599] mt-1 block">
          Resets at the start of each calendar month. Default $10.
        </span>
      </label>

      <div className="flex items-center gap-3">
        <button
          onClick={save}
          disabled={!dirty || update.isPending}
          className="px-4 py-2 bg-[#2563eb] hover:bg-[#1d4ed8] disabled:opacity-50 rounded text-white text-sm"
        >
          {update.isPending ? 'Saving…' : 'Save'}
        </button>
        {error && <span className="text-[#f87171] text-sm">{error}</span>}
        {success && <span className="text-[#34d399] text-sm">{success}</span>}
      </div>
    </div>
  );
}

// ── Concierge persona ─────────────────────────────────────────────────────

function ConciergePersonaCard({ settings }: { settings: RuntimeSettings }) {
  const update = useUpdateRuntimeSettings();
  const [enabled, setEnabled] = useState(settings.concierge_persona_enabled);
  const [success, setSuccess] = useState('');

  useEffect(() => {
    setEnabled(settings.concierge_persona_enabled);
  }, [settings.concierge_persona_enabled]);

  const toggle = async (next: boolean) => {
    if (update.isPending) return;
    setEnabled(next);
    setSuccess('');
    try {
      await update.mutateAsync({ concierge_persona_enabled: next });
      setSuccess(next ? 'Concierge on.' : 'Concierge off.');
      setTimeout(() => setSuccess(''), 2500);
    } catch {
      // surfaced via update.error; revert local state
      setEnabled(!next);
    }
  };

  const error = update.error instanceof Error ? update.error.message : '';

  return (
    <div className="bg-[#111820] border border-[#1e2738] rounded-xl p-4 space-y-3">
      <div>
        <h2 className="text-base font-semibold text-[#e2e8f0]">Concierge persona</h2>
        <p className="text-xs text-[#7a8599] mt-1">
          Wraps Commander's terse routing output in a warmer conversational voice
          for Signal direct messages and the <code className="text-[#60a5fa]">/cp/chat</code>
          {' '}panel. Tool output and <code className="text-[#60a5fa]">/cp/*</code> API
          consumers always see the structured underlying response.
        </p>
      </div>

      <label className="flex items-center gap-3 cursor-pointer">
        <input
          type="checkbox"
          checked={enabled}
          onChange={(e) => toggle(e.target.checked)}
          disabled={update.isPending}
          className="w-4 h-4 accent-[#60a5fa]"
        />
        <span className="text-sm text-[#e2e8f0]">
          Apply concierge persona to chat responses
        </span>
      </label>

      {error && <div className="text-[#f87171] text-sm">{error}</div>}
      {success && <div className="text-[#34d399] text-sm">{success}</div>}
    </div>
  );
}
