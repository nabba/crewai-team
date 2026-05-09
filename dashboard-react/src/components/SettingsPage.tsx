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
  useGovernanceRatchetQuery,
  useSetGovernanceRatchet,
  useRelaxGovernanceRatchet,
  useRunbookSettingsQuery,
  useToggleRunbook,
  useBackgroundTasksQuery,
  useSetBackgroundTasks,
  type RuntimeSettings,
  type VoiceMode,
  type GovernanceRatchetThreshold,
  type RunbookEntry,
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

      <BackgroundTasksCard />
      <VoiceModeCard settings={settingsQ.data} />
      <VisionComputerUseCard settings={settingsQ.data} />
      <ConciergePersonaCard settings={settingsQ.data} />
      <Tier3AmendmentCard settings={settingsQ.data} />
      <GovernanceRatchetCard />
      <GoodhartHardGateCard settings={settingsQ.data} />
      <SelfHealSubsystemsCard settings={settingsQ.data} />
      <WebPushCard />
    </div>
  );
}

// ── Background tasks kill switch ──────────────────────────────────────────

function BackgroundTasksCard() {
  const q = useBackgroundTasksQuery();
  const set = useSetBackgroundTasks();
  const enabled = q.data?.enabled ?? false;
  const loading = q.isLoading;

  const flip = async () => {
    if (loading || set.isPending) return;
    try {
      await set.mutateAsync(!enabled);
    } catch {
      /* error surfaces via set.error */
    }
  };

  const errMsg = (q.error || set.error) instanceof Error
    ? (q.error || set.error)!.toString()
    : '';

  return (
    <div className="bg-[#111820] border border-[#1e2738] rounded-xl p-4 space-y-3">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="text-sm font-medium text-[#e2e8f0]">Background tasks</h2>
          <p className="text-xs text-[#7a8599] mt-1">
            Master kill switch for the idle scheduler — fiction ingest, LLM
            discovery, training collector, atlas competence sync, system
            monitor, version snapshots, and ~30 other periodic jobs.
            Turn off to stop all background spend immediately; user-initiated
            crew runs (Signal, brainstorm, dashboard) keep working either way.
          </p>
        </div>
        <button
          onClick={flip}
          disabled={loading || set.isPending}
          className={
            enabled
              ? 'shrink-0 text-xs px-3 py-1.5 rounded-lg border border-[#34d399]/40 bg-[#34d399]/10 text-[#34d399] hover:bg-[#34d399]/20 disabled:opacity-50'
              : 'shrink-0 text-xs px-3 py-1.5 rounded-lg border border-[#f87171]/40 bg-[#f87171]/10 text-[#f87171] hover:bg-[#f87171]/20 disabled:opacity-50'
          }
          title={enabled ? 'Click to pause all background jobs' : 'Click to resume background jobs'}
        >
          {loading ? '…' : enabled ? 'ON' : 'OFF'}
        </button>
      </div>
      {errMsg && <p className="text-xs text-[#f87171]">{errMsg}</p>}
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

// ── Tier-3 amendment protocol ─────────────────────────────────────────────

function Tier3AmendmentCard({ settings }: { settings: RuntimeSettings }) {
  const update = useUpdateRuntimeSettings();
  const [pendingNext, setPendingNext] = useState<boolean | null>(null);
  const [success, setSuccess] = useState('');

  const enabled = settings.tier3_amendment_enabled;

  const handleToggleClick = (next: boolean) => {
    if (update.isPending) return;
    if (next === enabled) return;
    // Don't apply immediately — open the confirmation modal first.
    setPendingNext(next);
    setSuccess('');
  };

  const cancel = () => setPendingNext(null);

  const confirm = async () => {
    if (pendingNext === null) return;
    try {
      await update.mutateAsync({ tier3_amendment_enabled: pendingNext });
      setSuccess(pendingNext ? 'Tier-3 amendment protocol enabled.' : 'Tier-3 amendment protocol disabled.');
      setTimeout(() => setSuccess(''), 3000);
    } finally {
      setPendingNext(null);
    }
  };

  const error = update.error instanceof Error ? update.error.message : '';

  return (
    <div className="bg-[#111820] border border-[#1e2738] rounded-xl p-4 space-y-3">
      <div>
        <h2 className="text-base font-semibold text-[#e2e8f0]">
          Tier-3 amendment protocol
        </h2>
        <p className="text-xs text-[#7a8599] mt-1">
          Lets agents propose modifications to <code className="text-[#60a5fa]">TIER_IMMUTABLE</code>{' '}
          files (e.g. evolution-engine internals, prompt registry) after demonstrating
          a clean track record. Every proposal still requires your manual approval at
          the <em>operator-approve</em> step. Safety-core files
          (<code className="text-[#60a5fa]">governance.py</code>,{' '}
          <code className="text-[#60a5fa]">safety_guardian.py</code>,
          eval/sandbox infrastructure, and the protocol's own files) are
          self-quarantined and CANNOT be amended via this protocol —
          ever — only by direct human PR. See{' '}
          <code className="text-[#60a5fa]">docs/TIER3_AMENDMENT.md</code>.
        </p>
      </div>

      <label className="flex items-center gap-3 cursor-pointer">
        <input
          type="checkbox"
          checked={enabled}
          // The checkbox stays *visually* in lock-step with the persisted
          // ``enabled`` prop — the user's intended toggle is captured via
          // ``onClick`` + ``preventDefault`` so the DOM doesn't drift while
          // the confirmation modal is open. Without preventDefault the
          // browser flips the checkbox immediately and the user thinks
          // the toggle "took effect" before they confirmed.
          onChange={() => { /* controlled — handled by onClick */ }}
          onClick={(e) => {
            e.preventDefault();
            if (update.isPending) return;
            handleToggleClick(!enabled);
          }}
          disabled={update.isPending}
          className="w-4 h-4 accent-[#60a5fa]"
        />
        <span className="text-sm text-[#e2e8f0]">
          Enable Tier-3 amendment protocol
        </span>
        <span className="text-[10px] uppercase tracking-wider text-[#7a8599] ml-auto">
          {enabled ? (
            <span className="text-[#34d399]">ACTIVE</span>
          ) : (
            <span className="text-[#7a8599]">OFF</span>
          )}
        </span>
      </label>

      {error && <div className="text-[#f87171] text-sm">{error}</div>}
      {success && <div className="text-[#34d399] text-sm">{success}</div>}

      {pendingNext !== null && (
        <Tier3ConfirmModal
          enabling={pendingNext}
          pending={update.isPending}
          onConfirm={confirm}
          onCancel={cancel}
        />
      )}
    </div>
  );
}

function Tier3ConfirmModal({
  enabling,
  pending,
  onConfirm,
  onCancel,
}: {
  enabling: boolean;
  pending: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
      onClick={onCancel}
    >
      <div
        className="bg-[#111820] border border-[#1e2738] rounded-xl p-6 max-w-lg w-[90vw] space-y-4 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div>
          <h3 className="text-lg font-semibold text-[#e2e8f0]">
            {enabling ? 'Enable Tier-3 amendment protocol?' : 'Disable Tier-3 amendment protocol?'}
          </h3>
          <p className="text-sm text-[#7a8599] mt-1">Are you sure?</p>
        </div>

        {enabling ? (
          <div className="text-sm text-[#e2e8f0] space-y-2">
            <p>
              Enabling this lets agents propose modifications to{' '}
              <code className="text-[#60a5fa]">TIER_IMMUTABLE</code> files. Each
              proposal still goes through:
            </p>
            <ol className="list-decimal list-inside text-xs text-[#94a3b8] space-y-1 ml-2">
              <li>Programmatic eligibility (≥200 promotions/90d, &lt;5% rollback rate, no active alignment warnings)</li>
              <li>7-day cool-down window watching for any rollback signal</li>
              <li>Your manual approval — they can't apply without your 👍</li>
              <li>30-day post-apply monitoring with auto-rollback on regression</li>
            </ol>
            <p className="text-xs text-[#fbbf24]">
              Safety-core files (governance.py, safety_guardian.py, eval/sandbox,
              and ~30 others) are self-quarantined and unaffected by this toggle.
            </p>
          </div>
        ) : (
          <div className="text-sm text-[#e2e8f0] space-y-2">
            <p>
              Disabling stops agents from filing new amendment proposals immediately.
            </p>
            <ul className="list-disc list-inside text-xs text-[#94a3b8] space-y-1 ml-2">
              <li>Pending proposals already in the pipeline are unaffected — they continue through their state machine.</li>
              <li>Re-enabling later resumes acceptance of new proposals; existing audit trail is preserved.</li>
              <li>Safe to flip on/off as needed.</li>
            </ul>
          </div>
        )}

        <div className="flex items-center justify-end gap-3 pt-2">
          <button
            onClick={onCancel}
            disabled={pending}
            className="px-4 py-2 bg-[#0a0f18] border border-[#1e2738] hover:border-[#3b4659] disabled:opacity-50 rounded text-[#7a8599] hover:text-[#e2e8f0] text-sm"
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            disabled={pending}
            className={`px-4 py-2 rounded text-white text-sm disabled:opacity-50 ${
              enabling
                ? 'bg-[#dc2626] hover:bg-[#b91c1c]'
                : 'bg-[#2563eb] hover:bg-[#1d4ed8]'
            }`}
          >
            {pending
              ? 'Saving…'
              : enabling
                ? 'Yes, enable'
                : 'Yes, disable'}
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Governance ratchet (Wave 3 #6) ────────────────────────────────────────

function GovernanceRatchetCard() {
  const stateQ = useGovernanceRatchetQuery();
  const [activeRelaxName, setActiveRelaxName] = useState<string | null>(null);
  const [activeRatchetName, setActiveRatchetName] = useState<string | null>(null);

  if (stateQ.isLoading) {
    return (
      <div className="bg-[#111820] border border-[#1e2738] rounded-xl p-4">
        <h2 className="text-base font-semibold text-[#e2e8f0]">Governance ratchet</h2>
        <div className="text-[#7a8599] text-sm mt-2">Loading thresholds…</div>
      </div>
    );
  }
  if (stateQ.error || !stateQ.data) {
    return (
      <div className="bg-[#111820] border border-[#1e2738] rounded-xl p-4">
        <h2 className="text-base font-semibold text-[#e2e8f0]">Governance ratchet</h2>
        <div className="text-[#f87171] text-sm mt-2">
          Could not load ratchet state: {stateQ.error instanceof Error ? stateQ.error.message : 'unknown error'}
        </div>
      </div>
    );
  }

  const thresholds = stateQ.data.thresholds || [];

  return (
    <div className="bg-[#111820] border border-[#1e2738] rounded-xl p-4 space-y-4">
      <div>
        <h2 className="text-base font-semibold text-[#e2e8f0]">Governance ratchet</h2>
        <p className="text-xs text-[#7a8599] mt-1">
          Operator-controlled raising / relaxing of the promotion-gate
          floors in <code className="text-[#60a5fa]">app/governance.py</code>.
          Ratcheting <em>up</em> tightens the bar; relaxing <em>down</em> loosens
          it (but never below the hardcoded <code className="text-[#60a5fa]">FLOOR</code>).
          Both actions are audited as{' '}
          <code className="text-[#60a5fa]">actor=governance_ratchet</code>{' '}
          in the global Postgres audit log.
        </p>
      </div>

      <div className="space-y-3">
        {thresholds.map((t) => (
          <ThresholdRow
            key={t.name}
            threshold={t}
            onRatchetClick={() => setActiveRatchetName(t.name)}
            onRelaxClick={() => setActiveRelaxName(t.name)}
          />
        ))}
      </div>

      {activeRatchetName && (
        <RatchetUpModal
          threshold={thresholds.find((t) => t.name === activeRatchetName)!}
          onClose={() => setActiveRatchetName(null)}
        />
      )}
      {activeRelaxName && (
        <RatchetRelaxModal
          threshold={thresholds.find((t) => t.name === activeRelaxName)!}
          onClose={() => setActiveRelaxName(null)}
        />
      )}
    </div>
  );
}

function ThresholdRow({
  threshold,
  onRatchetClick,
  onRelaxClick,
}: {
  threshold: GovernanceRatchetThreshold;
  onRatchetClick: () => void;
  onRelaxClick: () => void;
}) {
  const { name, floor, current, effective, history } = threshold;
  const lastEntry = history[history.length - 1];
  const aboveFloor = current > floor;
  return (
    <div className="bg-[#0a0f18] border border-[#1e2738] rounded-lg p-3 space-y-2">
      <div className="flex items-center justify-between">
        <div>
          <div className="text-sm font-medium text-[#e2e8f0]">
            {name === 'safety_minimum' ? 'Safety minimum' : 'Quality minimum'}
          </div>
          <div className="text-xs text-[#7a8599] mt-1">
            Floor: <code className="text-[#94a3b8]">{floor.toFixed(3)}</code>{' '}
            · Current: <code className="text-[#94a3b8]">{current.toFixed(3)}</code>
            {aboveFloor && (
              <span className="text-[#34d399] ml-2">
                ↑ {((current - floor) * 100).toFixed(1)}% above floor
              </span>
            )}
          </div>
          <div className="text-xs text-[#7a8599]">
            Effective threshold the gates enforce:{' '}
            <code className="text-[#60a5fa]">{effective.toFixed(3)}</code>
          </div>
        </div>
        <div className="flex flex-col gap-2 ml-4 shrink-0">
          <button
            onClick={onRatchetClick}
            className="px-3 py-1.5 bg-[#2563eb] hover:bg-[#1d4ed8] rounded text-white text-xs"
          >
            Ratchet up ↑
          </button>
          <button
            onClick={onRelaxClick}
            disabled={!aboveFloor}
            className="px-3 py-1.5 bg-[#0a0f18] border border-[#dc2626]/40 hover:border-[#dc2626] disabled:opacity-30 disabled:cursor-not-allowed rounded text-[#f87171] text-xs"
          >
            Relax ↓
          </button>
        </div>
      </div>
      {lastEntry && (
        <div className="text-[10px] text-[#64748b] pt-2 border-t border-[#1e2738]">
          Last change: {lastEntry.direction === 'up' ? '↑' : lastEntry.direction === 'down' ? '↓' : '◇'}{' '}
          {lastEntry.old_value.toFixed(3)} → {lastEntry.new_value.toFixed(3)} by{' '}
          <code>{lastEntry.source}</code> on {lastEntry.ts.slice(0, 10)}
          {lastEntry.reason && (
            <span className="ml-1 text-[#94a3b8]">— {lastEntry.reason.slice(0, 60)}</span>
          )}
        </div>
      )}
    </div>
  );
}

function RatchetUpModal({
  threshold,
  onClose,
}: {
  threshold: GovernanceRatchetThreshold;
  onClose: () => void;
}) {
  const setRatchet = useSetGovernanceRatchet();
  const [newValue, setNewValue] = useState(
    Math.min(threshold.current + 0.01, 1.0).toFixed(3),
  );
  const [reason, setReason] = useState('');
  const [error, setError] = useState('');

  const proposed = parseFloat(newValue);
  const valid =
    !isNaN(proposed) && proposed > threshold.current && proposed <= 1.0;

  const submit = async () => {
    if (!valid) return;
    setError('');
    try {
      await setRatchet.mutateAsync({
        name: threshold.name,
        new_value: proposed,
        reason: reason.trim(),
      });
      onClose();
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : 'Unknown error');
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="bg-[#111820] border border-[#1e2738] rounded-xl p-6 max-w-lg w-[90vw] space-y-4 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div>
          <h3 className="text-lg font-semibold text-[#e2e8f0]">
            Ratchet up {threshold.name === 'safety_minimum' ? 'safety' : 'quality'} minimum
          </h3>
          <p className="text-sm text-[#7a8599] mt-1">
            Tighten the floor. Subsequent promotions must clear this stricter bar.
            Monotonic up — must be greater than current.
          </p>
        </div>

        <div className="text-xs text-[#94a3b8] space-y-1">
          <div>Current: <code className="text-[#e2e8f0]">{threshold.current.toFixed(3)}</code></div>
          <div>Floor (post-bootstrap contract): <code className="text-[#e2e8f0]">{threshold.floor.toFixed(3)}</code></div>
        </div>

        <label className="block">
          <span className="text-sm text-[#e2e8f0]">New value (0–1)</span>
          <input
            type="number"
            step="0.001"
            min={threshold.current}
            max="1.0"
            value={newValue}
            onChange={(e) => setNewValue(e.target.value)}
            className="mt-1 w-full bg-[#0a0f18] border border-[#1e2738] rounded px-3 py-2 text-[#e2e8f0] text-sm font-mono"
          />
          {!valid && proposed <= threshold.current && (
            <span className="text-[10px] text-[#f87171]">
              Must be greater than current ({threshold.current.toFixed(3)})
            </span>
          )}
        </label>

        <label className="block">
          <span className="text-sm text-[#e2e8f0]">Reason (optional but recommended)</span>
          <textarea
            value={reason}
            onChange={(e) => setReason(e.target.value)}
            rows={2}
            placeholder="e.g. last 50 promotions all scored ≥ 0.97 safety; raising bar from 0.95 to 0.96"
            className="mt-1 w-full bg-[#0a0f18] border border-[#1e2738] rounded px-3 py-2 text-[#e2e8f0] text-sm"
          />
        </label>

        {error && <div className="text-sm text-[#f87171]">{error}</div>}

        <div className="flex items-center justify-end gap-3 pt-2">
          <button
            onClick={onClose}
            disabled={setRatchet.isPending}
            className="px-4 py-2 bg-[#0a0f18] border border-[#1e2738] hover:border-[#3b4659] disabled:opacity-50 rounded text-[#7a8599] hover:text-[#e2e8f0] text-sm"
          >
            Cancel
          </button>
          <button
            onClick={submit}
            disabled={!valid || setRatchet.isPending}
            className="px-4 py-2 bg-[#2563eb] hover:bg-[#1d4ed8] disabled:opacity-50 rounded text-white text-sm"
          >
            {setRatchet.isPending ? 'Saving…' : `Ratchet up to ${proposed.toFixed(3)}`}
          </button>
        </div>
      </div>
    </div>
  );
}

function RatchetRelaxModal({
  threshold,
  onClose,
}: {
  threshold: GovernanceRatchetThreshold;
  onClose: () => void;
}) {
  const relax = useRelaxGovernanceRatchet();
  const [newValue, setNewValue] = useState(
    Math.max(threshold.current - 0.01, threshold.floor).toFixed(3),
  );
  const [reason, setReason] = useState('');
  const [confirmation, setConfirmation] = useState('');
  const [error, setError] = useState('');

  const expectedPhrase = `RELAX ${threshold.name.toUpperCase()}`;
  const proposed = parseFloat(newValue);
  const valid =
    !isNaN(proposed) &&
    proposed < threshold.current &&
    proposed >= threshold.floor;
  const phraseMatches = confirmation.trim().toUpperCase() === expectedPhrase;
  const reasonOk = reason.trim().length >= 10;

  const canSubmit = valid && phraseMatches && reasonOk;

  const submit = async () => {
    if (!canSubmit) return;
    setError('');
    try {
      await relax.mutateAsync({
        name: threshold.name,
        new_value: proposed,
        reason: reason.trim(),
        confirmation: expectedPhrase,
      });
      onClose();
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : 'Unknown error');
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="bg-[#111820] border border-[#dc2626]/40 rounded-xl p-6 max-w-lg w-[90vw] space-y-4 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div>
          <h3 className="text-lg font-semibold text-[#f87171]">
            ⚠️ Relax {threshold.name === 'safety_minimum' ? 'safety' : 'quality'} minimum
          </h3>
          <p className="text-sm text-[#fbbf24] mt-1">
            This <em>lowers</em> the gate that all promotions must clear.
            Future agent self-improvements pass at a looser bar. Cannot
            drop below the hardcoded FLOOR (post-bootstrap safety contract).
          </p>
        </div>

        <div className="text-xs text-[#94a3b8] space-y-1 bg-[#0a0f18] p-2 rounded">
          <div>Current: <code className="text-[#e2e8f0]">{threshold.current.toFixed(3)}</code></div>
          <div>FLOOR (cannot go below): <code className="text-[#fbbf24]">{threshold.floor.toFixed(3)}</code></div>
        </div>

        <label className="block">
          <span className="text-sm text-[#e2e8f0]">New value (must be ≥ {threshold.floor.toFixed(3)} and &lt; {threshold.current.toFixed(3)})</span>
          <input
            type="number"
            step="0.001"
            min={threshold.floor}
            max={threshold.current}
            value={newValue}
            onChange={(e) => setNewValue(e.target.value)}
            className="mt-1 w-full bg-[#0a0f18] border border-[#dc2626]/40 rounded px-3 py-2 text-[#e2e8f0] text-sm font-mono"
          />
          {!isNaN(proposed) && proposed < threshold.floor && (
            <span className="text-[10px] text-[#f87171]">Below FLOOR — refused.</span>
          )}
          {!isNaN(proposed) && proposed >= threshold.current && (
            <span className="text-[10px] text-[#f87171]">Must be lower than current.</span>
          )}
        </label>

        <label className="block">
          <span className="text-sm text-[#e2e8f0]">Reason (≥ 10 chars, mandatory)</span>
          <textarea
            value={reason}
            onChange={(e) => setReason(e.target.value)}
            rows={2}
            placeholder="why are we relaxing the gate?"
            className="mt-1 w-full bg-[#0a0f18] border border-[#dc2626]/40 rounded px-3 py-2 text-[#e2e8f0] text-sm"
          />
          {!reasonOk && reason.length > 0 && (
            <span className="text-[10px] text-[#f87171]">At least 10 chars.</span>
          )}
        </label>

        <label className="block">
          <span className="text-sm text-[#e2e8f0]">
            Type <code className="text-[#fbbf24]">{expectedPhrase}</code> to confirm:
          </span>
          <input
            type="text"
            value={confirmation}
            onChange={(e) => setConfirmation(e.target.value)}
            placeholder={expectedPhrase}
            className="mt-1 w-full bg-[#0a0f18] border border-[#dc2626]/40 rounded px-3 py-2 text-[#e2e8f0] text-sm font-mono"
          />
          {confirmation && !phraseMatches && (
            <span className="text-[10px] text-[#f87171]">
              Phrase doesn't match.
            </span>
          )}
        </label>

        {error && <div className="text-sm text-[#f87171]">{error}</div>}

        <div className="flex items-center justify-end gap-3 pt-2">
          <button
            onClick={onClose}
            disabled={relax.isPending}
            className="px-4 py-2 bg-[#0a0f18] border border-[#1e2738] hover:border-[#3b4659] disabled:opacity-50 rounded text-[#7a8599] hover:text-[#e2e8f0] text-sm"
          >
            Cancel
          </button>
          <button
            onClick={submit}
            disabled={!canSubmit || relax.isPending}
            className="px-4 py-2 bg-[#dc2626] hover:bg-[#b91c1c] disabled:opacity-30 rounded text-white text-sm"
          >
            {relax.isPending ? 'Saving…' : `Relax to ${proposed.toFixed(3)}`}
          </button>
        </div>
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

// ── Goodhart hard gate (Wave 3 #2 + Wave 4 React UI) ──────────────────────

type GoodhartMode = 'off' | 'advisory' | 'enforcing';

function modeFor(settings: RuntimeSettings): GoodhartMode {
  if (settings.goodhart_hard_gate_disabled) return 'off';
  return settings.goodhart_hard_gate_enforcing ? 'enforcing' : 'advisory';
}

const GOODHART_OPTIONS: Array<{
  value: GoodhartMode;
  label: string;
  detail: string;
  badgeColor: string;
}> = [
  {
    value: 'off',
    label: 'Off (emergency disable)',
    detail:
      "Skip the Goodhart gate entirely. Use only when a buggy detector is blocking legitimate promotions. Promotions are evaluated on safety + quality only — no gaming-signal check.",
    badgeColor: '#f87171',
  },
  {
    value: 'advisory',
    label: 'Advisory (default)',
    detail:
      "Detector runs and records the recent severity in every promotion's audit trail (gate_results.goodhart). Does NOT block. Good for the first ~2 weeks after enabling — lets you characterise the false-positive rate before flipping to enforcing.",
    badgeColor: '#fbbf24',
  },
  {
    value: 'enforcing',
    label: 'Enforcing (blocking)',
    detail:
      "Severity='high' BLOCKS promotion (severity='medium' / 'low' do not). The gate runs BEFORE safety + quality checks, so a high-severity gaming signal pre-empts everything else. Reason in PromotionResult: \"Goodhart hard gate blocked\".",
    badgeColor: '#34d399',
  },
];

function GoodhartHardGateCard({ settings }: { settings: RuntimeSettings }) {
  const update = useUpdateRuntimeSettings();
  const [success, setSuccess] = useState('');

  const currentMode = modeFor(settings);

  const choose = async (next: GoodhartMode) => {
    if (next === currentMode || update.isPending) return;
    setSuccess('');
    const body =
      next === 'off'
        ? { goodhart_hard_gate_disabled: true, goodhart_hard_gate_enforcing: false }
        : next === 'advisory'
          ? { goodhart_hard_gate_disabled: false, goodhart_hard_gate_enforcing: false }
          : { goodhart_hard_gate_disabled: false, goodhart_hard_gate_enforcing: true };
    try {
      await update.mutateAsync(body);
      setSuccess(`Goodhart gate → ${next}.`);
      setTimeout(() => setSuccess(''), 2500);
    } catch {
      // surface via update.error
    }
  };

  const error = update.error instanceof Error ? update.error.message : '';

  return (
    <div className="bg-[#111820] border border-[#1e2738] rounded-xl p-4 space-y-3">
      <div>
        <h2 className="text-base font-semibold text-[#e2e8f0]">
          Goodhart hard gate
        </h2>
        <p className="text-xs text-[#7a8599] mt-1">
          Reads the goodhart_guard.py severity signal (kept-ratio spikes,
          category concentration, rollback silence) before each promotion
          and (when enforcing) blocks promotions while gaming is detected.
          The gate runs as <em>Gate 0</em> — before safety and quality —
          so a high-severity block pre-empts everything else. See{' '}
          <code className="text-[#60a5fa]">app/goodhart_guard.py</code>{' '}
          for the detection logic.
        </p>
      </div>

      <div className="space-y-2">
        {GOODHART_OPTIONS.map((opt) => {
          const active = currentMode === opt.value;
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
                  <span
                    className="text-[10px] uppercase tracking-wider"
                    style={{ color: opt.badgeColor }}
                  >
                    Active
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

// ── Self-heal subsystems (Wave 4 React UI) ────────────────────────────────

interface SubsystemSwitch {
  key:
    | 'error_runbooks_enabled'
    | 'tool_supervisor_enabled'
    | 'recovery_loop_enabled';
  label: string;
  detail: string;
}

const SELF_HEAL_SWITCHES: SubsystemSwitch[] = [
  {
    key: 'error_runbooks_enabled',
    label: 'Runbook dispatcher',
    detail:
      "Pattern-matched error runbooks. When ON, errors_monitor.py anomalies fire registered handlers in app/healing/handlers/ — db_pool_reset, schema-drift CRs, code-drift alerts, etc. 7 safety gates apply (severity / pattern / per-runbook enabled / recurrence threshold / success-rate / concurrency cap / no TIER_IMMUTABLE writes).",
  },
  {
    key: 'tool_supervisor_enabled',
    label: 'Tool exception supervisor',
    detail:
      "Mid-iteration tool exception classifier (rate_limit / auth / network / timeout / schema / unknown). When ON, transient tool failures retry with exp backoff, schema mismatches substitute via the tool registry, and unknown errors soft-fail with diagnostic text. Composes with — does not replace — the recovery loop.",
  },
  {
    key: 'recovery_loop_enabled',
    label: 'Refusal recovery loop',
    detail:
      "Detects refusal-shaped final answers (\"I can't…\" / \"I don't have access…\") and retries via 6 ranked strategies: direct_tool, sandbox_execute, re_route, skill_chain, escalate_tier, forge_queue. Off-by-default because each retry costs an extra LLM call.",
  },
];

function SelfHealSubsystemsCard({ settings }: { settings: RuntimeSettings }) {
  const update = useUpdateRuntimeSettings();
  const [success, setSuccess] = useState('');
  const runbookSettingsQ = useRunbookSettingsQuery();
  const toggleRunbook = useToggleRunbook();

  const toggle = async (key: SubsystemSwitch['key'], next: boolean) => {
    if (update.isPending) return;
    setSuccess('');
    try {
      await update.mutateAsync({ [key]: next } as Partial<RuntimeSettings>);
      setSuccess(`${key.replace(/_/g, ' ')} → ${next ? 'on' : 'off'}.`);
      setTimeout(() => setSuccess(''), 2500);
    } catch {
      // surface via update.error
    }
  };

  const error = update.error instanceof Error ? update.error.message : '';
  const runbooks = runbookSettingsQ.data?.runbooks ?? {};
  const runbookEntries = Object.entries(runbooks).sort((a, b) =>
    a[0].localeCompare(b[0]),
  );

  return (
    <div className="bg-[#111820] border border-[#1e2738] rounded-xl p-4 space-y-4">
      <div>
        <h2 className="text-base font-semibold text-[#e2e8f0]">
          Self-heal subsystems
        </h2>
        <p className="text-xs text-[#7a8599] mt-1">
          Three master switches for the healing infrastructure. Each
          composes with the others — a full healing pipeline has all
          three on. Per-runbook overrides for the dispatcher are below
          (only relevant when <em>Runbook dispatcher</em> is on).
        </p>
      </div>

      <div className="space-y-2">
        {SELF_HEAL_SWITCHES.map((sw) => {
          const enabled = Boolean(settings[sw.key]);
          return (
            <label
              key={sw.key}
              className={`flex items-start gap-3 cursor-pointer px-3 py-2.5 rounded-lg border ${
                enabled
                  ? 'bg-[#34d399]/5 border-[#34d399]/30'
                  : 'bg-[#0a0f18] border-[#1e2738]'
              }`}
            >
              <input
                type="checkbox"
                checked={enabled}
                onChange={(e) => toggle(sw.key, e.target.checked)}
                disabled={update.isPending}
                className="w-4 h-4 mt-1 accent-[#60a5fa]"
              />
              <div className="flex-1">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-[#e2e8f0]">
                    {sw.label}
                  </span>
                  <span className="text-[10px] uppercase tracking-wider">
                    {enabled ? (
                      <span className="text-[#34d399]">ON</span>
                    ) : (
                      <span className="text-[#7a8599]">OFF</span>
                    )}
                  </span>
                </div>
                <div className="text-xs text-[#7a8599] mt-1">{sw.detail}</div>
              </div>
            </label>
          );
        })}
      </div>

      {error && <div className="text-[#f87171] text-sm">{error}</div>}
      {success && <div className="text-[#34d399] text-sm">{success}</div>}

      <div className="pt-3 border-t border-[#1e2738]">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-medium text-[#e2e8f0]">
            Registered runbooks ({runbookEntries.length})
          </h3>
          <span className="text-[10px] text-[#7a8599]">
            {settings.error_runbooks_enabled
              ? 'Active when dispatcher is ON'
              : 'Dormant — dispatcher is OFF'}
          </span>
        </div>
        <p className="text-[11px] text-[#7a8599] mt-1">
          Each runbook has its own <code>enabled</code> flag and{' '}
          <code>min_recurrence</code> (the dispatcher requires this many
          occurrences in 24 h before firing). Toggle individual runbooks
          when you want fine-grained control without disabling the whole
          dispatcher.
        </p>

        {runbookSettingsQ.isLoading && (
          <div className="text-xs text-[#7a8599] mt-2">Loading runbook list…</div>
        )}
        {runbookSettingsQ.error && (
          <div className="text-xs text-[#f87171] mt-2">
            Could not load runbook list:{' '}
            {runbookSettingsQ.error instanceof Error
              ? runbookSettingsQ.error.message
              : 'unknown'}
          </div>
        )}

        <ul className="space-y-1.5 mt-3">
          {runbookEntries.map(([name, entry]) => (
            <RunbookRow
              key={name}
              name={name}
              entry={entry}
              dispatcherOn={settings.error_runbooks_enabled}
              onToggle={(next) =>
                toggleRunbook.mutateAsync({ name, enabled: next })
              }
              pending={toggleRunbook.isPending}
            />
          ))}
        </ul>
      </div>
    </div>
  );
}

function RunbookRow({
  name,
  entry,
  dispatcherOn,
  onToggle,
  pending,
}: {
  name: string;
  entry: RunbookEntry;
  dispatcherOn: boolean;
  onToggle: (next: boolean) => Promise<unknown>;
  pending: boolean;
}) {
  const [enabled, setEnabled] = useState(entry.enabled);
  useEffect(() => {
    setEnabled(entry.enabled);
  }, [entry.enabled]);

  const handle = async (next: boolean) => {
    setEnabled(next);
    try {
      await onToggle(next);
    } catch {
      setEnabled(!next);
    }
  };

  return (
    <li className="flex items-center gap-3 text-xs px-2 py-1.5 rounded bg-[#0a0f18] border border-[#1e2738]">
      <input
        type="checkbox"
        checked={enabled}
        onChange={(e) => handle(e.target.checked)}
        disabled={pending || !dispatcherOn}
        className="w-3.5 h-3.5 accent-[#60a5fa]"
      />
      <span
        className={`flex-1 font-mono ${
          enabled && dispatcherOn ? 'text-[#e2e8f0]' : 'text-[#7a8599]'
        }`}
      >
        {name}
      </span>
      <span className="text-[10px] text-[#94a3b8]">
        min_recurrence: <code>{entry.min_recurrence}</code>
      </span>
      {entry._comment && (
        <span
          title={entry._comment}
          className="text-[10px] text-[#7a8599] cursor-help"
        >
          ⓘ
        </span>
      )}
    </li>
  );
}
