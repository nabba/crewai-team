// WP D Phase 5b — React wizard for the cloud migration backend.
// Six steps: Account → Preflight → Configure → Confirm → Execute → Outcome.
// Side panel always shows recent runs (clickable to inspect).
//
// Backend lives at /api/cp/migrate/* — see app/control_plane/migrate_api.py.
// Safety gates are enforced API-side too, but we refuse first for fast
// feedback (typed phrase + budget cap ≤ $500).

import { useEffect, useMemo, useState } from 'react';
import { Skeleton } from './ui/Skeleton';
import { ErrorPanel } from './ui/ErrorPanel';
import {
  useMigrateAccountsQuery,
  useMigrateCancelMutation,
  useMigrateCostMutation,
  useMigratePreflightQuery,
  useHardeningPreviewQuery,
  useBootstrapProjectMutation,
  useMigrateRunQuery,
  useMigrateRunsQuery,
  useMigrateStartMutation,
} from '../api/migrate';
import {
  useRuntimeSettingsQuery,
  useUpdateRuntimeSettings,
} from '../api/queries';
import type {
  CloudAccount,
  CostResponse,
  PreflightProbe,
  ProbeStatus,
  RunRecord,
  RunStatus,
  Tier,
} from '../types/migrate';
import { ApiError } from '../api/client';

// ── Constants ───────────────────────────────────────────────────────

const TYPED_PHRASE = 'MIGRATE TO GCP';
const DEFAULT_REGION = 'europe-north1';
const DEFAULT_BUDGET_USD = 300;
const MAX_BUDGET_USD = 500;

const STEPS = [
  { id: 'account', label: 'Account' },
  { id: 'preflight', label: 'Preflight' },
  { id: 'configure', label: 'Configure' },
  { id: 'confirm', label: 'Confirm' },
  { id: 'execute', label: 'Execute' },
  { id: 'outcome', label: 'Outcome' },
] as const;

type StepId = (typeof STEPS)[number]['id'];

const TERMINAL_STATUSES = new Set<RunStatus>([
  'succeeded',
  'failed',
  'preflight_failed',
  'cancelled',
]);

// ── Style maps ──────────────────────────────────────────────────────

const PROBE_STYLE: Record<
  ProbeStatus,
  { glyph: string; fg: string; label: string }
> = {
  OK:      { glyph: '✓', fg: 'text-[#34d399]', label: 'OK' },
  STALE:   { glyph: '◷', fg: 'text-[#fbbf24]', label: 'STALE' },
  MISSING: { glyph: '✕', fg: 'text-[#f87171]', label: 'MISSING' },
  UNKNOWN: { glyph: '?', fg: 'text-[#7a8599]', label: 'UNKNOWN' },
};

const STATUS_BADGE: Record<
  RunStatus,
  { bg: string; fg: string; border: string; label: string }
> = {
  queued: {
    bg: 'bg-[#7a8599]/15',
    fg: 'text-[#7a8599]',
    border: 'border-[#7a8599]/30',
    label: 'QUEUED',
  },
  preparing: {
    bg: 'bg-[#60a5fa]/15',
    fg: 'text-[#60a5fa]',
    border: 'border-[#60a5fa]/30',
    label: 'PREPARING',
  },
  running: {
    bg: 'bg-[#60a5fa]/15',
    fg: 'text-[#60a5fa]',
    border: 'border-[#60a5fa]/30',
    label: 'RUNNING',
  },
  succeeded: {
    bg: 'bg-[#34d399]/15',
    fg: 'text-[#34d399]',
    border: 'border-[#34d399]/30',
    label: 'SUCCEEDED',
  },
  failed: {
    bg: 'bg-[#f87171]/15',
    fg: 'text-[#f87171]',
    border: 'border-[#f87171]/30',
    label: 'FAILED',
  },
  preflight_failed: {
    bg: 'bg-[#fbbf24]/15',
    fg: 'text-[#fbbf24]',
    border: 'border-[#fbbf24]/30',
    label: 'PREFLIGHT FAILED',
  },
  cancelled: {
    bg: 'bg-[#7a8599]/15',
    fg: 'text-[#7a8599]',
    border: 'border-[#7a8599]/30',
    label: 'CANCELLED',
  },
};

function StatusBadge({ status }: { status: RunStatus }) {
  const s = STATUS_BADGE[status];
  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-medium border ${s.bg} ${s.fg} ${s.border}`}
    >
      {s.label}
    </span>
  );
}

function StepIndicator({
  current,
  reached,
}: {
  current: StepId;
  reached: Set<StepId>;
}) {
  const currentIdx = STEPS.findIndex((s) => s.id === current);
  return (
    <div className="flex items-center gap-1 flex-wrap">
      {STEPS.map((step, i) => {
        const isCurrent = step.id === current;
        const isReached = reached.has(step.id);
        const isPast = i < currentIdx;
        const cls = isCurrent
          ? 'bg-[#60a5fa]/15 text-[#60a5fa] border-[#60a5fa]/40'
          : isReached || isPast
          ? 'bg-[#34d399]/10 text-[#34d399] border-[#34d399]/30'
          : 'bg-[#1e2738] text-[#7a8599] border-[#1e2738]';
        return (
          <span
            key={step.id}
            className={`inline-flex items-center gap-1.5 px-2 py-1 rounded-md text-[11px] font-medium border ${cls}`}
          >
            <span className="font-mono">{i + 1}</span>
            <span>{step.label}</span>
          </span>
        );
      })}
    </div>
  );
}

// ── Step 1: Account picker ──────────────────────────────────────────

function AccountStep({
  selected,
  onSelect,
  onNext,
}: {
  selected: string | null;
  onSelect: (account: string, type: CloudAccount['type']) => void;
  onNext: () => void;
}) {
  const q = useMigrateAccountsQuery();
  const selectedRow = q.data?.accounts.find((a) => a.account === selected);
  const isSvcAcct = selectedRow?.type === 'service_account';

  return (
    <section className="space-y-4">
      <header>
        <h2 className="text-base font-semibold text-[#e2e8f0]">
          1. Pick gcloud account
        </h2>
        <p className="text-xs text-[#7a8599] mt-1">
          The migration will provision resources as this account. User accounts
          usually have project-Owner; service accounts often don't.
        </p>
      </header>

      {q.isLoading && (
        <div className="space-y-2">
          <Skeleton className="h-12" />
          <Skeleton className="h-12" />
        </div>
      )}
      {q.error && <ErrorPanel error={q.error} onRetry={() => q.refetch()} />}

      {q.data && q.data.accounts.length === 0 && (
        <div className="rounded-md border border-[#fbbf24]/30 bg-[#fbbf24]/10 p-3 text-xs text-[#fbbf24]">
          No gcloud accounts found. Run{' '}
          <code className="font-mono">gcloud auth login</code> on the gateway
          host first.
        </div>
      )}

      <div className="space-y-2">
        {q.data?.accounts.map((acct) => {
          const isSelected = acct.account === selected;
          return (
            <button
              key={acct.account}
              onClick={() => onSelect(acct.account, acct.type)}
              className={`w-full text-left p-3 rounded-lg border transition-colors ${
                isSelected
                  ? 'border-[#60a5fa]/40 bg-[#60a5fa]/5'
                  : 'border-[#1e2738] bg-[#111820] hover:border-[#60a5fa]/30 hover:bg-[#1e2738]/50'
              }`}
            >
              <div className="flex items-center justify-between gap-3">
                <div className="min-w-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    <code className="text-sm font-mono text-[#e2e8f0] truncate">
                      {acct.account}
                    </code>
                    {acct.active === 'yes' && (
                      <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-[#34d399]/15 text-[#34d399] border border-[#34d399]/30">
                        ACTIVE
                      </span>
                    )}
                    <span
                      className={`text-[10px] px-1.5 py-0.5 rounded-full border ${
                        acct.type === 'service_account'
                          ? 'bg-[#fbbf24]/15 text-[#fbbf24] border-[#fbbf24]/30'
                          : 'bg-[#1e2738] text-[#7a8599] border-[#1e2738]'
                      }`}
                    >
                      {acct.type === 'service_account' ? 'SERVICE' : 'USER'}
                    </span>
                  </div>
                </div>
                {isSelected && (
                  <span className="text-[#60a5fa] text-base flex-shrink-0">
                    ●
                  </span>
                )}
              </div>
            </button>
          );
        })}
      </div>

      {isSvcAcct && (
        <div className="rounded-md border border-[#fbbf24]/30 bg-[#fbbf24]/10 p-3 text-xs text-[#fbbf24]">
          ⚠ Service accounts typically lack the project-Owner role required for
          first-time provisioning. If terraform apply fails on IAM, pick a user
          account instead and re-run.
        </div>
      )}

      <div className="flex justify-end pt-2">
        <button
          disabled={!selected}
          onClick={onNext}
          className="px-4 py-2 rounded-md text-sm font-medium bg-[#60a5fa]/15 text-[#60a5fa] hover:bg-[#60a5fa]/25 border border-[#60a5fa]/30 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          Next: Preflight →
        </button>
      </div>
    </section>
  );
}

// ── Step 2: Preflight ───────────────────────────────────────────────

function PreflightStep({
  onBack,
  onNext,
}: {
  onBack: () => void;
  onNext: () => void;
}) {
  const q = useMigratePreflightQuery('gcp');
  const overall = q.data?.overall;
  const canAdvance = overall === 'OK';

  return (
    <section className="space-y-4">
      <header className="flex items-baseline justify-between gap-3">
        <div>
          <h2 className="text-base font-semibold text-[#e2e8f0]">
            2. Preflight checks
          </h2>
          <p className="text-xs text-[#7a8599] mt-1">
            Probes gcloud login, terraform, billing API, IAM. Fix the failing
            ones, then recheck.
          </p>
        </div>
        <button
          onClick={() => q.refetch()}
          disabled={q.isFetching}
          className="px-3 py-1.5 rounded-md text-xs font-medium bg-[#1e2738] text-[#cbd5e1] hover:bg-[#1e2738]/70 border border-[#1e2738] disabled:opacity-50 transition-colors"
        >
          {q.isFetching ? '↻ Checking…' : '↻ Recheck'}
        </button>
      </header>

      {q.isLoading && !q.data && (
        <div className="space-y-2">
          <Skeleton className="h-10" />
          <Skeleton className="h-10" />
          <Skeleton className="h-10" />
        </div>
      )}
      {q.error && <ErrorPanel error={q.error} onRetry={() => q.refetch()} />}

      {q.data && (
        <>
          <div
            className={`rounded-md border px-3 py-2 text-sm ${
              canAdvance
                ? 'bg-[#34d399]/10 border-[#34d399]/30 text-[#34d399]'
                : 'bg-[#fbbf24]/10 border-[#fbbf24]/30 text-[#fbbf24]'
            }`}
          >
            <strong>{overall}</strong>
            {' — '}
            <span className="opacity-80">
              {canAdvance
                ? 'all required probes pass; you may continue.'
                : 'at least one required probe failed. Fix and recheck.'}
            </span>
          </div>

          {/* Stage 0a — Bootstrap project. Only shown when (a) the
              `gcloud project exists` probe is MISSING AND (b) the
              gcp_bootstrap_enabled runtime setting is ON. */}
          <BootstrapProjectCard
            preflight={q.data}
            onSuccess={() => q.refetch()}
          />

          <div className="space-y-2">
            {q.data.probes.map((p) => (
              <ProbeRow key={p.name} probe={p} />
            ))}
          </div>
        </>
      )}

      <div className="flex items-center justify-between pt-2">
        <button
          onClick={onBack}
          className="px-3 py-1.5 rounded-md text-xs font-medium text-[#7a8599] hover:text-[#cbd5e1] hover:bg-[#1e2738] transition-colors"
        >
          ← Back
        </button>
        <button
          disabled={!canAdvance}
          onClick={onNext}
          className="px-4 py-2 rounded-md text-sm font-medium bg-[#60a5fa]/15 text-[#60a5fa] hover:bg-[#60a5fa]/25 border border-[#60a5fa]/30 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          Next: Configure →
        </button>
      </div>
    </section>
  );
}

function BootstrapProjectCard({
  preflight,
  onSuccess,
}: {
  preflight: { probes: PreflightProbe[] };
  onSuccess: () => void;
}) {
  const rs = useRuntimeSettingsQuery();
  const enabled = Boolean(rs.data?.gcp_bootstrap_enabled);
  const projectMissing = preflight.probes.some(
    (p) => p.name === 'gcloud project exists' && p.status === 'MISSING',
  );
  const [projectId, setProjectId] = useState('');
  const [billing, setBilling] = useState('');
  const [orgId, setOrgId] = useState('');
  const [phrase, setPhrase] = useState('');
  const mut = useBootstrapProjectMutation();

  if (!enabled || !projectMissing) return null;

  const canSubmit =
    projectId.trim().length >= 6 &&
    /^[A-Z0-9]{6}-[A-Z0-9]{6}-[A-Z0-9]{6}$/.test(billing.trim()) &&
    phrase === 'CREATE GCP PROJECT' &&
    !mut.isPending;

  return (
    <section
      data-testid="bootstrap-card"
      className="rounded-lg border border-[#60a5fa]/40 bg-[#60a5fa]/5 p-4 space-y-3"
    >
      <h3 className="text-sm font-semibold text-[#60a5fa]">
        ☁ Bootstrap project (Stage 0a)
      </h3>
      <p className="text-[11px] text-[#cbd5e1]">
        The target project doesn't exist yet. Because{' '}
        <code className="font-mono">gcp_bootstrap_enabled</code> is ON, the wizard
        can create it for you (typed-phrase gated). The bootstrap is
        idempotent — re-runs against an existing project are no-ops.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
        <label className="block">
          <div className="text-[10px] uppercase tracking-wider text-[#7a8599] mb-1">
            Project ID
          </div>
          <input
            type="text"
            value={projectId}
            onChange={(e) => setProjectId(e.target.value)}
            placeholder="botarmy-prod-abc123"
            className="w-full px-3 py-1.5 rounded-md bg-[#0a0e14] border border-[#1e2738] font-mono text-[#e2e8f0] focus:outline-none focus:border-[#60a5fa]/50"
          />
        </label>
        <label className="block">
          <div className="text-[10px] uppercase tracking-wider text-[#7a8599] mb-1">
            Billing account
          </div>
          <input
            type="text"
            value={billing}
            onChange={(e) => setBilling(e.target.value.toUpperCase())}
            placeholder="01ABCD-EFGH12-IJ34KL"
            className="w-full px-3 py-1.5 rounded-md bg-[#0a0e14] border border-[#1e2738] font-mono text-[#e2e8f0] focus:outline-none focus:border-[#60a5fa]/50"
          />
        </label>
        <label className="block">
          <div className="text-[10px] uppercase tracking-wider text-[#7a8599] mb-1">
            Org ID (optional)
          </div>
          <input
            type="text"
            value={orgId}
            onChange={(e) => setOrgId(e.target.value)}
            placeholder="(empty = no organization parent)"
            className="w-full px-3 py-1.5 rounded-md bg-[#0a0e14] border border-[#1e2738] font-mono text-[#e2e8f0] focus:outline-none focus:border-[#60a5fa]/50"
          />
        </label>
        <label className="block">
          <div className="text-[10px] uppercase tracking-wider text-[#7a8599] mb-1">
            Type 'CREATE GCP PROJECT' to confirm
          </div>
          <input
            type="text"
            value={phrase}
            onChange={(e) => setPhrase(e.target.value)}
            placeholder="CREATE GCP PROJECT"
            className={`w-full px-3 py-1.5 rounded-md bg-[#0a0e14] border font-mono focus:outline-none ${
              phrase && phrase !== 'CREATE GCP PROJECT'
                ? 'border-[#f87171]/50 text-[#f87171]'
                : 'border-[#1e2738] text-[#e2e8f0] focus:border-[#60a5fa]/50'
            }`}
          />
        </label>
      </div>

      {mut.isError && (
        <div className="text-xs text-[#f87171]">
          Bootstrap failed: {readableError(mut.error)}
        </div>
      )}
      {mut.data && (
        <div className="text-xs">
          <div
            className={
              mut.data.ok
                ? 'text-[#34d399]'
                : 'text-[#f87171]'
            }
          >
            {mut.data.ok ? '✓ Bootstrap succeeded' : `✗ Bootstrap failed (rc=${mut.data.rc})`}
          </div>
          {mut.data.stdout && (
            <pre className="mt-2 max-h-40 overflow-auto rounded-md border border-[#1e2738] bg-[#0a0e14] p-2 font-mono text-[10px] text-[#cbd5e1]">
              {mut.data.stdout}
            </pre>
          )}
        </div>
      )}

      <div className="flex items-center justify-end gap-2">
        <button
          disabled={!canSubmit}
          onClick={() => {
            mut.mutate(
              {
                project_id: projectId.trim(),
                billing_account: billing.trim(),
                org_id: orgId.trim() || null,
                confirm_phrase: phrase,
                dry_run: false,
              },
              { onSuccess: () => onSuccess() },
            );
          }}
          className="px-4 py-1.5 rounded-md text-xs font-medium bg-[#60a5fa]/15 text-[#60a5fa] hover:bg-[#60a5fa]/25 border border-[#60a5fa]/30 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          {mut.isPending ? 'Creating…' : 'Create project'}
        </button>
      </div>
    </section>
  );
}

function ProbeRow({ probe }: { probe: PreflightProbe }) {
  const s = PROBE_STYLE[probe.status];
  return (
    <div className="rounded-md border border-[#1e2738] bg-[#111820] p-3">
      <div className="flex items-baseline gap-3">
        <span className={`text-base ${s.fg} flex-shrink-0`}>{s.glyph}</span>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 flex-wrap">
            <code className="text-sm font-mono text-[#e2e8f0]">
              {probe.name}
            </code>
            <span className={`text-[10px] font-medium ${s.fg}`}>{s.label}</span>
            {!probe.required && (
              <span className="text-[10px] text-[#7a8599]">optional</span>
            )}
          </div>
          {probe.detail && (
            <div
              className={`text-xs mt-1 ${
                probe.status === 'OK'
                  ? 'text-[#7a8599]'
                  : 'text-[#cbd5e1]'
              }`}
            >
              {probe.detail}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Step 3: Configure ───────────────────────────────────────────────

function ConfigureStep({
  projectId,
  setProjectId,
  region,
  setRegion,
  tier,
  setTier,
  onBack,
  onNext,
  cost,
  costLoading,
  costError,
}: {
  projectId: string;
  setProjectId: (v: string) => void;
  region: string;
  setRegion: (v: string) => void;
  tier: Tier;
  setTier: (v: Tier) => void;
  onBack: () => void;
  onNext: () => void;
  cost: CostResponse | null;
  costLoading: boolean;
  costError: unknown;
}) {
  const canAdvance = projectId.trim().length > 0 && region.trim().length > 0;

  return (
    <section className="space-y-4">
      <header>
        <h2 className="text-base font-semibold text-[#e2e8f0]">
          3. Configure target
        </h2>
        <p className="text-xs text-[#7a8599] mt-1">
          Project, region, and tier. Cost estimate refreshes as you type.
        </p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2 space-y-4">
          <Field label="Project ID">
            <input
              type="text"
              value={projectId}
              onChange={(e) => setProjectId(e.target.value)}
              placeholder="botarmy-prod-abc123"
              className="w-full px-3 py-2 rounded-md bg-[#0a0e14] border border-[#1e2738] text-sm text-[#e2e8f0] font-mono focus:outline-none focus:border-[#60a5fa]/50"
            />
            <p className="text-[10px] text-[#7a8599] mt-1">
              Must already exist with billing enabled. Terraform will refuse
              otherwise.
            </p>
          </Field>

          <Field label="Region">
            <input
              type="text"
              value={region}
              onChange={(e) => setRegion(e.target.value)}
              placeholder={DEFAULT_REGION}
              className="w-full px-3 py-2 rounded-md bg-[#0a0e14] border border-[#1e2738] text-sm text-[#e2e8f0] font-mono focus:outline-none focus:border-[#60a5fa]/50"
            />
            <p className="text-[10px] text-[#7a8599] mt-1">
              Default <code className="font-mono">{DEFAULT_REGION}</code>{' '}
              (Hamina, FI) is cheapest for EU operators.
            </p>
          </Field>

          <Field label="Tier">
            <div className="space-y-2">
              <TierRadio
                value="cheapest"
                checked={tier === 'cheapest'}
                onChange={() => setTier('cheapest')}
                label="cheapest"
                hint="Single GKE Autopilot node, no monitoring, no domain. Target: $40–60/mo."
              />
              <TierRadio
                value="prod"
                checked={tier === 'prod'}
                onChange={() => setTier('prod')}
                label="prod"
                hint="HA pool + Cloud Monitoring + Cloud SQL HA. Target: $200–300/mo."
              />
            </div>
          </Field>
        </div>

        {/* Cost sidebar */}
        <aside className="space-y-3">
          <h3 className="text-xs font-semibold text-[#7a8599] uppercase tracking-wider">
            Estimated cost
          </h3>
          {costLoading && <Skeleton className="h-24" />}
          {Boolean(costError) && (
            <div className="text-xs text-[#f87171]">
              Cost estimate failed: {readableError(costError)}
            </div>
          )}
          {cost && !costLoading && (
            <div className="rounded-md border border-[#1e2738] bg-[#111820] p-3 space-y-2">
              <div className="flex items-baseline justify-between">
                <span className="text-xs text-[#7a8599]">monthly</span>
                <span className="text-lg font-semibold text-[#e2e8f0]">
                  ${cost.total_monthly_usd.toFixed(2)}
                </span>
              </div>
              <div className="flex items-baseline justify-between">
                <span className="text-xs text-[#7a8599]">annual</span>
                <span className="text-xs font-mono text-[#cbd5e1]">
                  ${cost.total_annual_usd.toFixed(2)}
                </span>
              </div>
              <div className="border-t border-[#1e2738] pt-2 space-y-1">
                {cost.line_items.map((li, idx) => (
                  <div
                    key={`${li.category}-${li.resource}-${idx}`}
                    className="flex items-baseline justify-between gap-2 text-[11px]"
                  >
                    <div className="min-w-0">
                      <div className="text-[#cbd5e1] truncate">
                        {li.resource}
                      </div>
                      <div className="text-[#7a8599] text-[10px] font-mono">
                        {li.category}
                        {li.note ? ` · ${li.note}` : ''}
                      </div>
                    </div>
                    <span className="font-mono text-[#cbd5e1] flex-shrink-0">
                      ${li.monthly_usd.toFixed(2)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </aside>
      </div>

      {/* Hardening card (Step 3.5 — surfaced inside Configure to keep the 6-step flow) */}
      <HardeningCard />

      <div className="flex items-center justify-between pt-2">
        <button
          onClick={onBack}
          className="px-3 py-1.5 rounded-md text-xs font-medium text-[#7a8599] hover:text-[#cbd5e1] hover:bg-[#1e2738] transition-colors"
        >
          ← Back
        </button>
        <button
          disabled={!canAdvance}
          onClick={onNext}
          className="px-4 py-2 rounded-md text-sm font-medium bg-[#60a5fa]/15 text-[#60a5fa] hover:bg-[#60a5fa]/25 border border-[#60a5fa]/30 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          Next: Confirm →
        </button>
      </div>
    </section>
  );
}

function HardeningCard() {
  const rs = useRuntimeSettingsQuery();
  const profile: string = (rs.data?.hardening_profile as string) ?? 'strict';
  const binauthzMode: string = (rs.data?.binauthz_mode as string) ?? 'AUDIT';
  const q = useHardeningPreviewQuery(profile, binauthzMode);
  const preview = q.data;

  return (
    <section
      data-testid="hardening-card"
      className="mt-2 rounded-lg border border-[#1e2738] bg-[#0a0e14] p-4 space-y-3"
    >
      <header className="flex items-baseline justify-between">
        <h3 className="text-sm font-semibold text-[#e2e8f0]">
          🛡 Hardening (auto-detected)
        </h3>
        <span className="text-[10px] text-[#7a8599] uppercase tracking-wider">
          profile:{' '}
          <span className="text-[#cbd5e1]">{profile}</span>
          {' · binauthz: '}
          <span className="text-[#cbd5e1]">{binauthzMode}</span>
        </span>
      </header>
      <p className="text-[11px] text-[#7a8599]">
        Tune the profile (off / basic / strict) and Binary Authorization mode (AUDIT / ENFORCE) in{' '}
        <a
          href="/cp/settings"
          className="text-[#60a5fa] hover:underline"
        >
          /cp/settings
        </a>
        . Default <code className="font-mono">strict + AUDIT</code> is safe to apply.
      </p>

      {q.isLoading && <Skeleton className="h-20" />}
      {q.isError && (
        <div className="text-xs text-[#f87171]">
          Hardening preview failed: {readableError(q.error)}
        </div>
      )}
      {preview && !q.isLoading && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-[11px]">
          <div className="space-y-1">
            <div className="flex items-baseline justify-between">
              <span className="text-[#7a8599]">Tailnet reachable</span>
              <span
                className={
                  preview.tailnet_reachable
                    ? 'text-[#34d399]'
                    : 'text-[#fbbf24]'
                }
              >
                {preview.tailnet_reachable ? '✓ yes' : '◷ no'}
              </span>
            </div>
            <div className="flex items-baseline justify-between">
              <span className="text-[#7a8599]">Tailnet CIDR</span>
              <span className="font-mono text-[#cbd5e1]">
                {preview.tailnet_cidr ?? '(none)'}
              </span>
            </div>
            <div className="flex items-baseline justify-between">
              <span className="text-[#7a8599]">Laptop public IP</span>
              <span className="font-mono text-[#cbd5e1]">
                {preview.laptop_public_ip ?? '(unable to detect)'}
              </span>
            </div>
            <div className="flex items-baseline justify-between">
              <span className="text-[#7a8599]">Workspace org_id</span>
              <span className="font-mono text-[#cbd5e1]">
                {preview.org_id ?? '(not detected)'}
              </span>
            </div>
          </div>

          <div className="space-y-1">
            <div className="text-[#7a8599] mb-1">
              Master allowlist ({preview.recommended_cidrs.length})
            </div>
            {preview.recommended_cidrs.length === 0 && (
              <div className="text-[#fbbf24]">
                ⚠ empty — anyone with valid IAM can reach K8s API
              </div>
            )}
            {preview.recommended_cidrs.map((c) => (
              <div
                key={c.cidr_block}
                className="flex items-baseline justify-between gap-2"
              >
                <code className="font-mono text-[#cbd5e1]">{c.cidr_block}</code>
                <span className="text-[10px] text-[#7a8599] truncate">
                  {c.display_name}
                </span>
              </div>
            ))}
          </div>

          {preview.notes.length > 0 && (
            <ul className="md:col-span-2 space-y-1 mt-1">
              {preview.notes.map((n, i) => (
                <li key={i} className="text-[#fbbf24]">
                  • {n}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </section>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <div className="text-xs font-semibold text-[#7a8599] uppercase tracking-wider mb-1.5">
        {label}
      </div>
      {children}
    </label>
  );
}

function TierRadio({
  value,
  checked,
  onChange,
  label,
  hint,
}: {
  value: string;
  checked: boolean;
  onChange: () => void;
  label: string;
  hint: string;
}) {
  return (
    <label
      className={`flex items-start gap-3 p-3 rounded-md border cursor-pointer transition-colors ${
        checked
          ? 'border-[#60a5fa]/40 bg-[#60a5fa]/5'
          : 'border-[#1e2738] bg-[#111820] hover:border-[#60a5fa]/30'
      }`}
    >
      <input
        type="radio"
        name="tier"
        value={value}
        checked={checked}
        onChange={onChange}
        className="mt-0.5 accent-[#60a5fa]"
      />
      <div className="min-w-0">
        <div className="text-sm font-mono text-[#e2e8f0]">{label}</div>
        <div className="text-[11px] text-[#7a8599] mt-0.5">{hint}</div>
      </div>
    </label>
  );
}

// ── Step 4: Confirm ─────────────────────────────────────────────────

function ConfirmStep({
  account,
  projectId,
  region,
  tier,
  cost,
  phrase,
  setPhrase,
  budgetCap,
  setBudgetCap,
  onBack,
  onStart,
  starting,
  startError,
}: {
  account: string;
  projectId: string;
  region: string;
  tier: Tier;
  cost: CostResponse | null;
  phrase: string;
  setPhrase: (v: string) => void;
  budgetCap: number;
  setBudgetCap: (v: number) => void;
  onBack: () => void;
  onStart: () => void;
  starting: boolean;
  startError: unknown;
}) {
  const phraseOk = phrase === TYPED_PHRASE;
  const budgetOk = budgetCap > 0 && budgetCap <= MAX_BUDGET_USD;
  const canStart = phraseOk && budgetOk && !starting;

  // Productization plan WP D — the execute-gate is now a runtime
  // setting toggleable from this very page. Reads the live value via
  // useRuntimeSettingsQuery + flips via useUpdateRuntimeSettings.
  const rs = useRuntimeSettingsQuery();
  const updateRs = useUpdateRuntimeSettings();
  const liveExecute = Boolean(rs.data?.migrate_live_execute);

  return (
    <section className="space-y-4">
      <header>
        <h2 className="text-base font-semibold text-[#e2e8f0]">
          4. Confirm + start
        </h2>
        <p className="text-xs text-[#7a8599] mt-1">
          Review the plan, type the safety phrase, set a budget cap.
        </p>
      </header>

      <div className="rounded-md border border-[#1e2738] bg-[#111820] p-4 space-y-2 text-sm">
        <SummaryRow label="Target" value="gcp" mono />
        <SummaryRow label="Account" value={account} mono />
        <SummaryRow label="Project" value={projectId} mono />
        <SummaryRow label="Region" value={region} mono />
        <SummaryRow label="Tier" value={tier} mono />
        {cost && (
          <>
            <SummaryRow
              label="Estimated $/mo"
              value={`$${cost.total_monthly_usd.toFixed(2)}`}
              mono
            />
            <SummaryRow
              label="Annual"
              value={`$${cost.total_annual_usd.toFixed(2)}`}
              mono
            />
          </>
        )}
      </div>

      {cost && (
        <details className="rounded-md border border-[#1e2738] bg-[#111820] p-3">
          <summary className="text-xs font-semibold text-[#7a8599] uppercase tracking-wider cursor-pointer">
            Cost line items ({cost.line_items.length})
          </summary>
          <div className="mt-2 space-y-1">
            {cost.line_items.map((li, idx) => (
              <div
                key={`${li.category}-${li.resource}-${idx}`}
                className="flex items-baseline justify-between gap-2 text-[11px] py-0.5"
              >
                <span className="text-[#cbd5e1] truncate">
                  {li.resource}
                  <span className="text-[#7a8599] ml-2">{li.category}</span>
                </span>
                <span className="font-mono text-[#cbd5e1]">
                  ${li.monthly_usd.toFixed(2)}
                </span>
              </div>
            ))}
          </div>
        </details>
      )}

      {/* Execute-gate toggle (productization WP D, 2026-05-17).
          Mirrors runtime_settings.migrate_live_execute. Flipping ON
          here makes the very next /start invocation actually run
          terraform/gcloud/kubectl against real cloud APIs. The flip
          itself emits a `cloud_migration:execute_policy_changed`
          identity-ledger landmark so annual reflection sees it. */}
      <div
        className={`rounded-md border p-3 ${
          liveExecute
            ? 'border-[#f87171]/40 bg-[#f87171]/5'
            : 'border-[#fbbf24]/30 bg-[#fbbf24]/5'
        }`}
      >
        <div className="flex items-center justify-between gap-3">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <span className="text-xs font-semibold text-[#e2e8f0] uppercase tracking-wider">
                Execute mode
              </span>
              <span
                className={`text-[10px] font-mono px-2 py-0.5 rounded ${
                  liveExecute
                    ? 'bg-[#f87171]/15 text-[#f87171] border border-[#f87171]/30'
                    : 'bg-[#fbbf24]/15 text-[#fbbf24] border border-[#fbbf24]/30'
                }`}
              >
                {liveExecute ? 'LIVE — real spend' : 'DRY-SHELL — report only'}
              </span>
            </div>
            <p className="text-[11px] text-[#cbd5e1] mt-1">
              {liveExecute ? (
                <>
                  Every subprocess will actually run. terraform apply will
                  create real cloud resources. <strong>Real money will be spent.</strong>
                </>
              ) : (
                <>
                  Orchestrator runs end-to-end but every cloud-mutating
                  subprocess returns a <code className="font-mono">&lt;dry: …&gt;</code> placeholder.
                  Useful for proving the pipeline is correct without spending money.
                </>
              )}
            </p>
            <p className="text-[10px] text-[#7a8599] mt-1">
              Flipping this writes to <code className="font-mono">workspace/runtime_settings.json</code>{' '}
              and emits a continuity-ledger landmark. Persists across gateway restart.
            </p>
          </div>
          <button
            type="button"
            disabled={updateRs.isPending || rs.isLoading}
            onClick={() => {
              const next = !liveExecute;
              const confirmMsg = next
                ? 'Enable LIVE execute mode? The next migration will spend real cloud money.'
                : 'Return to DRY-SHELL mode? Future migrations will produce reports only.';
              if (window.confirm(confirmMsg)) {
                updateRs.mutate({ migrate_live_execute: next });
              }
            }}
            className={`shrink-0 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
              liveExecute
                ? 'bg-[#f87171]/15 text-[#f87171] hover:bg-[#f87171]/25 border border-[#f87171]/30'
                : 'bg-[#34d399]/15 text-[#34d399] hover:bg-[#34d399]/25 border border-[#34d399]/30'
            } disabled:opacity-40`}
          >
            {updateRs.isPending
              ? '↻ Flipping…'
              : liveExecute
              ? 'Switch to DRY-SHELL'
              : 'Enable LIVE'}
          </button>
        </div>
        {updateRs.error instanceof Error && (
          <div className="text-[10px] text-[#f87171] mt-2">
            Toggle failed: {updateRs.error.message}
          </div>
        )}
      </div>

      <Field label={`Safety phrase (type: ${TYPED_PHRASE})`}>
        <input
          type="text"
          value={phrase}
          onChange={(e) => setPhrase(e.target.value)}
          placeholder={TYPED_PHRASE}
          autoComplete="off"
          spellCheck={false}
          className={`w-full px-3 py-2 rounded-md bg-[#0a0e14] border text-sm text-[#e2e8f0] font-mono focus:outline-none ${
            phraseOk
              ? 'border-[#34d399]/40 focus:border-[#34d399]/60'
              : 'border-[#1e2738] focus:border-[#fbbf24]/40'
          }`}
        />
        <p className="text-[10px] text-[#7a8599] mt-1">
          Must match exactly. The backend also enforces this; we refuse first
          for fast feedback.
        </p>
      </Field>

      <Field label={`Budget cap (USD, max $${MAX_BUDGET_USD})`}>
        <input
          type="number"
          min={1}
          max={MAX_BUDGET_USD}
          value={Number.isFinite(budgetCap) ? budgetCap : ''}
          onChange={(e) => setBudgetCap(Number(e.target.value))}
          className={`w-full px-3 py-2 rounded-md bg-[#0a0e14] border text-sm text-[#e2e8f0] font-mono focus:outline-none ${
            budgetOk
              ? 'border-[#1e2738] focus:border-[#60a5fa]/50'
              : 'border-[#f87171]/40 focus:border-[#f87171]/60'
          }`}
        />
        <p className="text-[10px] text-[#7a8599] mt-1">
          Hard ceiling. If the cost estimate exceeds this, the orchestrator
          refuses to proceed.
        </p>
      </Field>

      {cost && cost.total_monthly_usd > budgetCap && budgetOk && (
        <div className="text-xs text-[#fbbf24] bg-[#fbbf24]/10 border border-[#fbbf24]/30 rounded-md p-2">
          ⚠ Estimate (${cost.total_monthly_usd.toFixed(2)}) exceeds your cap
          ($&nbsp;{budgetCap}). Raise the cap or switch tiers.
        </div>
      )}

      {Boolean(startError) && (
        <div className="text-xs text-[#f87171] bg-[#f87171]/10 border border-[#f87171]/30 rounded-md p-2">
          {readableError(startError)}
        </div>
      )}

      <div className="flex items-center justify-between pt-2">
        <button
          onClick={onBack}
          disabled={starting}
          className="px-3 py-1.5 rounded-md text-xs font-medium text-[#7a8599] hover:text-[#cbd5e1] hover:bg-[#1e2738] disabled:opacity-40 transition-colors"
        >
          ← Back
        </button>
        <button
          disabled={!canStart}
          onClick={onStart}
          className="px-4 py-2 rounded-md text-sm font-medium bg-[#34d399]/15 text-[#34d399] hover:bg-[#34d399]/25 border border-[#34d399]/30 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          {starting ? '↻ Starting…' : '▶ Start migration'}
        </button>
      </div>
    </section>
  );
}

function SummaryRow({
  label,
  value,
  mono = false,
}: {
  label: string;
  value: string;
  mono?: boolean;
}) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <span className="text-xs text-[#7a8599]">{label}</span>
      <span
        className={`text-sm text-[#e2e8f0] truncate ${
          mono ? 'font-mono' : ''
        }`}
      >
        {value}
      </span>
    </div>
  );
}

// ── Step 5: Execute (progress) ──────────────────────────────────────

function ExecuteStep({
  runId,
  dryShell,
  onBack,
  onComplete,
  onCancel,
  cancelling,
}: {
  runId: string;
  dryShell: boolean;
  onBack: () => void;
  onComplete: (record: RunRecord) => void;
  onCancel: () => void;
  cancelling: boolean;
}) {
  const q = useMigrateRunQuery(runId);
  const record = q.data;

  // Auto-advance to Outcome when terminal.
  useEffect(() => {
    if (record && TERMINAL_STATUSES.has(record.status)) {
      onComplete(record);
    }
  }, [record, onComplete]);

  return (
    <section className="space-y-4">
      <header className="flex items-baseline justify-between gap-3">
        <div>
          <h2 className="text-base font-semibold text-[#e2e8f0]">
            5. Migrating…
          </h2>
          <p className="text-xs text-[#7a8599] mt-1">
            Polling every 2 s. Run id:{' '}
            <code className="font-mono text-[#cbd5e1]">{runId}</code>
          </p>
        </div>
        {record && <StatusBadge status={record.status} />}
      </header>

      {dryShell && (
        <div className="text-xs text-[#fbbf24] bg-[#fbbf24]/10 border border-[#fbbf24]/30 rounded-md p-3">
          <strong className="block mb-1">⚠ Dry-shell mode</strong>
          Orchestrator runs but no real cloud spend. Every subprocess returns a{' '}
          <code className="font-mono">&lt;dry: …&gt;</code> placeholder. To
          actually provision, go back to step 4, flip{' '}
          <strong>Execute mode → LIVE</strong>, and start a new run. (The legacy{' '}
          <code className="font-mono">BOTARMY_MIGRATE_LIVE_EXECUTE=1</code> env
          var on the gateway also works.)
        </div>
      )}

      {!record && <Skeleton className="h-32" />}

      {record && (
        <div className="space-y-3">
          <ProgressBar pct={record.progress_pct} status={record.status} />
          <div className="rounded-md border border-[#1e2738] bg-[#111820] p-3 space-y-2">
            <div className="flex items-baseline justify-between gap-3">
              <span className="text-xs text-[#7a8599]">Current step</span>
              <code className="text-sm font-mono text-[#e2e8f0]">
                {record.current_step || '—'}
              </code>
            </div>
            <div className="flex items-baseline justify-between gap-3">
              <span className="text-xs text-[#7a8599]">Progress</span>
              <span className="text-sm font-mono text-[#e2e8f0]">
                {record.progress_pct}%
              </span>
            </div>
            {record.detail && (
              <div>
                <div className="text-xs text-[#7a8599] mb-1">Detail</div>
                <div className="text-xs text-[#cbd5e1] whitespace-pre-wrap break-words">
                  {record.detail}
                </div>
              </div>
            )}
            {record.error && (
              <div>
                <div className="text-xs text-[#f87171] mb-1">Error</div>
                <code className="text-xs font-mono text-[#f87171] whitespace-pre-wrap break-words">
                  {record.error}
                </code>
              </div>
            )}
          </div>
        </div>
      )}

      <div className="flex items-center justify-between pt-2">
        <button
          onClick={onBack}
          className="px-3 py-1.5 rounded-md text-xs font-medium text-[#7a8599] hover:text-[#cbd5e1] hover:bg-[#1e2738] transition-colors"
        >
          ← Back (does not cancel run)
        </button>
        {record && !TERMINAL_STATUSES.has(record.status) && (
          <button
            onClick={onCancel}
            disabled={cancelling}
            className="px-4 py-2 rounded-md text-sm font-medium bg-[#f87171]/15 text-[#f87171] hover:bg-[#f87171]/25 border border-[#f87171]/30 disabled:opacity-40 transition-colors"
          >
            {cancelling ? '↻ Cancelling…' : '⏹ Request cancel'}
          </button>
        )}
      </div>
    </section>
  );
}

function ProgressBar({ pct, status }: { pct: number; status: RunStatus }) {
  const safePct = Math.max(0, Math.min(100, pct));
  const barColor =
    status === 'succeeded'
      ? 'bg-[#34d399]'
      : status === 'failed' || status === 'preflight_failed'
      ? 'bg-[#f87171]'
      : status === 'cancelled'
      ? 'bg-[#7a8599]'
      : 'bg-[#60a5fa]';
  return (
    <div className="w-full h-3 rounded-full bg-[#1e2738] overflow-hidden">
      <div
        className={`h-full ${barColor} transition-all duration-500`}
        style={{ width: `${safePct}%` }}
      />
    </div>
  );
}

// ── Step 6: Outcome ─────────────────────────────────────────────────

function OutcomeStep({
  record,
  onRestart,
}: {
  record: RunRecord;
  onRestart: () => void;
}) {
  const terminal = TERMINAL_STATUSES.has(record.status);
  const ok = record.status === 'succeeded';

  return (
    <section className="space-y-4">
      <header className="flex items-baseline justify-between gap-3">
        <div>
          <h2 className="text-base font-semibold text-[#e2e8f0]">6. Outcome</h2>
          <p className="text-xs text-[#7a8599] mt-1">
            {ok
              ? 'Migration succeeded. Manual follow-up below.'
              : terminal
              ? 'Migration ended without success.'
              : 'Still in flight.'}
          </p>
        </div>
        <StatusBadge status={record.status} />
      </header>

      <div className="rounded-md border border-[#1e2738] bg-[#111820] p-4 space-y-2 text-sm">
        <SummaryRow label="Run id" value={record.run_id} mono />
        <SummaryRow label="Target" value={record.target} mono />
        <SummaryRow label="Project" value={record.project_id} mono />
        <SummaryRow label="Region" value={record.region} mono />
        <SummaryRow label="Tier" value={record.tier} mono />
        <SummaryRow label="Account" value={record.active_account} mono />
        {record.started_at && (
          <SummaryRow
            label="Started"
            value={new Date(record.started_at).toLocaleString()}
          />
        )}
        {record.completed_at && (
          <SummaryRow
            label="Completed"
            value={new Date(record.completed_at).toLocaleString()}
          />
        )}
        {record.report_path && (
          <SummaryRow label="Report" value={record.report_path} mono />
        )}
      </div>

      {record.error && (
        <div className="rounded-md border border-[#f87171]/30 bg-[#f87171]/10 p-3">
          <div className="text-xs font-semibold text-[#f87171] uppercase tracking-wider mb-2">
            Error
          </div>
          <code className="text-xs font-mono text-[#f87171] whitespace-pre-wrap break-words">
            {record.error}
          </code>
        </div>
      )}

      {ok && (
        <div className="rounded-md border border-[#34d399]/30 bg-[#34d399]/5 p-4 space-y-2 text-xs text-[#cbd5e1]">
          <div className="text-sm font-semibold text-[#34d399] mb-1">
            ✓ Next steps
          </div>
          <ol className="list-decimal pl-5 space-y-1">
            <li>
              Inspect the migration report at{' '}
              <code className="font-mono">{record.report_path || '<path>'}</code>
            </li>
            <li>
              Run the Phase 4 cutover (DNS, traffic, monitoring handoff) when
              ready. The new cluster is provisioned but no traffic is routed
              yet.
            </li>
            <li>
              If you want to tear down: run{' '}
              <code className="font-mono">terraform destroy</code> in{' '}
              <code className="font-mono">deploy/terraform/{record.target}/</code>{' '}
              manually. The wizard does not auto-rollback infrastructure.
            </li>
          </ol>
        </div>
      )}

      <div className="flex justify-end pt-2">
        <button
          onClick={onRestart}
          className="px-4 py-2 rounded-md text-sm font-medium bg-[#60a5fa]/15 text-[#60a5fa] hover:bg-[#60a5fa]/25 border border-[#60a5fa]/30 transition-colors"
        >
          ↺ Start a new migration
        </button>
      </div>
    </section>
  );
}

// ── Side panel: recent runs ─────────────────────────────────────────

function RecentRunsPanel({
  onInspect,
}: {
  onInspect: (record: RunRecord) => void;
}) {
  const q = useMigrateRunsQuery(10);

  return (
    <aside className="space-y-3">
      <header className="flex items-baseline justify-between">
        <h3 className="text-xs font-semibold text-[#7a8599] uppercase tracking-wider">
          Recent runs
        </h3>
        {q.data?.active_run_id && (
          <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-[#60a5fa]/15 text-[#60a5fa] border border-[#60a5fa]/30 animate-pulse">
            ACTIVE
          </span>
        )}
      </header>

      {q.isLoading && <Skeleton className="h-32" />}
      {q.error && <div className="text-xs text-[#f87171]">Failed to load runs.</div>}

      {q.data && q.data.runs.length === 0 && (
        <div className="text-xs text-[#7a8599] italic">No runs yet.</div>
      )}

      <div className="space-y-2">
        {q.data?.runs.map((r) => (
          <button
            key={r.run_id}
            onClick={() => onInspect(r)}
            className="w-full text-left p-2 rounded-md border border-[#1e2738] bg-[#111820] hover:border-[#60a5fa]/30 hover:bg-[#1e2738]/50 transition-colors"
          >
            <div className="flex items-center justify-between gap-2">
              <code className="text-[10px] font-mono text-[#cbd5e1] truncate">
                {r.run_id}
              </code>
              <StatusBadge status={r.status} />
            </div>
            <div className="text-[10px] text-[#7a8599] mt-1 font-mono truncate">
              {r.project_id || '—'} · {r.target}/{r.tier}
            </div>
            <div className="text-[10px] text-[#7a8599] mt-0.5">
              {r.started_at
                ? new Date(r.started_at).toLocaleString()
                : 'not started'}
            </div>
          </button>
        ))}
      </div>
    </aside>
  );
}

// ── Run detail drawer (for clicking a past run in the side panel) ──

function RunDetailDrawer({
  runId,
  onClose,
}: {
  runId: string | null;
  onClose: () => void;
}) {
  // Hooks run on every render — early-return after.
  const q = useMigrateRunQuery(runId ?? undefined);
  if (!runId) return null;

  return (
    <div
      className="fixed inset-0 z-40 flex justify-end bg-black/60"
      onClick={onClose}
    >
      <div
        className="w-full max-w-2xl bg-[#0a0e14] border-l border-[#1e2738] flex flex-col h-full pt-[env(safe-area-inset-top)] pb-[env(safe-area-inset-bottom)]"
        onClick={(e) => e.stopPropagation()}
      >
        <header className="flex items-center justify-between px-5 py-4 border-b border-[#1e2738] flex-shrink-0">
          <div className="min-w-0">
            <div className="text-xs text-[#7a8599] font-mono truncate">
              Run detail
            </div>
            <code className="text-sm font-mono text-[#e2e8f0] truncate block">
              {runId}
            </code>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded text-[#7a8599] hover:text-[#e2e8f0] hover:bg-[#1e2738]"
            aria-label="Close drawer"
          >
            <svg
              className="w-5 h-5"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        </header>

        <div className="flex-1 overflow-y-auto p-5 space-y-4">
          {q.isLoading && !q.data && <Skeleton className="h-48" />}
          {q.error && <ErrorPanel error={q.error} onRetry={() => q.refetch()} />}
          {q.data && (
            <>
              <div className="flex items-center gap-2 flex-wrap">
                <StatusBadge status={q.data.status} />
                <span className="text-xs text-[#7a8599] font-mono">
                  {q.data.target}/{q.data.tier}
                </span>
              </div>

              <ProgressBar pct={q.data.progress_pct} status={q.data.status} />

              <div className="rounded-md border border-[#1e2738] bg-[#111820] p-3 space-y-2 text-sm">
                <SummaryRow label="Project" value={q.data.project_id} mono />
                <SummaryRow label="Region" value={q.data.region} mono />
                <SummaryRow label="Account" value={q.data.active_account} mono />
                <SummaryRow
                  label="Current step"
                  value={q.data.current_step || '—'}
                  mono
                />
                <SummaryRow label="Progress" value={`${q.data.progress_pct}%`} mono />
                {q.data.started_at && (
                  <SummaryRow
                    label="Started"
                    value={new Date(q.data.started_at).toLocaleString()}
                  />
                )}
                {q.data.completed_at && (
                  <SummaryRow
                    label="Completed"
                    value={new Date(q.data.completed_at).toLocaleString()}
                  />
                )}
                {q.data.report_path && (
                  <SummaryRow label="Report" value={q.data.report_path} mono />
                )}
              </div>

              {q.data.detail && (
                <div>
                  <div className="text-xs font-semibold text-[#7a8599] uppercase tracking-wider mb-1">
                    Detail
                  </div>
                  <div className="text-xs text-[#cbd5e1] whitespace-pre-wrap break-words">
                    {q.data.detail}
                  </div>
                </div>
              )}

              {q.data.error && (
                <div className="rounded-md border border-[#f87171]/30 bg-[#f87171]/10 p-3">
                  <div className="text-xs font-semibold text-[#f87171] uppercase tracking-wider mb-1">
                    Error
                  </div>
                  <code className="text-xs font-mono text-[#f87171] whitespace-pre-wrap break-words">
                    {q.data.error}
                  </code>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Helpers ─────────────────────────────────────────────────────────

function readableError(err: unknown): string {
  if (err instanceof ApiError) {
    // Try to parse {"detail": "…"} from FastAPI errors.
    try {
      const parsed = JSON.parse(err.body);
      if (parsed && typeof parsed.detail === 'string') return parsed.detail;
    } catch {
      // not JSON
    }
    return `${err.status}: ${err.body.slice(0, 200)}`;
  }
  if (err instanceof Error) return err.message;
  return String(err);
}

// ── Wizard root ─────────────────────────────────────────────────────

export function MigratePage() {
  const [step, setStep] = useState<StepId>('account');
  const [reached, setReached] = useState<Set<StepId>>(new Set(['account']));

  // Form state
  const [account, setAccount] = useState<string | null>(null);
  const [projectId, setProjectId] = useState('');
  const [region, setRegion] = useState(DEFAULT_REGION);
  const [tier, setTier] = useState<Tier>('cheapest');
  const [phrase, setPhrase] = useState('');
  const [budgetCap, setBudgetCap] = useState<number>(DEFAULT_BUDGET_USD);

  // Run state
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const [dryShell, setDryShell] = useState(false);
  const [terminalRecord, setTerminalRecord] = useState<RunRecord | null>(null);

  // Side-panel inspection state (independent of the wizard flow)
  const [inspectId, setInspectId] = useState<string | null>(null);

  // Mutations
  const costMut = useMigrateCostMutation();
  const startMut = useMigrateStartMutation();
  const cancelMut = useMigrateCancelMutation();

  // Refetch cost whenever configure inputs change.
  useEffect(() => {
    if (step !== 'configure' && step !== 'confirm') return;
    if (!projectId && step === 'configure') return;
    costMut.mutate({
      target: 'gcp',
      tier,
      region: region || undefined,
      enable_monitoring: tier === 'prod',
      has_domain: false,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [step, tier, region]);

  const advance = (next: StepId) => {
    setReached((prev) => {
      const ns = new Set(prev);
      ns.add(next);
      return ns;
    });
    setStep(next);
  };

  const handleStart = () => {
    startMut.mutate(
      {
        target: 'gcp',
        tier,
        region: region || undefined,
        project_id: projectId.trim(),
        active_account: account ?? '',
        confirm_phrase: phrase,
        budget_cap_usd: budgetCap,
      },
      {
        onSuccess: (resp) => {
          setActiveRunId(resp.run_id);
          setDryShell(resp.dry_shell_mode);
          setTerminalRecord(null);
          advance('execute');
        },
      },
    );
  };

  const handleCancel = () => {
    if (!activeRunId) return;
    cancelMut.mutate({ runId: activeRunId });
  };

  const handleRestart = () => {
    setStep('account');
    setReached(new Set(['account']));
    setActiveRunId(null);
    setDryShell(false);
    setTerminalRecord(null);
    setPhrase('');
    setProjectId('');
    setBudgetCap(DEFAULT_BUDGET_USD);
    setTier('cheapest');
    setRegion(DEFAULT_REGION);
    setAccount(null);
  };

  // Step renderer
  const stepBody = useMemo(() => {
    switch (step) {
      case 'account':
        return (
          <AccountStep
            selected={account}
            onSelect={(a) => setAccount(a)}
            onNext={() => advance('preflight')}
          />
        );
      case 'preflight':
        return (
          <PreflightStep
            onBack={() => setStep('account')}
            onNext={() => advance('configure')}
          />
        );
      case 'configure':
        return (
          <ConfigureStep
            projectId={projectId}
            setProjectId={setProjectId}
            region={region}
            setRegion={setRegion}
            tier={tier}
            setTier={setTier}
            onBack={() => setStep('preflight')}
            onNext={() => advance('confirm')}
            cost={costMut.data ?? null}
            costLoading={costMut.isPending}
            costError={costMut.error}
          />
        );
      case 'confirm':
        return (
          <ConfirmStep
            account={account ?? ''}
            projectId={projectId}
            region={region}
            tier={tier}
            cost={costMut.data ?? null}
            phrase={phrase}
            setPhrase={setPhrase}
            budgetCap={budgetCap}
            setBudgetCap={setBudgetCap}
            onBack={() => setStep('configure')}
            onStart={handleStart}
            starting={startMut.isPending}
            startError={startMut.error}
          />
        );
      case 'execute':
        if (!activeRunId) {
          return (
            <div className="text-sm text-[#f87171]">
              No active run — go back and click Start.
            </div>
          );
        }
        return (
          <ExecuteStep
            runId={activeRunId}
            dryShell={dryShell}
            onBack={() => setStep('confirm')}
            onComplete={(rec) => {
              setTerminalRecord(rec);
              advance('outcome');
            }}
            onCancel={handleCancel}
            cancelling={cancelMut.isPending}
          />
        );
      case 'outcome':
        if (!terminalRecord) {
          return (
            <div className="text-sm text-[#7a8599]">
              Waiting for run to complete…
            </div>
          );
        }
        return (
          <OutcomeStep record={terminalRecord} onRestart={handleRestart} />
        );
      default:
        return null;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    step,
    account,
    projectId,
    region,
    tier,
    phrase,
    budgetCap,
    activeRunId,
    dryShell,
    terminalRecord,
    costMut.data,
    costMut.isPending,
    costMut.error,
    startMut.isPending,
    startMut.error,
    cancelMut.isPending,
  ]);

  return (
    <div className="space-y-4 max-w-6xl">
      <header className="space-y-2">
        <div className="flex items-baseline justify-between gap-3">
          <div>
            <h1 className="text-xl font-semibold text-[#e2e8f0]">
              ☁ Cloud migration
            </h1>
            <p className="text-xs text-[#7a8599] mt-1">
              Six-step wizard to provision BotArmy on GCP. Each step is
              cancellable; the API enforces a typed safety phrase + $500
              budget cap.
            </p>
          </div>
        </div>
        <StepIndicator current={step} reached={reached} />
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_18rem] gap-6">
        <main className="space-y-4">{stepBody}</main>
        <RecentRunsPanel onInspect={(r) => setInspectId(r.run_id)} />
      </div>

      <RunDetailDrawer runId={inspectId} onClose={() => setInspectId(null)} />
    </div>
  );
}
