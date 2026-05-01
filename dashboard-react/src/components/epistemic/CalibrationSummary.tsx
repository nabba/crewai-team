import type { EpistemicNowReport } from '../../types/epistemic';

export function CalibrationSummary({ report }: { report: EpistemicNowReport }) {
  const total = report.ledger?.length ?? 0;
  const verifiedFraction =
    total > 0
      ? (report.ledger ?? []).filter((c) => c.status === 'verified').length /
        total
      : 1;
  const ledgerHealth =
    report.unverified_load_bearing_count === 0
      ? 1
      : Math.max(0, 1 - report.unverified_load_bearing_count / 5);
  const grounding = report.calibration?.factual_grounding ?? null;
  const attractor = report.calibration?.attractor ?? null;

  // Composite weights ledger health, verified fraction, and (when
  // available) the affect grounding signal equally. When grounding is
  // null, fall back to the two-signal mean.
  const composite =
    grounding === null
      ? (verifiedFraction + ledgerHealth) / 2
      : (verifiedFraction + ledgerHealth + grounding) / 3;

  return (
    <section className="rounded-lg bg-[#111820] border border-[#1e2738] p-4 grid grid-cols-1 md:grid-cols-5 gap-4">
      <Tile
        label="Claims"
        value={String(total)}
        sub={`${report.load_bearing_count} load-bearing`}
      />
      <Tile
        label="Verified"
        value={`${Math.round(verifiedFraction * 100)}%`}
        sub={`${total - (report.ledger ?? []).filter((c) => c.status === 'verified').length} unverified`}
        bar={verifiedFraction}
      />
      <Tile
        label="Ledger health"
        value={`${Math.round(ledgerHealth * 100)}%`}
        sub={`${report.unverified_load_bearing_count} unverified load-bearing`}
        bar={ledgerHealth}
      />
      <Tile
        label="Felt grounding"
        value={grounding === null ? '—' : `${Math.round(grounding * 100)}%`}
        sub={
          grounding === null
            ? 'no affect signal'
            : attractor
              ? `attractor: ${attractor}`
              : `factual_grounding`
        }
        bar={grounding ?? undefined}
      />
      <Tile
        label="Composite"
        value={`${Math.round(composite * 100)}%`}
        sub={
          composite >= 0.7
            ? 'calibrated'
            : composite >= 0.4
              ? 'caution'
              : 'shaky'
        }
        tone={composite >= 0.7 ? 'good' : composite >= 0.4 ? 'warn' : 'bad'}
        bar={composite}
      />
    </section>
  );
}

function Tile({
  label,
  value,
  sub,
  bar,
  tone,
}: {
  label: string;
  value: string;
  sub: string;
  bar?: number;
  tone?: 'good' | 'warn' | 'bad';
}) {
  const valueColor =
    tone === 'good'
      ? 'text-[#34d399]'
      : tone === 'warn'
      ? 'text-[#fbbf24]'
      : tone === 'bad'
      ? 'text-[#f87171]'
      : 'text-[#e2e8f0]';

  const barTone =
    bar === undefined
      ? ''
      : bar >= 0.7
      ? 'bg-[#34d399]'
      : bar >= 0.4
      ? 'bg-[#fbbf24]'
      : 'bg-[#f87171]';

  return (
    <div>
      <div className="text-xs text-[#7a8599] uppercase tracking-wide">
        {label}
      </div>
      <div className={`text-2xl font-semibold mt-1 ${valueColor}`}>{value}</div>
      {bar !== undefined && (
        <div className="h-1.5 bg-[#1e2738] rounded-full overflow-hidden mt-1">
          <div
            className={`h-full ${barTone}`}
            style={{ width: `${Math.max(0, Math.min(1, bar)) * 100}%` }}
          />
        </div>
      )}
      <div className="text-xs text-[#7a8599] mt-1">{sub}</div>
    </div>
  );
}
