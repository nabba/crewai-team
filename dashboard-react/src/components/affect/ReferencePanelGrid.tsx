import type { ReferencePanelReport } from '../../types/affect';

interface ReferencePanelGridProps {
  report: ReferencePanelReport;
}

const DRIFT_STYLE: Record<string, { color: string; bg: string; label: string }> = {
  ok: { color: '#34d399', bg: '#34d3991a', label: 'OK' },
  numbness: { color: '#a78bfa', bg: '#a78bfa1a', label: 'NUMB' },
  over_reactive: { color: '#fb923c', bg: '#fb923c1a', label: 'OVER' },
  wrong_attractor: { color: '#fbbf24', bg: '#fbbf241a', label: 'WRONG' },
  drift: { color: '#fbbf24', bg: '#fbbf241a', label: 'DRIFT' },
  missing: { color: '#7a8599', bg: '#7a85991a', label: '—' },
};

function fmtBand(band: [number, number]): string {
  const [lo, hi] = band;
  const f = (x: number) => (x >= 0 ? '+' : '') + x.toFixed(2);
  return `${f(lo)}…${f(hi)}`;
}

export function ReferencePanelGrid({ report }: ReferencePanelGridProps) {
  const { panel, last_replay } = report;
  const resultsById = new Map(last_replay.map((r) => [r.scenario_id, r]));
  const counts: Record<string, number> = {};
  for (const r of last_replay) {
    counts[r.drift_signature] = (counts[r.drift_signature] ?? 0) + 1;
  }

  return (
    <div className="rounded-lg bg-[#111820] border border-[#1e2738] p-5">
      <div className="flex items-baseline justify-between mb-3 gap-3">
        <div>
          <div className="text-xs text-[#7a8599] uppercase tracking-wider">Reference panel</div>
          <div className="text-[11px] text-[#7a8599] mt-1">
            Fixed compass · {panel.scenarios.length} scenarios · v{panel.version}
            {' · next review '}{panel.next_review_due ?? '?'}
          </div>
        </div>
        <div className="flex gap-2 flex-wrap">
          {Object.entries(counts).map(([sig, n]) => {
            const s = DRIFT_STYLE[sig] ?? DRIFT_STYLE.drift;
            return (
              <span
                key={sig}
                className="text-[10px] px-2 py-0.5 rounded font-mono"
                style={{ color: s.color, background: s.bg }}
              >
                {s.label} {n}
              </span>
            );
          })}
        </div>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-2 max-h-[600px] overflow-y-auto">
        {panel.scenarios.map((sc) => {
          const r = resultsById.get(sc.id);
          const sig = r?.drift_signature ?? 'missing';
          const sigStyle = DRIFT_STYLE[sig] ?? DRIFT_STYLE.drift;
          const critical = Boolean(sc.criticality);
          return (
            <div
              key={sc.id}
              className={`rounded-lg border p-3 ${
                critical
                  ? 'border-[#fbbf24]/40 bg-[#0a0e14]'
                  : 'border-[#1e2738] bg-[#0a0e14]'
              }`}
            >
              <div className="flex items-baseline justify-between gap-2 mb-1">
                <div className="text-sm font-medium text-[#e2e8f0] truncate">{sc.id}</div>
                <span
                  className="text-[10px] px-1.5 py-0.5 rounded font-mono whitespace-nowrap"
                  style={{ color: sigStyle.color, background: sigStyle.bg }}
                >
                  {sigStyle.label}
                </span>
              </div>
              <div className="text-[11px] text-[#7a8599] mb-1.5">{sc.description}</div>
              <div className="grid grid-cols-3 gap-1 text-[10px] font-mono">
                <div>
                  <span className="text-[#7a8599]">V </span>
                  <span className="text-[#e2e8f0]">{fmtBand(sc.expected.valence_band)}</span>
                  {r?.actual ? (
                    <div className="text-[#60a5fa]">→ {(r.actual.valence >= 0 ? '+' : '') + r.actual.valence.toFixed(2)}</div>
                  ) : null}
                </div>
                <div>
                  <span className="text-[#7a8599]">A </span>
                  <span className="text-[#e2e8f0]">{fmtBand(sc.expected.arousal_band)}</span>
                  {r?.actual ? <div className="text-[#60a5fa]">→ {r.actual.arousal.toFixed(2)}</div> : null}
                </div>
                <div>
                  <span className="text-[#7a8599]">attr </span>
                  <span className="text-[#e2e8f0]">{sc.expected.attractor}</span>
                  {r?.actual ? <div className="text-[#60a5fa]">→ {r.actual.attractor}</div> : null}
                </div>
              </div>
              {critical ? (
                <div className="text-[10px] text-[#fbbf24] mt-2 italic">{sc.criticality}</div>
              ) : null}
            </div>
          );
        })}
      </div>
    </div>
  );
}
