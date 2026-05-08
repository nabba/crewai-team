import { useEffect, useState } from 'react';
import type { AffectState, ViabilityFrame, WelfareBreach } from '../../types/affect';

interface AffectStatusStripProps {
  affect: AffectState | null;
  viability: ViabilityFrame | null;
  gateRaised: boolean;
  gateComposite: number;
  recentBreaches: WelfareBreach[];
  lastUpdatedTs: string | null;
  isFetching: boolean;
}

const ATTRACTOR_COLORS: Record<string, string> = {
  peace: '#34d399', contentment: '#34d399', oneness: '#a5f3fc',
  excitement: '#fbbf24', urgency: '#fbbf24', hunger: '#fb923c',
  separation: '#fb923c', distress: '#f87171', discouragement: '#a78bfa',
  depletion: '#a78bfa', boredom: '#7a8599', overwhelm: '#f472b6', neutral: '#7a8599',
};

function ageString(ts: string | null): string {
  if (!ts) return '—';
  try {
    const ms = Date.now() - new Date(ts).getTime();
    if (ms < 0) return 'just now';
    const s = Math.floor(ms / 1000);
    if (s < 60) return `${s}s ago`;
    const m = Math.floor(s / 60);
    if (m < 60) return `${m}m ago`;
    const h = Math.floor(m / 60);
    if (h < 24) return `${h}h ago`;
    return `${Math.floor(h / 24)}d ago`;
  } catch {
    return '—';
  }
}

function MiniBar({ label, value, lo, hi, color }: { label: string; value: number; lo: number; hi: number; color: string }) {
  const pct = ((value - lo) / (hi - lo)) * 100;
  return (
    <div className="flex items-center gap-2">
      <span className="text-[10px] uppercase tracking-wider text-[#7a8599] w-3">{label}</span>
      <div className="w-16 h-1 rounded-full bg-[#1e2738] overflow-hidden">
        <div className="h-full transition-all duration-500" style={{ width: `${Math.max(0, Math.min(100, pct))}%`, background: color }} />
      </div>
      <span className="text-[11px] font-mono text-[#e2e8f0] w-10 text-right">
        {value >= 0 && lo < 0 ? '+' : ''}{value.toFixed(2)}
      </span>
    </div>
  );
}

export function AffectStatusStrip({
  affect,
  viability,
  gateRaised,
  gateComposite,
  recentBreaches,
  lastUpdatedTs,
  isFetching,
}: AffectStatusStripProps) {
  // Tick once a second to update the "ago" string.
  const [, setTick] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setTick((t) => t + 1), 1000);
    return () => clearInterval(id);
  }, []);

  const attractorColor = affect ? (ATTRACTOR_COLORS[affect.attractor] ?? '#7a8599') : '#7a8599';
  const valenceColor = affect
    ? affect.valence >= 0.3 ? '#34d399'
    : affect.valence >= 0 ? '#a5f3fc'
    : affect.valence >= -0.3 ? '#fbbf24'
    : '#f87171'
    : '#7a8599';
  const arousalColor = affect && affect.arousal >= 0.7 ? '#f87171' : affect && affect.arousal >= 0.45 ? '#fbbf24' : '#60a5fa';
  const breachCount = recentBreaches.length;
  const criticalBreaches = recentBreaches.filter((b) => b.severity === 'critical').length;

  return (
    <div className="sticky top-0 z-10 -mx-4 lg:-mx-6 px-4 lg:px-6 py-2.5 bg-[#0a0e14]/95 backdrop-blur border-b border-[#1e2738]">
      <div className="flex items-center justify-between flex-wrap gap-y-2 gap-x-6">
        {/* Attractor + E_t */}
        <div className="flex items-center gap-3 min-w-[12rem]">
          <span
            className="text-base font-semibold whitespace-nowrap"
            style={{ color: attractorColor }}
          >
            {affect?.attractor ?? '—'}
          </span>
          <span className="text-[10px] text-[#7a8599] whitespace-nowrap">
            E_t <span className="text-[#e2e8f0] font-mono">{viability ? viability.total_error.toFixed(3) : '—'}</span>
          </span>
        </div>

        {/* V/A/C mini bars */}
        <div className="flex items-center gap-3 flex-wrap">
          <MiniBar label="V" value={affect?.valence ?? 0} lo={-1} hi={1} color={valenceColor} />
          <MiniBar label="A" value={affect?.arousal ?? 0} lo={0} hi={1} color={arousalColor} />
          <MiniBar label="C" value={affect?.controllability ?? 0.5} lo={0} hi={1} color="#60a5fa" />
        </div>

        {/* Welfare + gate */}
        <div className="flex items-center gap-2 flex-wrap">
          {criticalBreaches > 0 ? (
            <span className="text-[10px] px-1.5 py-0.5 rounded font-mono text-[#f87171] bg-[#f871711a]">
              CRIT {criticalBreaches}
            </span>
          ) : breachCount > 0 ? (
            <span className="text-[10px] px-1.5 py-0.5 rounded font-mono text-[#fbbf24] bg-[#fbbf241a]">
              WARN {breachCount}
            </span>
          ) : (
            <span className="text-[10px] px-1.5 py-0.5 rounded font-mono text-[#34d399] bg-[#34d3991a]">
              welfare ok
            </span>
          )}
          <span
            className="text-[10px] px-1.5 py-0.5 rounded font-mono whitespace-nowrap"
            style={{
              color: gateRaised ? '#fb923c' : '#34d399',
              background: gateRaised ? '#fb923c1a' : '#34d3991a',
            }}
          >
            gate {gateRaised ? 'raised' : 'clear'} · {gateComposite.toFixed(2)}
          </span>
        </div>

        {/* Last-updated */}
        <div className="flex items-center gap-2 text-[10px] text-[#7a8599] whitespace-nowrap">
          <span
            className={`inline-block w-1.5 h-1.5 rounded-full ${
              isFetching ? 'bg-[#34d399] animate-pulse' : 'bg-[#7a8599]'
            }`}
          />
          <span>{ageString(lastUpdatedTs)}</span>
        </div>
      </div>
    </div>
  );
}
