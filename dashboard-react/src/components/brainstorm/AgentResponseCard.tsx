import type { AgentResponsePayload } from '../../types/brainstorm';

interface Props {
  resp: AgentResponsePayload;
}

const ROLE_COLORS: Record<string, { bg: string; border: string; text: string }> = {
  researcher: { bg: 'bg-[#1e3a5f]/40', border: 'border-[#60a5fa]/40', text: 'text-[#60a5fa]' },
  writer: { bg: 'bg-[#3a2f5e]/40', border: 'border-[#a78bfa]/40', text: 'text-[#a78bfa]' },
  coder: { bg: 'bg-[#2f4a3a]/40', border: 'border-[#4ade80]/40', text: 'text-[#4ade80]' },
  critic: { bg: 'bg-[#5e2f2f]/40', border: 'border-[#f87171]/40', text: 'text-[#f87171]' },
};

const FALLBACK = { bg: 'bg-[#1e2738]', border: 'border-[#7a8599]/40', text: 'text-[#e2e8f0]' };

export function AgentResponseCard({ resp }: Props) {
  const palette = ROLE_COLORS[resp.role] ?? FALLBACK;
  return (
    <div
      className={`rounded-md border ${palette.border} ${palette.bg} p-3 text-sm`}
    >
      <div className={`font-semibold mb-1 ${palette.text}`}>
        {resp.role}
        {resp.duration_s > 0 && (
          <span className="ml-2 text-[10px] text-[#7a8599] font-normal">
            {resp.duration_s.toFixed(1)}s
          </span>
        )}
      </div>
      {resp.error ? (
        <div className="text-[#f87171] text-xs italic">{resp.error}</div>
      ) : resp.text.trim() ? (
        <div className="text-[#e2e8f0] whitespace-pre-wrap leading-snug">
          {resp.text}
        </div>
      ) : (
        <div className="text-[#7a8599] italic text-xs">(no response)</div>
      )}
    </div>
  );
}
