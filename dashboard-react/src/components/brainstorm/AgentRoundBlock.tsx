import type { AgentResponsePayload } from '../../types/brainstorm';
import { AgentResponseCard } from './AgentResponseCard';

interface Props {
  label: string;
  responses: AgentResponsePayload[];
}

export function AgentRoundBlock({ label, responses }: Props) {
  if (!responses.length) return null;
  return (
    <div className="space-y-2">
      <div className="text-[11px] uppercase tracking-wider text-[#7a8599] font-semibold">
        {label}
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
        {responses.map((r, i) => (
          <AgentResponseCard key={`${r.role}-${i}`} resp={r} />
        ))}
      </div>
    </div>
  );
}
