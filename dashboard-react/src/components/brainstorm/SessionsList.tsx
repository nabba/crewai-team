import { useDeleteMutation, useResumeMutation } from '../../api/brainstorm';
import type { BrainstormSessionPayload } from '../../types/brainstorm';

interface Props {
  sessions: BrainstormSessionPayload[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}

const STATUS_PALETTE: Record<string, string> = {
  active: 'bg-[#60a5fa]/15 text-[#60a5fa] border-[#60a5fa]/30',
  paused: 'bg-[#fbbf24]/15 text-[#fbbf24] border-[#fbbf24]/30',
  complete: 'bg-[#4ade80]/15 text-[#4ade80] border-[#4ade80]/30',
  cancelled: 'bg-[#7a8599]/15 text-[#7a8599] border-[#7a8599]/30',
};

export function SessionsList({ sessions, selectedId, onSelect }: Props) {
  const resumeMut = useResumeMutation();
  const deleteMut = useDeleteMutation();

  if (!sessions.length) {
    return (
      <div className="text-sm text-[#7a8599] italic">
        No past sessions yet.
      </div>
    );
  }

  return (
    <div className="space-y-1.5">
      {sessions.map((s) => {
        const palette = STATUS_PALETTE[s.status] ?? STATUS_PALETTE.cancelled;
        const isSelected = s.session_id === selectedId;
        const stepFraction =
          s.total_steps !== null
            ? `${s.responses.length}/${s.total_steps}`
            : `${s.responses.length}`;
        return (
          <div
            key={s.session_id}
            className={`rounded-md border p-2 cursor-pointer transition-colors ${
              isSelected
                ? 'bg-[#60a5fa]/10 border-[#60a5fa]/40'
                : 'bg-[#0c1118] border-[#1e2738] hover:border-[#3a4a5f]'
            }`}
            onClick={() => onSelect(s.session_id)}
          >
            <div className="flex items-start justify-between gap-2">
              <div className="min-w-0 flex-1">
                <div className="text-sm text-[#e2e8f0] font-medium truncate">
                  {s.topic || '(no topic)'}
                </div>
                <div className="text-[11px] text-[#7a8599] mt-0.5 flex items-center gap-1.5 flex-wrap">
                  <span>{s.technique}</span>
                  <span>·</span>
                  <span>{stepFraction}</span>
                  {s.mode === 'team' && (
                    <>
                      <span>·</span>
                      <span>+{s.participants.length}</span>
                    </>
                  )}
                </div>
              </div>
              <span
                className={`text-[10px] px-1.5 py-0.5 rounded border whitespace-nowrap ${palette}`}
              >
                {s.status}
              </span>
            </div>
            {isSelected && (
              <div className="mt-2 pt-2 border-t border-[#1e2738] flex items-center gap-1">
                {s.status === 'paused' && (
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      resumeMut.mutate({ sessionId: s.session_id });
                    }}
                    disabled={resumeMut.isPending}
                    className="text-[11px] px-2 py-0.5 rounded bg-[#60a5fa]/10 text-[#60a5fa] border border-[#60a5fa]/20 hover:bg-[#60a5fa]/20 disabled:opacity-50"
                  >
                    Resume
                  </button>
                )}
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    if (
                      window.confirm('Delete this session permanently?')
                    ) {
                      deleteMut.mutate({ sessionId: s.session_id });
                    }
                  }}
                  className="text-[11px] px-2 py-0.5 rounded bg-[#0c1118] text-[#7a8599] border border-[#1e2738] hover:text-[#f87171] hover:border-[#f87171]/40"
                >
                  Delete
                </button>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
