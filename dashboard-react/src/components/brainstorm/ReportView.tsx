import ReactMarkdown from 'react-markdown';
import type { BrainstormSessionPayload } from '../../types/brainstorm';

interface Props {
  session: BrainstormSessionPayload;
  onClose: () => void;
}

export function ReportView({ session, onClose }: Props) {
  const md = session.final_report ?? '';
  return (
    <div className="space-y-3">
      <div className="flex items-start justify-between gap-3 pb-3 border-b border-[#1e2738]">
        <div className="min-w-0">
          <h2 className="text-lg font-semibold text-[#e2e8f0]">
            Final report — {session.technique_title}
          </h2>
          <div className="text-sm text-[#7a8599] truncate">{session.topic}</div>
          {session.final_report_path && (
            <div className="text-[11px] text-[#7a8599] mt-1 font-mono">
              {session.final_report_path}
            </div>
          )}
        </div>
        <button
          type="button"
          onClick={onClose}
          className="text-xs px-3 py-1.5 rounded-md bg-[#0c1118] border border-[#1e2738] text-[#7a8599] hover:text-[#e2e8f0] hover:border-[#3a4a5f] flex-shrink-0"
        >
          Close
        </button>
      </div>
      {md ? (
        <article
          className="prose prose-invert prose-sm max-w-none rounded-md bg-[#0c1118] border border-[#1e2738] p-4 text-[#e2e8f0] [&_h1]:text-[#e2e8f0] [&_h2]:text-[#e2e8f0] [&_h3]:text-[#e2e8f0] [&_strong]:text-[#e2e8f0] [&_a]:text-[#60a5fa] [&_code]:text-[#a78bfa] [&_blockquote]:text-[#7a8599] [&_blockquote]:border-l-[#1e2738]"
        >
          <ReactMarkdown>{md}</ReactMarkdown>
        </article>
      ) : (
        <div className="text-sm text-[#7a8599] italic">
          No report was produced for this session.
        </div>
      )}
    </div>
  );
}
