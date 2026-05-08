import { useState } from 'react';
import { ErrorPanel } from './ui/ErrorPanel';
import {
  useFilesQuery,
  useSendFile,
  type FileEntry,
  type FilesRoot,
  type SendChannel,
} from '../api/queries';
import { endpoints } from '../api/endpoints';

const ROOT_LABELS: Record<FilesRoot, { label: string; description: string }> = {
  output: {
    label: 'Generated documents',
    description: 'PDFs, Word docs, spreadsheets, slide decks, and HTML pages produced by document_generator and pdf_compose.',
  },
  skills: {
    label: 'Skill markdown',
    description: 'Markdown skill files written by the self-improvement crew under workspace/skills/.',
  },
  notes: {
    label: 'Notes',
    description: 'Obsidian-style markdown notes under workspace/notes/.',
  },
};

export function FilesPage() {
  const filesQ = useFilesQuery();
  const [filter, setFilter] = useState('');

  if (filesQ.isLoading) {
    return (
      <div className="bg-[#111820] border border-[#1e2738] rounded-xl p-4">
        <div className="text-[#7a8599] text-sm">Loading files…</div>
      </div>
    );
  }
  if (filesQ.error) return <ErrorPanel error={filesQ.error} onRetry={filesQ.refetch} />;
  const data = filesQ.data;
  if (!data) return null;

  const trimmed = filter.trim().toLowerCase();
  const filterFn = (e: FileEntry) =>
    !trimmed || e.name.toLowerCase().includes(trimmed) || e.path.toLowerCase().includes(trimmed);

  return (
    <div className="space-y-4 max-w-4xl">
      <div>
        <h1 className="text-xl font-semibold text-[#e2e8f0]">Files</h1>
        <p className="text-xs text-[#7a8599] mt-1">
          Download generated artifacts directly, or deliver them via Signal,
          email, or Discord. All paths are sandboxed under{' '}
          <code className="text-[#60a5fa]">workspace/</code>.
        </p>
      </div>

      <input
        type="search"
        placeholder="Filter by name…"
        value={filter}
        onChange={(e) => setFilter(e.target.value)}
        className="w-full bg-[#0a0f18] border border-[#1e2738] rounded px-3 py-2 text-[#e2e8f0] text-sm"
      />

      {(Object.keys(ROOT_LABELS) as FilesRoot[]).map((root) => {
        const meta = ROOT_LABELS[root];
        const entries = (data.roots[root] ?? []).filter(filterFn);
        return (
          <RootSection
            key={root}
            label={meta.label}
            description={meta.description}
            entries={entries}
            empty={(data.roots[root]?.length ?? 0) === 0
              ? 'No files yet — agents will populate this as they generate.'
              : 'No matches for the current filter.'}
          />
        );
      })}
    </div>
  );
}

function RootSection({
  label, description, entries, empty,
}: {
  label: string;
  description: string;
  entries: FileEntry[];
  empty: string;
}) {
  return (
    <div className="bg-[#111820] border border-[#1e2738] rounded-xl p-4 space-y-3">
      <div>
        <h2 className="text-base font-semibold text-[#e2e8f0]">{label}</h2>
        <p className="text-xs text-[#7a8599] mt-1">{description}</p>
      </div>
      {entries.length === 0 ? (
        <div className="text-xs text-[#7a8599] italic">{empty}</div>
      ) : (
        <ul className="divide-y divide-[#1e2738]">
          {entries.map((e) => <FileRow key={e.path} entry={e} />)}
        </ul>
      )}
    </div>
  );
}

function FileRow({ entry }: { entry: FileEntry }) {
  return (
    <li className="py-2 flex flex-col sm:flex-row sm:items-center gap-2 sm:gap-3">
      <div className="min-w-0 flex-1">
        <div className="text-sm text-[#e2e8f0] truncate" title={entry.path}>
          {entry.name}
        </div>
        <div className="text-[11px] text-[#7a8599]">
          {humanSize(entry.size)} · {entry.extension.toUpperCase() || 'FILE'} ·{' '}
          {new Date(entry.modified).toLocaleString()}
        </div>
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <a
          href={endpoints.fileDownload(entry.path)}
          download={entry.name}
          className="px-2 py-1 bg-[#2563eb] hover:bg-[#1d4ed8] rounded text-white text-xs"
        >
          Download
        </a>
        <SendButton path={entry.path} />
      </div>
    </li>
  );
}

function SendButton({ path }: { path: string }) {
  const [open, setOpen] = useState(false);
  const [channel, setChannel] = useState<SendChannel>('signal');
  const [emailTo, setEmailTo] = useState('');
  const [emailSubject, setEmailSubject] = useState('');
  const [body, setBody] = useState('');
  const [feedback, setFeedback] = useState('');
  const send = useSendFile();

  const error = send.error instanceof Error ? send.error.message : '';

  const submit = async () => {
    setFeedback('');
    try {
      const res = await send.mutateAsync({
        channel,
        path,
        body: body || undefined,
        to: channel === 'email' ? emailTo : undefined,
        subject: channel === 'email' ? (emailSubject || undefined) : undefined,
      });
      setFeedback(`✓ ${res.detail}`);
      setTimeout(() => {
        setFeedback('');
        setOpen(false);
      }, 2500);
    } catch {/* surfaced via send.error */}
  };

  return (
    <div className="relative inline-flex">
      <button
        onClick={() => setOpen((v) => !v)}
        className="px-2 py-1 bg-[#0a0f18] border border-[#1e2738] hover:border-[#3b4659] rounded text-[#7a8599] hover:text-[#e2e8f0] text-xs"
      >
        Send…
      </button>
      {open && (
        <div className="absolute z-10 right-0 top-full mt-1 w-72 bg-[#0a0f18] border border-[#1e2738] rounded-lg p-3 shadow-xl space-y-2">
          <div className="flex gap-1 text-xs">
            {(['signal', 'email', 'discord'] as SendChannel[]).map((c) => (
              <button
                key={c}
                onClick={() => setChannel(c)}
                className={`flex-1 px-2 py-1 rounded ${
                  channel === c
                    ? 'bg-[#60a5fa]/10 border border-[#60a5fa]/40 text-[#e2e8f0]'
                    : 'border border-[#1e2738] text-[#7a8599] hover:text-[#e2e8f0]'
                }`}
              >
                {c}
              </button>
            ))}
          </div>

          {channel === 'email' && (
            <>
              <input
                type="email"
                value={emailTo}
                onChange={(e) => setEmailTo(e.target.value)}
                placeholder="recipient@example.com"
                className="w-full bg-[#111820] border border-[#1e2738] rounded px-2 py-1 text-[#e2e8f0] text-xs"
              />
              <input
                type="text"
                value={emailSubject}
                onChange={(e) => setEmailSubject(e.target.value)}
                placeholder="Subject (optional)"
                className="w-full bg-[#111820] border border-[#1e2738] rounded px-2 py-1 text-[#e2e8f0] text-xs"
              />
            </>
          )}

          <textarea
            value={body}
            onChange={(e) => setBody(e.target.value)}
            placeholder={
              channel === 'signal'
                ? '1-3 sentence message (required by signal)'
                : 'Optional message'
            }
            rows={2}
            className="w-full bg-[#111820] border border-[#1e2738] rounded px-2 py-1 text-[#e2e8f0] text-xs resize-y"
          />

          <div className="flex items-center gap-2">
            <button
              onClick={submit}
              disabled={send.isPending || (channel === 'email' && !emailTo.trim())}
              className="flex-1 px-2 py-1 bg-[#2563eb] hover:bg-[#1d4ed8] disabled:opacity-50 rounded text-white text-xs"
            >
              {send.isPending ? 'Sending…' : `Send via ${channel}`}
            </button>
            <button
              onClick={() => setOpen(false)}
              className="px-2 py-1 border border-[#1e2738] text-[#7a8599] hover:text-[#e2e8f0] rounded text-xs"
            >
              Cancel
            </button>
          </div>

          {error && <div className="text-[#f87171] text-xs">{error}</div>}
          {feedback && <div className="text-[#34d399] text-xs">{feedback}</div>}
        </div>
      )}
    </div>
  );
}

function humanSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}
