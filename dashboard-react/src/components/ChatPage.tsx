import { useEffect, useMemo, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeHighlight from 'rehype-highlight';
import 'katex/dist/katex.min.css';
import 'highlight.js/styles/github-dark.css';
import {
  useChatMessagesQuery,
  useChatSend,
  useSignalCommandsQuery,
  type ChatMessage,
  type SignalCommandEntry,
} from '../api/queries';
import { Skeleton } from './ui/Skeleton';
import { ErrorPanel } from './ui/ErrorPanel';

const SENDER = 'andrus';
const MESSAGE_LIMIT = 100;

// Mirror of the Signal chat — read history, send a message, render
// every reply through the same Commander.handle path. Markdown +
// GFM + math + code highlighting, same renderer as the Notes view.

export function ChatPage() {
  const historyQ = useChatMessagesQuery(SENDER, MESSAGE_LIMIT);
  const send = useChatSend();
  const [draft, setDraft] = useState('');
  const [showCommands, setShowCommands] = useState(true);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  // Auto-scroll to the bottom on new history.
  const messages = historyQ.data?.messages ?? [];
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [messages.length, send.isPending]);

  const handleSend = async () => {
    const text = draft.trim();
    if (!text || send.isPending) return;
    setDraft('');
    try {
      await send.mutateAsync({ sender: SENDER, message: text });
    } catch {
      // error surfaces via send.error
    }
  };

  const onKey = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex flex-col lg:flex-row gap-4 h-[calc(100vh-9rem)]">
      <div className="flex-1 flex flex-col min-w-0">
        <header className="mb-2">
          <h1 className="text-xl font-semibold text-[#e2e8f0]">Chat</h1>
          <p className="text-xs text-[#7a8599]">
            Same dispatch path as Signal — slash commands, recovery loop,
            project routing, lifecycle hooks all fire identically. Replies
            render full markdown.
          </p>
        </header>

        <div
          ref={scrollRef}
          className="flex-1 overflow-y-auto rounded-lg bg-[#0a0e14] border border-[#1e2738] p-4 space-y-3"
        >
          {historyQ.isLoading ? (
            <div className="space-y-2">
              <Skeleton className="h-12" />
              <Skeleton className="h-16" />
              <Skeleton className="h-12" />
            </div>
          ) : historyQ.error ? (
            <ErrorPanel error={historyQ.error} onRetry={historyQ.refetch} />
          ) : historyQ.data?.error ? (
            <p className="text-xs text-[#f87171]">{historyQ.data.error}</p>
          ) : messages.length === 0 ? (
            <p className="text-sm text-[#7a8599] italic">
              No messages yet. Try <code>/help</code> or <code>status</code>.
            </p>
          ) : (
            messages.map((m, i) => <ChatBubble key={`${m.ts}-${i}`} m={m} />)
          )}
          {send.isPending && (
            <div className="text-xs text-[#7a8599] italic px-3">… working …</div>
          )}
          {send.error instanceof Error && (
            <div className="text-xs text-[#f87171] px-3">
              Send failed: {send.error.message}
            </div>
          )}
        </div>

        <div className="mt-3 flex gap-2">
          <textarea
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={onKey}
            placeholder="Type a message — Enter to send, Shift+Enter for newline. /help for commands."
            rows={2}
            className="flex-1 bg-[#0a0e14] border border-[#1e2738] rounded-lg px-3 py-2 text-sm text-[#e2e8f0] placeholder-[#7a8599] focus:outline-none focus:border-[#60a5fa] resize-y"
          />
          <button
            onClick={handleSend}
            disabled={!draft.trim() || send.isPending}
            className="self-stretch px-4 py-2 bg-[#60a5fa]/20 border border-[#60a5fa]/30 text-[#60a5fa] text-sm font-medium rounded-lg hover:bg-[#60a5fa]/30 disabled:opacity-50 transition-colors"
          >
            {send.isPending ? 'Sending…' : 'Send'}
          </button>
        </div>

        <div className="mt-2 flex items-center justify-between text-[11px] text-[#7a8599]">
          <span>sender: {SENDER}</span>
          <button
            onClick={() => setShowCommands((v) => !v)}
            className="lg:hidden underline"
          >
            {showCommands ? 'Hide' : 'Show'} command catalogue
          </button>
        </div>
      </div>

      {showCommands && (
        <aside className="w-full lg:w-[22rem] shrink-0">
          <CommandCatalogue
            onPick={(syntax) => setDraft((prev) => (prev ? prev + ' ' : '') + syntax)}
          />
        </aside>
      )}
    </div>
  );
}

function ChatBubble({ m }: { m: ChatMessage }) {
  const isUser = m.role === 'user';
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={
          isUser
            ? 'max-w-[80%] rounded-lg px-3 py-2 bg-[#60a5fa]/10 border border-[#60a5fa]/30'
            : 'max-w-[88%] rounded-lg px-3 py-2 bg-[#111820] border border-[#1e2738]'
        }
      >
        <div className="flex items-baseline gap-2 mb-1">
          <span className="text-[10px] uppercase tracking-wider text-[#7a8599]">
            {isUser ? 'you' : m.role || 'assistant'}
          </span>
          <span className="text-[10px] text-[#7a8599]/70">
            {new Date(m.ts * 1000).toLocaleString()}
          </span>
        </div>
        {isUser ? (
          <pre className="text-sm text-[#e2e8f0] whitespace-pre-wrap font-sans">
            {m.content}
          </pre>
        ) : (
          <MarkdownReply content={m.content} />
        )}
      </div>
    </div>
  );
}

function MarkdownReply({ content }: { content: string }) {
  return (
    <div className="prose prose-invert prose-sm max-w-none prose-pre:bg-black/30 prose-pre:border prose-pre:border-[#1e2738] prose-code:text-[#a5f3fc] prose-code:before:content-none prose-code:after:content-none prose-headings:text-[#e2e8f0] prose-strong:text-[#e2e8f0] prose-a:text-[#60a5fa]">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex, rehypeHighlight]}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}

function CommandCatalogue({ onPick }: { onPick: (syntax: string) => void }) {
  const q = useSignalCommandsQuery();
  const [filter, setFilter] = useState('');

  const filtered = useMemo(() => {
    const cmds = q.data?.commands ?? [];
    if (!filter.trim()) return cmds;
    const f = filter.toLowerCase();
    return cmds.filter(
      (c) =>
        c.command.toLowerCase().includes(f) ||
        c.syntax.toLowerCase().includes(f) ||
        c.description.toLowerCase().includes(f) ||
        c.aliases.some((a) => a.toLowerCase().includes(f)),
    );
  }, [q.data?.commands, filter]);

  const grouped = useMemo(() => {
    const m = new Map<string, SignalCommandEntry[]>();
    for (const c of filtered) {
      const arr = m.get(c.category) ?? [];
      arr.push(c);
      m.set(c.category, arr);
    }
    // Preserve catalogue category order from the backend.
    const order = q.data?.categories ?? [];
    return order
      .filter((cat) => m.has(cat))
      .map((cat) => ({ category: cat, items: m.get(cat) ?? [] }));
  }, [filtered, q.data?.categories]);

  return (
    <div className="h-full flex flex-col rounded-lg bg-[#0a0e14] border border-[#1e2738] p-3">
      <div className="flex items-baseline justify-between mb-2 gap-2">
        <h2 className="text-sm font-medium text-[#e2e8f0]">Commands</h2>
        <span className="text-[10px] text-[#7a8599]">
          {q.data?.commands.length ?? 0} total
        </span>
      </div>
      <input
        type="search"
        value={filter}
        onChange={(e) => setFilter(e.target.value)}
        placeholder="Filter…"
        className="mb-2 w-full bg-[#111820] border border-[#1e2738] rounded px-2 py-1 text-xs text-[#e2e8f0] placeholder-[#7a8599] focus:outline-none focus:border-[#60a5fa]"
      />
      <div className="flex-1 overflow-y-auto space-y-3 pr-1">
        {q.isLoading ? (
          <Skeleton className="h-32" />
        ) : q.error ? (
          <ErrorPanel error={q.error} onRetry={q.refetch} />
        ) : grouped.length === 0 ? (
          <p className="text-xs text-[#7a8599] italic">No matches.</p>
        ) : (
          grouped.map(({ category, items }) => (
            <section key={category}>
              <div className="text-[10px] uppercase tracking-wider text-[#7a8599] mb-1">
                {category}
              </div>
              <ul className="space-y-1.5">
                {items.map((c) => (
                  <li
                    key={`${c.category}:${c.command}`}
                    className="rounded border border-[#1e2738] px-2 py-1.5 hover:border-[#60a5fa]/40 cursor-pointer"
                    onClick={() => onPick(c.syntax)}
                    title="Click to insert into the message box"
                  >
                    <code className="text-[11px] text-[#a5f3fc] block">
                      {c.syntax}
                    </code>
                    {c.aliases.length > 0 && (
                      <div className="text-[10px] text-[#7a8599]">
                        aliases: {c.aliases.join(', ')}
                      </div>
                    )}
                    <p className="text-[11px] text-[#cbd5e1] mt-0.5">
                      {c.description}
                    </p>
                  </li>
                ))}
              </ul>
            </section>
          ))
        )}
      </div>
    </div>
  );
}
