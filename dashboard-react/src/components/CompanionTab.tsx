// Companion Layer dashboard — sub-tab inside /cp/ops.
// Wraps every backend endpoint shipped across Phases 0–13.

import { useEffect, useState } from 'react';
import {
  fetchWikiPageBody,
  useAcceptGrandTask,
  useAcceptXWorkspaceKernel,
  useAddSource,
  useCompanionConfig,
  useCompanionIdeas,
  useCompanionSources,
  useDismissXWorkspaceKernel,
  useGrandTaskProposals,
  usePromoteIdea,
  useRejectGrandTask,
  useRemoveSource,
  useSourceSuggestions,
  useThumbsFeedback,
  useUpdateCompanionConfig,
  useWikiPages,
  useXWorkspaceInbox,
  type CompanionConfig,
  type CompanionConfigPatch,
  type IdeaSummary,
} from '../api/companion';
import { api } from '../api/client';
import { TensionsCard } from './TensionsCard';

// Workspace selector — pulls from the existing /api/workspaces (Phase 0).
interface WorkspaceListItem {
  project_id: string;
  display_name: string;
}

function useWorkspaces() {
  // Re-uses the already-shipped /api/workspaces endpoint. Pulls every CP
  // project the gateway knows about so the user can pick one.
  const [data, setData] = useState<WorkspaceListItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let cancelled = false;
    api<{ workspaces: WorkspaceListItem[] }>('/api/workspaces')
      .then((r) => {
        if (!cancelled) setData(r.workspaces ?? []);
      })
      .catch((e) => {
        if (!cancelled) setError(String(e));
      });
    return () => {
      cancelled = true;
    };
  }, []);
  return { data, error };
}

type SubTab =
  | 'live'
  | 'ideas'
  | 'tensions'
  | 'documents'
  | 'wiki'
  | 'sources'
  | 'grand_task'
  | 'inbox'
  | 'settings';

const SUB_TABS: { key: SubTab; label: string; icon: string }[] = [
  { key: 'live', label: 'Live', icon: '🌀' },
  { key: 'ideas', label: 'Ideas', icon: '💡' },
  { key: 'tensions', label: 'Tensions', icon: '❓' },
  { key: 'documents', label: 'Documents', icon: '📄' },
  { key: 'wiki', label: 'Wiki', icon: '📚' },
  { key: 'sources', label: 'Sources', icon: '🔍' },
  { key: 'grand_task', label: 'Grand task', icon: '🎯' },
  { key: 'inbox', label: 'Inbox', icon: '📨' },
  { key: 'settings', label: 'Settings', icon: '⚙️' },
];

// ── Page ───────────────────────────────────────────────────────────────────

export function CompanionTab() {
  const { data: workspaces, error: wsError } = useWorkspaces();
  const [workspaceId, setWorkspaceId] = useState<string>('');
  const [sub, setSub] = useState<SubTab>('live');

  // Default-select the first workspace once the list loads.
  useEffect(() => {
    if (workspaces && workspaces.length > 0 && !workspaceId) {
      setWorkspaceId(workspaces[0].project_id);
    }
  }, [workspaces, workspaceId]);

  if (wsError) {
    return (
      <div className="text-[#f87171] text-sm">
        Couldn't load workspaces: {wsError}
      </div>
    );
  }
  if (!workspaces) {
    return <div className="text-[#7a8599] text-sm">Loading workspaces…</div>;
  }
  if (workspaces.length === 0) {
    return (
      <div className="text-[#7a8599] text-sm">
        No workspaces yet. Create one via{' '}
        <code className="text-[#e2e8f0]">POST /api/workspaces</code>.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <label className="text-sm text-[#7a8599]">Workspace:</label>
        <select
          value={workspaceId}
          onChange={(e) => setWorkspaceId(e.target.value)}
          className="bg-[#0c1118] border border-[#1e2738] rounded px-3 py-1 text-sm text-[#e2e8f0]"
        >
          {workspaces.map((w) => (
            <option key={w.project_id} value={w.project_id}>
              {w.display_name || w.project_id}
            </option>
          ))}
        </select>
      </div>

      <div className="flex gap-1 bg-[#0c1118] rounded-lg p-1 border border-[#1e2738] w-fit overflow-x-auto">
        {SUB_TABS.map((t) => (
          <button
            key={t.key}
            onClick={() => setSub(t.key)}
            className={`px-3 py-1 rounded-md text-sm transition-colors flex items-center gap-1.5 whitespace-nowrap ${
              sub === t.key
                ? 'bg-[#60a5fa]/15 text-[#60a5fa] font-medium'
                : 'text-[#7a8599] hover:text-[#e2e8f0]'
            }`}
          >
            <span>{t.icon}</span>
            <span>{t.label}</span>
          </button>
        ))}
      </div>

      <div className="bg-[#0c1118] border border-[#1e2738] rounded-lg p-4">
        {sub === 'live' && <LiveTab workspaceId={workspaceId} />}
        {sub === 'ideas' && <IdeasTab workspaceId={workspaceId} />}
        {sub === 'tensions' && <TensionsCard />}
        {sub === 'documents' && <DocumentsTab workspaceId={workspaceId} />}
        {sub === 'wiki' && <WikiTab workspaceId={workspaceId} />}
        {sub === 'sources' && <SourcesTab workspaceId={workspaceId} />}
        {sub === 'grand_task' && <GrandTaskTab workspaceId={workspaceId} />}
        {sub === 'inbox' && <InboxTab workspaceId={workspaceId} />}
        {sub === 'settings' && <SettingsTab workspaceId={workspaceId} />}
      </div>
    </div>
  );
}

// ── Live ───────────────────────────────────────────────────────────────────

function LiveTab({ workspaceId }: { workspaceId: string }) {
  const ideasQ = useCompanionIdeas(workspaceId, 12);
  const cfgQ = useCompanionConfig(workspaceId);
  const ideas = ideasQ.data?.ideas ?? [];
  const cfg = cfgQ.data?.config;

  const surfaced = ideas.filter((i) => i.current_state === 'surfaced');
  const documented = ideas.filter((i) => i.current_state === 'documented');
  const recent = ideas.slice(0, 5);

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
        <Stat label="Seed" value={cfg?.seed_prompt?.slice(0, 24) ?? '(none)'} />
        <Stat label="Budget/day" value={`$${cfg?.daily_budget_usd?.toFixed(2) ?? '—'}`} />
        <Stat label="Surfaced" value={surfaced.length.toString()} />
        <Stat label="Documented" value={documented.length.toString()} />
      </div>
      <div>
        <h3 className="text-sm font-medium text-[#e2e8f0] mb-2">
          Recent ideas
        </h3>
        {ideasQ.isLoading && (
          <div className="text-[#7a8599] text-sm">Loading…</div>
        )}
        {recent.length === 0 && !ideasQ.isLoading && (
          <div className="text-[#7a8599] text-sm italic">
            No ideas yet. The Companion runs during idle windows.
          </div>
        )}
        <div className="space-y-2">
          {recent.map((i) => (
            <IdeaCard key={i.idea_id} workspaceId={workspaceId} idea={i} />
          ))}
        </div>
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-[#111820] border border-[#1e2738] rounded p-2">
      <div className="text-[10px] uppercase tracking-wide text-[#7a8599]">
        {label}
      </div>
      <div className="text-sm text-[#e2e8f0] truncate">{value}</div>
    </div>
  );
}

// ── Ideas ──────────────────────────────────────────────────────────────────

function IdeasTab({ workspaceId }: { workspaceId: string }) {
  const ideasQ = useCompanionIdeas(workspaceId, 100);
  const ideas = ideasQ.data?.ideas ?? [];

  if (ideasQ.isLoading)
    return <div className="text-[#7a8599] text-sm">Loading ideas…</div>;
  if (ideas.length === 0)
    return (
      <div className="text-[#7a8599] text-sm italic">
        No ideas in this workspace yet.
      </div>
    );

  return (
    <div className="space-y-2">
      <div className="text-xs text-[#7a8599]">{ideas.length} ideas</div>
      {ideas.map((i) => (
        <IdeaCard key={i.idea_id} workspaceId={workspaceId} idea={i} />
      ))}
    </div>
  );
}

function IdeaCard({
  workspaceId,
  idea,
}: {
  workspaceId: string;
  idea: IdeaSummary;
}) {
  const thumbs = useThumbsFeedback();
  const promote = usePromoteIdea();
  const stateColor =
    idea.current_state === 'documented'
      ? 'text-[#34d399]'
      : idea.current_state === 'surfaced'
        ? 'text-[#60a5fa]'
        : idea.current_state === 'archived'
          ? 'text-[#7a8599]'
          : 'text-[#fbbf24]';

  return (
    <div className="bg-[#111820] border border-[#1e2738] rounded p-3 space-y-2">
      <div className="flex items-center justify-between gap-2">
        <code className="text-[10px] text-[#7a8599]">
          {idea.idea_id.slice(0, 16)}
        </code>
        <span className={`text-xs ${stateColor}`}>{idea.current_state}</span>
      </div>
      <div className="text-sm text-[#e2e8f0] whitespace-pre-wrap">
        {idea.text}
      </div>
      <div className="flex flex-wrap items-center gap-3 text-[11px] text-[#7a8599]">
        <span>novelty {idea.novelty.toFixed(2)}</span>
        <span>quality {idea.quality.toFixed(2)}</span>
        <span>panel {(idea.panel_score ?? 0).toFixed(2)}</span>
        <span>transferability {idea.transferability.toFixed(2)}</span>
      </div>
      <div className="flex gap-2">
        <button
          onClick={() =>
            thumbs.mutate({
              workspaceId,
              ideaId: idea.idea_id,
              polarity: 'up',
            })
          }
          disabled={thumbs.isPending}
          className="text-sm px-2 py-0.5 bg-[#34d399]/15 text-[#34d399] border border-[#34d399]/40 rounded hover:bg-[#34d399]/25"
        >
          👍
        </button>
        <button
          onClick={() =>
            thumbs.mutate({
              workspaceId,
              ideaId: idea.idea_id,
              polarity: 'down',
            })
          }
          disabled={thumbs.isPending}
          className="text-sm px-2 py-0.5 bg-[#f87171]/15 text-[#f87171] border border-[#f87171]/40 rounded hover:bg-[#f87171]/25"
        >
          👎
        </button>
        <button
          onClick={() =>
            promote.mutate({
              workspaceId,
              ideaId: idea.idea_id,
              formats: ['md', 'docx', 'pdf'],
            })
          }
          disabled={promote.isPending}
          className="text-sm px-2 py-0.5 bg-[#60a5fa]/15 text-[#60a5fa] border border-[#60a5fa]/40 rounded hover:bg-[#60a5fa]/25 ml-auto"
        >
          {promote.isPending ? 'Promoting…' : 'Promote → docs + wiki'}
        </button>
      </div>
    </div>
  );
}

// ── Documents ──────────────────────────────────────────────────────────────

function DocumentsTab({ workspaceId }: { workspaceId: string }) {
  const ideasQ = useCompanionIdeas(workspaceId, 100);
  const ideas = ideasQ.data?.ideas ?? [];
  const documented = ideas.filter((i) => i.current_state === 'documented');

  if (documented.length === 0)
    return (
      <div className="text-[#7a8599] text-sm italic">
        No documented ideas yet. Promote an idea from the Ideas tab.
      </div>
    );

  return (
    <div className="space-y-2">
      {documented.map((i) => (
        <DocumentRow key={i.idea_id} workspaceId={workspaceId} idea={i} />
      ))}
    </div>
  );
}

function DocumentRow({
  workspaceId,
  idea,
}: {
  workspaceId: string;
  idea: IdeaSummary;
}) {
  const formats = ['md', 'docx', 'pdf'];
  return (
    <div className="bg-[#111820] border border-[#1e2738] rounded p-3 flex items-center justify-between gap-2">
      <div className="min-w-0">
        <div className="text-sm text-[#e2e8f0] truncate">
          {idea.text.slice(0, 80)}
        </div>
        <code className="text-[10px] text-[#7a8599]">{idea.idea_id}</code>
      </div>
      <div className="flex gap-2">
        {formats.map((fmt) => (
          <a
            key={fmt}
            href={`/api/cp/companion/document/${encodeURIComponent(workspaceId)}/${encodeURIComponent(idea.idea_id)}/${fmt}`}
            className="text-xs px-2 py-0.5 bg-[#60a5fa]/15 text-[#60a5fa] border border-[#60a5fa]/40 rounded hover:bg-[#60a5fa]/25"
          >
            {fmt}
          </a>
        ))}
      </div>
    </div>
  );
}

// ── Wiki ───────────────────────────────────────────────────────────────────

function WikiTab({ workspaceId }: { workspaceId: string }) {
  const pagesQ = useWikiPages(workspaceId);
  const pages = pagesQ.data?.pages ?? [];
  const [selected, setSelected] = useState<string | null>(null);
  const [body, setBody] = useState<string>('');
  const [bodyError, setBodyError] = useState<string | null>(null);

  useEffect(() => {
    if (!selected) {
      setBody('');
      setBodyError(null);
      return;
    }
    let cancelled = false;
    fetchWikiPageBody(workspaceId, selected)
      .then((t) => {
        if (!cancelled) {
          setBody(t);
          setBodyError(null);
        }
      })
      .catch((e) => {
        if (!cancelled) setBodyError(String(e));
      });
    return () => {
      cancelled = true;
    };
  }, [workspaceId, selected]);

  if (pagesQ.isLoading)
    return <div className="text-[#7a8599] text-sm">Loading wiki…</div>;
  if (pages.length === 0)
    return (
      <div className="text-[#7a8599] text-sm italic">
        No wiki pages yet. Promote an idea to register it in the workspace
        wiki + Mem0 + system wiki.
      </div>
    );

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
      <div className="space-y-1">
        <div className="text-xs text-[#7a8599] mb-1">{pages.length} pages</div>
        {pages.map((p) => (
          <button
            key={p.idea_id}
            onClick={() => setSelected(p.idea_id)}
            className={`w-full text-left px-2 py-1 rounded text-sm truncate ${
              selected === p.idea_id
                ? 'bg-[#60a5fa]/15 text-[#60a5fa]'
                : 'text-[#e2e8f0] hover:bg-[#1e2738]'
            }`}
          >
            {p.title || p.filename}
          </button>
        ))}
      </div>
      <div className="md:col-span-2 bg-[#0a0e14] border border-[#1e2738] rounded p-3 min-h-[300px]">
        {selected ? (
          bodyError ? (
            <div className="text-[#f87171] text-sm">{bodyError}</div>
          ) : (
            <pre className="text-xs text-[#e2e8f0] whitespace-pre-wrap font-mono">
              {body}
            </pre>
          )
        ) : (
          <div className="text-[#7a8599] text-sm italic">
            Select a page to view its markdown body.
          </div>
        )}
      </div>
    </div>
  );
}

// ── Sources ────────────────────────────────────────────────────────────────

function SourcesTab({ workspaceId }: { workspaceId: string }) {
  const sourcesQ = useCompanionSources(workspaceId);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const suggestionsQ = useSourceSuggestions(workspaceId, showSuggestions);
  const addSrc = useAddSource();
  const removeSrc = useRemoveSource();
  const [newQuery, setNewQuery] = useState('');

  const sources = sourcesQ.data?.sources ?? [];
  const suggestions = suggestionsQ.data?.suggestions ?? [];

  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-sm font-medium text-[#e2e8f0] mb-2">Sources</h3>
        {sources.length === 0 && (
          <div className="text-[#7a8599] text-sm italic">
            No sources yet. Auto-suggest a few or add manually below.
          </div>
        )}
        <div className="space-y-1">
          {sources.map((s) => (
            <div
              key={s.source_id}
              className="flex items-center justify-between gap-2 bg-[#111820] border border-[#1e2738] rounded p-2"
            >
              <div className="min-w-0 flex-1">
                <div className="text-sm text-[#e2e8f0] truncate">
                  {s.type}: {String((s.config as Record<string, unknown>)?.query ?? '')}
                </div>
                <div className="text-[10px] text-[#7a8599]">
                  status: {s.last_ingest_status || 'never ingested'}
                </div>
              </div>
              <button
                onClick={() =>
                  removeSrc.mutate({
                    workspaceId,
                    sourceId: s.source_id,
                  })
                }
                disabled={removeSrc.isPending}
                className="text-xs px-2 py-0.5 bg-[#f87171]/15 text-[#f87171] border border-[#f87171]/40 rounded hover:bg-[#f87171]/25"
              >
                remove
              </button>
            </div>
          ))}
        </div>
      </div>

      <div className="flex gap-2">
        <input
          type="text"
          value={newQuery}
          onChange={(e) => setNewQuery(e.target.value)}
          placeholder="search query for new web_search source"
          className="flex-1 bg-[#0c1118] border border-[#1e2738] rounded px-2 py-1 text-sm text-[#e2e8f0]"
        />
        <button
          onClick={() => {
            if (!newQuery.trim()) return;
            addSrc.mutate({
              workspaceId,
              type: 'web_search',
              config: { query: newQuery.trim() },
            });
            setNewQuery('');
          }}
          disabled={addSrc.isPending || !newQuery.trim()}
          className="text-sm px-3 py-1 bg-[#60a5fa]/15 text-[#60a5fa] border border-[#60a5fa]/40 rounded hover:bg-[#60a5fa]/25 disabled:opacity-50"
        >
          Add
        </button>
      </div>

      <div>
        <button
          onClick={() => setShowSuggestions((v) => !v)}
          className="text-sm px-2 py-1 text-[#60a5fa] hover:text-[#93c5fd]"
        >
          {showSuggestions ? 'Hide' : 'Get'} LLM suggestions
        </button>
        {showSuggestions && suggestionsQ.isLoading && (
          <div className="text-[#7a8599] text-xs mt-2">Generating…</div>
        )}
        {showSuggestions && suggestions.length > 0 && (
          <div className="space-y-1 mt-2">
            {suggestions.map((s, idx) => (
              <div
                key={idx}
                className="bg-[#111820] border border-[#1e2738] rounded p-2 flex items-center justify-between gap-2"
              >
                <div className="min-w-0 flex-1">
                  <div className="text-sm text-[#e2e8f0]">
                    {String((s.config as Record<string, unknown>)?.query ?? '')}
                  </div>
                  <div className="text-[10px] text-[#7a8599]">{s.reason}</div>
                </div>
                <button
                  onClick={() =>
                    addSrc.mutate({
                      workspaceId,
                      type: s.type,
                      config: s.config,
                    })
                  }
                  className="text-xs px-2 py-0.5 bg-[#60a5fa]/15 text-[#60a5fa] border border-[#60a5fa]/40 rounded hover:bg-[#60a5fa]/25"
                >
                  accept
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ── Grand task ─────────────────────────────────────────────────────────────

function GrandTaskTab({ workspaceId }: { workspaceId: string }) {
  const proposalsQ = useGrandTaskProposals(workspaceId);
  const accept = useAcceptGrandTask();
  const reject = useRejectGrandTask();
  const proposals = proposalsQ.data?.proposals ?? [];

  if (proposalsQ.isLoading)
    return <div className="text-[#7a8599] text-sm">Loading…</div>;
  if (proposals.length === 0)
    return (
      <div className="text-[#7a8599] text-sm italic">
        No grand-task proposals yet. They land every 12 h once the workspace
        has at least 3 polished ideas.
      </div>
    );

  return (
    <div className="space-y-2">
      {proposals.map((p) => (
        <div
          key={p.proposal_id}
          className="bg-[#111820] border border-[#1e2738] rounded p-3 space-y-2"
        >
          <div className="text-sm text-[#e2e8f0] font-medium">{p.text}</div>
          {p.rationale && (
            <div className="text-xs text-[#7a8599] italic">{p.rationale}</div>
          )}
          {p.superseded_seed && (
            <div className="text-[10px] text-[#7a8599]">
              would replace seed: "{p.superseded_seed}"
            </div>
          )}
          <div className="flex gap-2">
            <button
              onClick={() =>
                accept.mutate({
                  workspaceId,
                  proposalId: p.proposal_id,
                })
              }
              disabled={accept.isPending}
              className="text-sm px-2 py-0.5 bg-[#34d399]/15 text-[#34d399] border border-[#34d399]/40 rounded hover:bg-[#34d399]/25"
            >
              accept · rotate seed
            </button>
            <button
              onClick={() =>
                reject.mutate({
                  workspaceId,
                  proposalId: p.proposal_id,
                })
              }
              disabled={reject.isPending}
              className="text-sm px-2 py-0.5 bg-[#7a8599]/15 text-[#7a8599] border border-[#7a8599]/40 rounded hover:bg-[#7a8599]/25"
            >
              reject
            </button>
          </div>
        </div>
      ))}
    </div>
  );
}

// ── Cross-workspace inbox ──────────────────────────────────────────────────

function InboxTab({ workspaceId }: { workspaceId: string }) {
  const inboxQ = useXWorkspaceInbox(workspaceId);
  const accept = useAcceptXWorkspaceKernel();
  const dismiss = useDismissXWorkspaceKernel();
  const proposals = inboxQ.data?.proposals ?? [];

  if (inboxQ.isLoading)
    return <div className="text-[#7a8599] text-sm">Loading inbox…</div>;
  if (proposals.length === 0)
    return (
      <div className="text-[#7a8599] text-sm italic">
        Inbox empty. Cross-workspace kernels arrive when another workspace
        produces a structurally-relevant idea.
      </div>
    );

  return (
    <div className="space-y-2">
      {proposals.map((p) => (
        <div
          key={p.kernel_id}
          className="bg-[#111820] border border-[#1e2738] rounded p-3 space-y-2"
        >
          <div className="flex items-center justify-between text-[10px] text-[#7a8599]">
            <code>from {p.source_workspace_id}</code>
            <span>relevance {p.relevance_score.toFixed(2)}</span>
          </div>
          <div className="text-sm text-[#e2e8f0] whitespace-pre-wrap">
            {p.text}
          </div>
          <div className="flex gap-2">
            <button
              onClick={() =>
                accept.mutate({
                  workspaceId,
                  kernelId: p.kernel_id,
                })
              }
              disabled={accept.isPending}
              className="text-sm px-2 py-0.5 bg-[#34d399]/15 text-[#34d399] border border-[#34d399]/40 rounded hover:bg-[#34d399]/25"
            >
              accept · feed into next cycles
            </button>
            <button
              onClick={() =>
                dismiss.mutate({
                  workspaceId,
                  kernelId: p.kernel_id,
                })
              }
              disabled={dismiss.isPending}
              className="text-sm px-2 py-0.5 bg-[#7a8599]/15 text-[#7a8599] border border-[#7a8599]/40 rounded hover:bg-[#7a8599]/25"
            >
              dismiss
            </button>
          </div>
        </div>
      ))}
    </div>
  );
}

// ── Settings ───────────────────────────────────────────────────────────────

function SettingsTab({ workspaceId }: { workspaceId: string }) {
  const cfgQ = useCompanionConfig(workspaceId);
  const update = useUpdateCompanionConfig();
  const [draft, setDraft] = useState<CompanionConfig | null>(null);

  // Reset draft whenever the loaded config changes (workspace switch / save).
  useEffect(() => {
    if (cfgQ.data?.config) setDraft({ ...cfgQ.data.config });
  }, [cfgQ.data?.config]);

  if (cfgQ.isLoading || !draft)
    return <div className="text-[#7a8599] text-sm">Loading config…</div>;

  const save = () => {
    if (!draft || !cfgQ.data?.config) return;
    const original = cfgQ.data.config;
    const patch: CompanionConfigPatch = {};
    if (draft.enabled !== original.enabled) patch.enabled = draft.enabled;
    if (draft.seed_prompt !== original.seed_prompt)
      patch.seed_prompt = draft.seed_prompt;
    if (draft.daily_budget_usd !== original.daily_budget_usd)
      patch.daily_budget_usd = draft.daily_budget_usd;
    if (draft.surface_threshold !== original.surface_threshold)
      patch.surface_threshold = draft.surface_threshold;
    if (draft.novelty_threshold !== original.novelty_threshold)
      patch.novelty_threshold = draft.novelty_threshold;
    if (draft.transferability_threshold !== original.transferability_threshold)
      patch.transferability_threshold = draft.transferability_threshold;
    if (draft.panel_threshold !== original.panel_threshold)
      patch.panel_threshold = draft.panel_threshold;
    if (draft.quiet_hours_start !== original.quiet_hours_start)
      patch.quiet_hours_start = draft.quiet_hours_start;
    if (draft.quiet_hours_end !== original.quiet_hours_end)
      patch.quiet_hours_end = draft.quiet_hours_end;
    if (Object.keys(patch).length === 0) return;
    update.mutate({ workspaceId, patch });
  };

  return (
    <div className="space-y-3 max-w-xl">
      <Field label="Enabled">
        <input
          type="checkbox"
          checked={draft.enabled}
          onChange={(e) =>
            setDraft({ ...draft, enabled: e.target.checked })
          }
        />
      </Field>
      <Field label="Seed prompt">
        <textarea
          value={draft.seed_prompt ?? ''}
          onChange={(e) =>
            setDraft({ ...draft, seed_prompt: e.target.value })
          }
          rows={3}
          className="w-full bg-[#0c1118] border border-[#1e2738] rounded px-2 py-1 text-sm text-[#e2e8f0]"
        />
      </Field>
      <NumField
        label="Daily budget USD"
        value={draft.daily_budget_usd}
        step={0.1}
        min={0}
        onChange={(v) => setDraft({ ...draft, daily_budget_usd: v })}
      />
      <NumField
        label="Surface threshold"
        value={draft.surface_threshold}
        step={0.05}
        min={0}
        max={1}
        onChange={(v) => setDraft({ ...draft, surface_threshold: v })}
      />
      <NumField
        label="Novelty threshold"
        value={draft.novelty_threshold}
        step={0.05}
        min={0}
        max={1}
        onChange={(v) => setDraft({ ...draft, novelty_threshold: v })}
      />
      <NumField
        label="Panel threshold"
        value={draft.panel_threshold}
        step={0.05}
        min={0}
        max={1}
        onChange={(v) => setDraft({ ...draft, panel_threshold: v })}
      />
      <NumField
        label="Transferability threshold"
        value={draft.transferability_threshold}
        step={0.05}
        min={0}
        max={1}
        onChange={(v) =>
          setDraft({ ...draft, transferability_threshold: v })
        }
      />
      <div className="grid grid-cols-2 gap-2">
        <NumField
          label="Quiet hours start"
          value={draft.quiet_hours_start}
          step={1}
          min={0}
          max={23}
          onChange={(v) =>
            setDraft({ ...draft, quiet_hours_start: Math.round(v) })
          }
        />
        <NumField
          label="Quiet hours end"
          value={draft.quiet_hours_end}
          step={1}
          min={0}
          max={23}
          onChange={(v) =>
            setDraft({ ...draft, quiet_hours_end: Math.round(v) })
          }
        />
      </div>
      <div className="flex items-center gap-3 pt-2">
        <button
          onClick={save}
          disabled={update.isPending}
          className="text-sm px-3 py-1 bg-[#60a5fa]/15 text-[#60a5fa] border border-[#60a5fa]/40 rounded hover:bg-[#60a5fa]/25 disabled:opacity-50"
        >
          {update.isPending ? 'Saving…' : 'Save'}
        </button>
        {update.isSuccess && (
          <span className="text-xs text-[#34d399]">saved</span>
        )}
        {update.isError && (
          <span className="text-xs text-[#f87171]">save failed</span>
        )}
      </div>
    </div>
  );
}

function Field({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <label className="block">
      <div className="text-xs text-[#7a8599] mb-1">{label}</div>
      {children}
    </label>
  );
}

function NumField({
  label,
  value,
  step,
  min,
  max,
  onChange,
}: {
  label: string;
  value: number;
  step?: number;
  min?: number;
  max?: number;
  onChange: (v: number) => void;
}) {
  return (
    <Field label={label}>
      <input
        type="number"
        value={value}
        step={step}
        min={min}
        max={max}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full bg-[#0c1118] border border-[#1e2738] rounded px-2 py-1 text-sm text-[#e2e8f0]"
      />
    </Field>
  );
}
