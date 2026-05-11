// Social graph viz + analysis surfaces — Q4.2 L4 + L4.1-L4.3.
//
// Renders the co-appearance graph; surfaces communities + bridges
// when their sub-toggles are on. Path query is a small form.
//
// All views gated by master switches. When L4 is off, shows a clear
// "disabled" panel with the warning copy.

import { useState } from 'react';
import {
  useCompanionGraph,
  useCompanionCommunities,
  useCompanionStructural,
  useGraphPath,
  useForgetGraph,
  useDissolveCluster,
} from '../api/queries';

const TEXT_DIM = '#7a8599';
const TEXT_BRIGHT = '#e2e8f0';
const PANEL = '#111820';
const BORDER = '#1e2738';
const WARN = '#f87171';

export function SocialGraphCard() {
  const graphQ = useCompanionGraph();
  const data = graphQ.data;

  if (!data) return <div style={{ color: TEXT_DIM }}>Loading…</div>;

  if (!data.enabled) {
    return (
      <div className="rounded p-4 border" style={{ background: PANEL, borderColor: BORDER }}>
        <div className="text-sm font-medium" style={{ color: TEXT_BRIGHT }}>
          Social graph is disabled
        </div>
        <p className="text-xs mt-2" style={{ color: WARN }}>
          ⚠️ This data exists only when you opt in. Enabling requires
          typed-phrase confirmation in <code>/cp/settings</code>.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div
        className="rounded p-3 border"
        style={{ background: '#7f1d1d11', borderColor: '#7f1d1d44' }}
      >
        <p className="text-[11px]" style={{ color: WARN }}>
          ⚠️ This data exists only because you opted in. It is not
          synchronized off-host (excluded from DR backups).
        </p>
      </div>

      <GraphList edges={data.edges} nodes={data.nodes} />
      <CommunitiesView />
      <StructuralView />
      <PathQueryForm />
      <DangerZone />
    </div>
  );
}

function GraphList({
  edges,
  nodes,
}: {
  edges: { a: string; b: string; weight: number; last_touched: string }[];
  nodes: string[];
}) {
  return (
    <div
      className="rounded p-3 border"
      style={{ background: PANEL, borderColor: BORDER }}
    >
      <div className="text-sm font-medium mb-1" style={{ color: TEXT_BRIGHT }}>
        Co-appearance graph
      </div>
      <div className="text-[10px] mb-2" style={{ color: TEXT_DIM }}>
        {nodes.length} nodes · {edges.length} edges
      </div>
      {edges.length === 0 ? (
        <div className="text-xs italic" style={{ color: TEXT_DIM }}>
          No edges yet. Graph populates from calendar attendees who share
          events.
        </div>
      ) : (
        <div className="space-y-1 max-h-64 overflow-y-auto">
          {edges
            .slice()
            .sort((a, b) => b.weight - a.weight)
            .slice(0, 50)
            .map((e, i) => (
              <div
                key={`${e.a}-${e.b}-${i}`}
                className="text-[11px] flex justify-between"
                style={{ color: TEXT_DIM }}
              >
                <span>
                  {e.a} ↔ {e.b}
                </span>
                <span>w {e.weight.toFixed(1)}</span>
              </div>
            ))}
        </div>
      )}
    </div>
  );
}

function CommunitiesView() {
  const q = useCompanionCommunities();
  const dissolve = useDissolveCluster();
  const data = q.data;
  if (!data || !data.enabled) return null;

  return (
    <div
      className="rounded p-3 border"
      style={{ background: PANEL, borderColor: BORDER }}
    >
      <div className="text-sm font-medium mb-1" style={{ color: TEXT_BRIGHT }}>
        Communities (L4.2)
      </div>
      <div className="text-[10px] mb-2" style={{ color: TEXT_DIM }}>
        Modularity: {data.modularity.toFixed(3)}{' '}
        {data.modularity >= 0.3
          ? '· strong clustering'
          : '· weak clustering — interpretation unstable'}
      </div>
      {data.caveat && (
        <div className="text-[10px] italic mb-2" style={{ color: TEXT_DIM }}>
          {data.caveat}
        </div>
      )}
      {data.clusters.length === 0 && (
        <div className="text-xs italic" style={{ color: TEXT_DIM }}>
          No clusters detected.
        </div>
      )}
      {data.clusters.map((c) => (
        <div
          key={c.id}
          className="text-[11px] mb-2 pb-2 border-b"
          style={{ color: TEXT_DIM, borderColor: BORDER }}
        >
          <div className="flex justify-between items-center">
            <span style={{ color: TEXT_BRIGHT }}>
              Cluster {c.id} ({c.size} people, density {c.density.toFixed(2)})
            </span>
            <button
              onClick={() => {
                if (confirm(`Dissolve cluster ${c.id}?`)) {
                  dissolve.mutate({ member_emails: c.members });
                }
              }}
              className="text-[10px] px-1.5 py-0.5 rounded border"
              style={{ color: TEXT_DIM, borderColor: BORDER }}
            >
              dissolve
            </button>
          </div>
          <div className="ml-2 mt-1 text-[10px]">
            {c.members.slice(0, 6).join(', ')}
            {c.members.length > 6 && ` …+${c.members.length - 6}`}
          </div>
        </div>
      ))}
    </div>
  );
}

function StructuralView() {
  const q = useCompanionStructural();
  const data = q.data;
  if (!data || !data.enabled) return null;

  return (
    <div
      className="rounded p-3 border"
      style={{ background: PANEL, borderColor: BORDER }}
    >
      <div className="text-sm font-medium mb-1" style={{ color: TEXT_BRIGHT }}>
        Bridges + cut-vertices (L4.3)
      </div>
      {data.caveat && (
        <div className="text-[10px] italic mb-2" style={{ color: TEXT_DIM }}>
          {data.caveat}
        </div>
      )}
      <div className="text-[11px]" style={{ color: TEXT_DIM }}>
        <div className="mb-1">
          <span style={{ color: TEXT_BRIGHT }}>Cut-vertices:</span>{' '}
          {data.cut_vertices.length === 0
            ? '(none)'
            : data.cut_vertices.join(', ')}
        </div>
        <div>
          <span style={{ color: TEXT_BRIGHT }}>Bridges:</span>{' '}
          {data.bridges.length === 0
            ? '(none)'
            : data.bridges.map((b, i) => (
                <span key={i}>
                  {i > 0 && ' · '}
                  {b.join(' ↔ ')}
                </span>
              ))}
        </div>
      </div>
    </div>
  );
}

function PathQueryForm() {
  const [source, setSource] = useState('');
  const [target, setTarget] = useState('');
  const path = useGraphPath();

  return (
    <div
      className="rounded p-3 border"
      style={{ background: PANEL, borderColor: BORDER }}
    >
      <div className="text-sm font-medium mb-1" style={{ color: TEXT_BRIGHT }}>
        Shortest-path query (L4.1)
      </div>
      <div className="text-[10px] mb-2" style={{ color: TEXT_DIM }}>
        Operator-initiated. Logged to{' '}
        <code>workspace/companion/social_graph_query_log.jsonl</code> for
        your review.
      </div>
      <div className="flex gap-2 text-xs">
        <input
          value={source}
          onChange={(e) => setSource(e.target.value)}
          placeholder="from email"
          className="flex-1 bg-[#0c1118] border border-[#1e2738] rounded px-2 py-1"
          style={{ color: TEXT_BRIGHT }}
        />
        <input
          value={target}
          onChange={(e) => setTarget(e.target.value)}
          placeholder="to email"
          className="flex-1 bg-[#0c1118] border border-[#1e2738] rounded px-2 py-1"
          style={{ color: TEXT_BRIGHT }}
        />
        <button
          onClick={() => path.mutate({ source, target })}
          disabled={!source || !target || path.isPending}
          className="text-xs px-3 py-1 rounded border disabled:opacity-30"
          style={{ background: '#60a5fa22', color: '#60a5fa', borderColor: '#60a5fa66' }}
        >
          Find
        </button>
      </div>
      {path.data && (
        <div className="mt-2 text-xs">
          {path.data.ok && path.data.path ? (
            <div style={{ color: '#34d399' }}>
              Path ({path.data.hops} hops):{' '}
              {path.data.path.join(' → ')}
            </div>
          ) : (
            <div style={{ color: WARN }}>{path.data.error || 'no path'}</div>
          )}
        </div>
      )}
    </div>
  );
}

function DangerZone() {
  const forget = useForgetGraph();
  return (
    <div
      className="rounded p-3 border"
      style={{ background: PANEL, borderColor: '#7f1d1d44' }}
    >
      <button
        onClick={() => {
          if (confirm('Forget the entire social graph? Profiles are preserved.')) {
            forget.mutate();
          }
        }}
        className="text-xs px-3 py-1 rounded border"
        style={{ color: WARN, borderColor: WARN }}
      >
        Forget graph (preserves profiles)
      </button>
    </div>
  );
}
