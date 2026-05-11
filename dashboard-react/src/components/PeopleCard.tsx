// People list — Q4.2 L1 + L2 surface.
// Lists tracked people with counts + mute/forget actions.

import {
  useCompanionPeople,
  useCompanionCentrality,
  useMutePerson,
  useForgetPerson,
  useForgetAllPeople,
  useMuteSuggestionsFor,
  usePathOptOut,
  type PersonProfile,
} from '../api/queries';

const TEXT_DIM = '#7a8599';
const TEXT_BRIGHT = '#e2e8f0';
const PANEL = '#111820';
const BORDER = '#1e2738';

export function PeopleCard() {
  const peopleQ = useCompanionPeople();
  const centralityQ = useCompanionCentrality();
  const forgetAll = useForgetAllPeople();

  const data = peopleQ.data;
  if (!data) return <div style={{ color: TEXT_DIM }}>Loading…</div>;

  if (!data.enabled) {
    return (
      <div
        className="rounded p-4 border"
        style={{ background: PANEL, borderColor: BORDER }}
      >
        <div style={{ color: TEXT_BRIGHT }} className="text-sm font-medium">
          Person correlation is disabled
        </div>
        <p className="text-xs mt-2" style={{ color: TEXT_DIM }}>
          Enable in <code>/cp/settings</code> if you want this. Read{' '}
          <code>docs/PERSON_CORRELATION.md</code> for the ethics framing,
          opt-in flow, and per-level explainers before turning it on.
        </p>
      </div>
    );
  }

  const people = data.people || [];
  const centrality = centralityQ.data;
  const centralityMap = new Map<string, number>(
    (centrality?.scores || []).map((s) => [s.person_id, s.score]),
  );

  return (
    <div className="space-y-4">
      <div
        className="rounded p-3 border flex items-center justify-between"
        style={{ background: PANEL, borderColor: BORDER }}
      >
        <div>
          <div className="text-sm font-medium" style={{ color: TEXT_BRIGHT }}>
            {people.length} tracked person(s) · {data.muted.length} muted
          </div>
          {centrality?.enabled && (
            <div className="text-[10px]" style={{ color: TEXT_DIM }}>
              Centrality formula: <code>{centrality.formula}</code>{' '}
              {centrality.caveat && '· ⚠️ scores are math, not directives'}
            </div>
          )}
        </div>
        <button
          onClick={() => {
            if (confirm('Forget ALL person data? This cannot be undone.')) {
              forgetAll.mutate();
            }
          }}
          className="text-xs px-2 py-0.5 rounded border"
          style={{ color: '#f87171', borderColor: '#7f1d1d' }}
        >
          Forget all
        </button>
      </div>

      {people.length === 0 && (
        <div style={{ color: TEXT_DIM }} className="text-sm italic">
          No tracked people yet. The system will populate this as you receive
          emails, attend calendar events, etc.
        </div>
      )}

      <div className="space-y-2">
        {people.map((p) => (
          <PersonRow
            key={p.person_id}
            person={p}
            centralityScore={centralityMap.get(p.person_id)}
          />
        ))}
      </div>
    </div>
  );
}

function PersonRow({
  person,
  centralityScore,
}: {
  person: PersonProfile;
  centralityScore?: number;
}) {
  const mute = useMutePerson();
  const forget = useForgetPerson();
  const muteSug = useMuteSuggestionsFor();
  const optOut = usePathOptOut();

  const display =
    person.display_names?.[0] || person.person_id;

  return (
    <div
      className="rounded p-3 border space-y-1"
      style={{ background: PANEL, borderColor: BORDER }}
    >
      <div className="flex items-center justify-between gap-2">
        <div>
          <div className="text-sm" style={{ color: TEXT_BRIGHT }}>
            {display}
          </div>
          <code className="text-[10px]" style={{ color: TEXT_DIM }}>
            {person.person_id}
          </code>
        </div>
        {centralityScore !== undefined && (
          <span
            className="text-[10px] px-1.5 py-0.5 rounded"
            style={{ background: '#60a5fa22', color: '#60a5fa' }}
          >
            score {centralityScore.toFixed(2)}
          </span>
        )}
      </div>
      <div className="text-[10px] flex flex-wrap gap-2" style={{ color: TEXT_DIM }}>
        <span>{person.total_occurrences} hits</span>
        <span>·</span>
        <span>{person.modality_count} modalities</span>
        <span>·</span>
        <span>last {person.last_seen.slice(0, 10)}</span>
      </div>
      <div className="flex flex-wrap gap-1.5 mt-1">
        {Object.entries(person.occurrences_per_modality).map(([m, c]) => (
          <span
            key={m}
            className="text-[10px] px-1.5 py-0.5 rounded"
            style={{ background: '#1e2738', color: TEXT_DIM }}
          >
            {m} × {c}
          </span>
        ))}
      </div>
      <div className="flex flex-wrap gap-1 mt-2">
        <button
          onClick={() => mute.mutate({ person_id: person.person_id })}
          className="text-[10px] px-2 py-0.5 rounded border"
          style={{ color: TEXT_DIM, borderColor: BORDER }}
        >
          mute
        </button>
        <button
          onClick={() => muteSug.mutate({ person_id: person.person_id })}
          className="text-[10px] px-2 py-0.5 rounded border"
          style={{ color: TEXT_DIM, borderColor: BORDER }}
        >
          mute suggestions
        </button>
        <button
          onClick={() => optOut.mutate({ person_id: person.person_id })}
          className="text-[10px] px-2 py-0.5 rounded border"
          style={{ color: TEXT_DIM, borderColor: BORDER }}
        >
          opt-out of paths
        </button>
        <button
          onClick={() => {
            if (confirm(`Forget ${display}?`)) {
              forget.mutate({ person_id: person.person_id });
            }
          }}
          className="text-[10px] px-2 py-0.5 rounded border"
          style={{ color: '#f87171', borderColor: '#7f1d1d' }}
        >
          forget
        </button>
      </div>
    </div>
  );
}
