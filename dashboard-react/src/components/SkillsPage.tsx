import { useMemo, useState } from 'react';
import { ErrorPanel } from './ui/ErrorPanel';
import {
  useSkillsQuery,
  useSaveSkill,
  useDeleteSkill,
  useRunSkill,
  type Skill,
} from '../api/queries';

export function SkillsPage() {
  const skillsQ = useSkillsQuery();

  if (skillsQ.isLoading) {
    return (
      <div className="bg-[#111820] border border-[#1e2738] rounded-xl p-4">
        <div className="text-[#7a8599] text-sm">Loading skills…</div>
      </div>
    );
  }
  if (skillsQ.error) return <ErrorPanel error={skillsQ.error} onRetry={skillsQ.refetch} />;

  const skills = skillsQ.data?.skills ?? [];

  return (
    <div className="space-y-4 max-w-4xl">
      <div>
        <h1 className="text-xl font-semibold text-[#e2e8f0]">Skills</h1>
        <p className="text-xs text-[#7a8599] mt-1">
          Saved task templates that any crew can replay. Use{' '}
          <code className="text-[#60a5fa]">{'{placeholder}'}</code> in templates
          to expose runtime arguments. Also accessible via Signal:{' '}
          <code className="text-[#60a5fa]">/skill save</code>,{' '}
          <code className="text-[#60a5fa]">/skill run</code>.
        </p>
      </div>

      <NewSkillForm />

      {skills.length === 0 ? (
        <div className="bg-[#111820] border border-[#1e2738] rounded-xl p-4 text-sm text-[#7a8599]">
          No skills saved yet. Add one above or send <code>/skill save</code> in Signal.
        </div>
      ) : (
        <ul className="space-y-3">
          {skills.map((s) => <SkillCard key={s.name} skill={s} />)}
        </ul>
      )}
    </div>
  );
}

// ── New skill form ────────────────────────────────────────────────────────

function NewSkillForm() {
  const save = useSaveSkill();
  const [name, setName] = useState('');
  const [template, setTemplate] = useState('');
  const [description, setDescription] = useState('');
  const [feedback, setFeedback] = useState('');

  const placeholders = useMemo(() => extractPlaceholders(template), [template]);
  const error = save.error instanceof Error ? save.error.message : '';

  const submit = async () => {
    setFeedback('');
    if (!name.trim() || !template.trim()) return;
    try {
      await save.mutateAsync({
        name: name.trim(),
        task_template: template.trim(),
        description: description.trim(),
      });
      setName('');
      setTemplate('');
      setDescription('');
      setFeedback('Saved.');
      setTimeout(() => setFeedback(''), 2500);
    } catch {/* surfaced via save.error */}
  };

  return (
    <div className="bg-[#111820] border border-[#1e2738] rounded-xl p-4 space-y-3">
      <div className="text-sm font-semibold text-[#e2e8f0]">New skill</div>
      <label className="block">
        <span className="text-xs text-[#7a8599]">Name</span>
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="e.g. weekly status"
          className="mt-1 w-full bg-[#0a0f18] border border-[#1e2738] rounded px-3 py-2 text-[#e2e8f0] text-sm"
        />
      </label>
      <label className="block">
        <span className="text-xs text-[#7a8599]">Task template</span>
        <textarea
          value={template}
          onChange={(e) => setTemplate(e.target.value)}
          placeholder="Summarize my Q{quarter} status with focus on {topic}"
          rows={3}
          className="mt-1 w-full bg-[#0a0f18] border border-[#1e2738] rounded px-3 py-2 text-[#e2e8f0] text-sm resize-y font-mono"
        />
        {placeholders.length > 0 && (
          <span className="text-xs text-[#94a3b8] mt-1 block">
            args: {placeholders.map((p) => <code key={p} className="text-[#60a5fa] mr-2">{`{${p}}`}</code>)}
          </span>
        )}
      </label>
      <label className="block">
        <span className="text-xs text-[#7a8599]">Description (optional)</span>
        <input
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder="What this skill is useful for"
          className="mt-1 w-full bg-[#0a0f18] border border-[#1e2738] rounded px-3 py-2 text-[#e2e8f0] text-sm"
        />
      </label>
      <div className="flex items-center gap-3">
        <button
          onClick={submit}
          disabled={save.isPending || !name.trim() || !template.trim()}
          className="px-4 py-2 bg-[#2563eb] hover:bg-[#1d4ed8] disabled:opacity-50 rounded text-white text-sm"
        >
          {save.isPending ? 'Saving…' : 'Save skill'}
        </button>
        {error && <span className="text-[#f87171] text-sm">{error}</span>}
        {feedback && <span className="text-[#34d399] text-sm">{feedback}</span>}
      </div>
    </div>
  );
}

// ── One skill card ────────────────────────────────────────────────────────

function SkillCard({ skill }: { skill: Skill }) {
  const remove = useDeleteSkill();
  const run = useRunSkill();
  const [argValues, setArgValues] = useState<Record<string, string>>({});
  const [output, setOutput] = useState('');
  const [error, setError] = useState('');

  const successRate = skill.run_count > 0
    ? `${skill.success_count}/${skill.run_count} (${Math.round(100 * skill.success_count / skill.run_count)}%)`
    : '—';

  const submit = async () => {
    setError('');
    setOutput('');
    try {
      const res = await run.mutateAsync({ name: skill.name, args: argValues });
      setOutput(res.result);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    }
  };

  const onDelete = async () => {
    if (!confirm(`Delete skill "${skill.name}"?`)) return;
    try { await remove.mutateAsync(skill.name); } catch {/* ignored */}
  };

  return (
    <li className="bg-[#111820] border border-[#1e2738] rounded-xl p-4 space-y-3">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex items-baseline gap-3 flex-wrap">
            <span className="text-sm font-semibold text-[#e2e8f0]">{skill.name}</span>
            <span className="text-xs text-[#94a3b8]">runs: {successRate}</span>
            {skill.last_run_at && (
              <span className="text-xs text-[#7a8599]">last {skill.last_run_at}</span>
            )}
          </div>
          {skill.description && (
            <div className="text-xs text-[#7a8599] mt-1">{skill.description}</div>
          )}
        </div>
        <button
          onClick={onDelete}
          disabled={remove.isPending}
          className="text-xs px-2 py-1 text-[#94a3b8] hover:text-[#f87171] hover:bg-[#1e2738] rounded"
        >
          Delete
        </button>
      </div>

      <pre className="text-xs text-[#cbd5e1] bg-[#0a0f18] border border-[#1e2738] rounded p-2 whitespace-pre-wrap font-mono">
        {skill.task_template}
      </pre>

      {skill.args_schema.length > 0 && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
          {skill.args_schema.map((arg) => (
            <label key={arg} className="block">
              <span className="text-xs text-[#7a8599]">{arg}</span>
              <input
                value={argValues[arg] ?? ''}
                onChange={(e) => setArgValues((prev) => ({ ...prev, [arg]: e.target.value }))}
                className="mt-1 w-full bg-[#0a0f18] border border-[#1e2738] rounded px-2 py-1 text-[#e2e8f0] text-sm font-mono"
              />
            </label>
          ))}
        </div>
      )}

      <div className="flex items-center gap-3">
        <button
          onClick={submit}
          disabled={run.isPending}
          className="px-3 py-1.5 bg-[#2563eb] hover:bg-[#1d4ed8] disabled:opacity-50 rounded text-white text-sm"
        >
          {run.isPending ? 'Running…' : 'Run'}
        </button>
        {error && <span className="text-[#f87171] text-sm">{error}</span>}
      </div>

      {output && (
        <div className="border-t border-[#1e2738] pt-3">
          <div className="text-xs text-[#7a8599] mb-1">Result</div>
          <pre className="text-xs text-[#e2e8f0] bg-[#0a0f18] border border-[#1e2738] rounded p-3 whitespace-pre-wrap max-h-64 overflow-auto">
            {output}
          </pre>
        </div>
      )}
    </li>
  );
}

// ── helper ────────────────────────────────────────────────────────────────

function extractPlaceholders(template: string): string[] {
  const re = /\{([a-zA-Z_][a-zA-Z0-9_]*)\}/g;
  const seen: string[] = [];
  let m;
  while ((m = re.exec(template))) {
    if (!seen.includes(m[1])) seen.push(m[1]);
  }
  return seen;
}
