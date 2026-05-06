import { useEffect, useMemo, useState } from 'react';
import {
  useActiveSessionQuery,
  useSessionQuery,
  useSessionsQuery,
} from '../api/brainstorm';
import { ReportView } from './brainstorm/ReportView';
import { SessionView } from './brainstorm/SessionView';
import { SessionsList } from './brainstorm/SessionsList';
import { StartPanel } from './brainstorm/StartPanel';

type ViewMode = 'auto' | 'start' | 'inspect';

export function BrainstormPage() {
  const activeQuery = useActiveSessionQuery();
  const sessionsQuery = useSessionsQuery();

  const [mode, setMode] = useState<ViewMode>('auto');
  const [inspectId, setInspectId] = useState<string | null>(null);

  const activeSession = activeQuery.data?.session ?? null;
  const sessions = sessionsQuery.data ?? [];

  const inspectQuery = useSessionQuery(
    mode === 'inspect' ? inspectId : null,
  );
  const inspectSession = inspectQuery.data ?? null;

  // Auto-pick what to show in the main panel.
  const showing = useMemo<
    { kind: 'start' } | { kind: 'active' } | { kind: 'inspect' }
  >(() => {
    if (mode === 'start') return { kind: 'start' };
    if (mode === 'inspect' && inspectId) return { kind: 'inspect' };
    if (activeSession) return { kind: 'active' };
    return { kind: 'start' };
  }, [mode, inspectId, activeSession]);

  // When a session is finished or cancelled, drop into start mode.
  useEffect(() => {
    if (mode === 'auto' && !activeSession && inspectId == null) {
      // Stay in start mode by default; nothing to do.
    }
  }, [mode, activeSession, inspectId]);

  const onSelectFromList = (id: string) => {
    if (activeSession && id === activeSession.session_id) {
      setMode('auto');
      setInspectId(null);
      return;
    }
    setMode('inspect');
    setInspectId(id);
  };

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-2xl font-bold text-[#e2e8f0]">Brainstorm</h1>
        <p className="text-sm text-[#7a8599] mt-1">
          Run structured idea-generation sessions — solo or with creative
          agents in a joint effort.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_18rem] gap-4">
        {/* Main panel */}
        <div className="rounded-md bg-[#111820] border border-[#1e2738] p-4">
          {showing.kind === 'start' && (
            <StartPanel onStarted={() => setMode('auto')} />
          )}
          {showing.kind === 'active' && activeSession && (
            <SessionView
              session={activeSession}
              onAfterFinish={() => setMode('auto')}
            />
          )}
          {showing.kind === 'inspect' && inspectQuery.isLoading && (
            <div className="text-sm text-[#7a8599]">Loading session…</div>
          )}
          {showing.kind === 'inspect' && inspectSession && (
            inspectSession.status === 'complete' ? (
              <ReportView
                session={inspectSession}
                onClose={() => {
                  setMode('auto');
                  setInspectId(null);
                }}
              />
            ) : (
              <SessionView
                session={inspectSession}
                onAfterFinish={() => setMode('auto')}
              />
            )
          )}
        </div>

        {/* Sidebar */}
        <aside className="space-y-3">
          <div className="rounded-md bg-[#111820] border border-[#1e2738] p-3">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-semibold text-[#e2e8f0]">
                Sessions
              </h3>
              <button
                type="button"
                onClick={() => {
                  setMode('start');
                  setInspectId(null);
                }}
                className="text-[11px] px-2 py-0.5 rounded bg-[#60a5fa]/10 text-[#60a5fa] border border-[#60a5fa]/20 hover:bg-[#60a5fa]/20"
              >
                New
              </button>
            </div>
            <SessionsList
              sessions={sessions}
              selectedId={
                showing.kind === 'inspect'
                  ? inspectId
                  : activeSession?.session_id ?? null
              }
              onSelect={onSelectFromList}
            />
          </div>
        </aside>
      </div>
    </div>
  );
}
