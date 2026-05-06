import { lazy, Suspense } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { ProjectProvider } from './context/ProjectContext';
import { Layout } from './components/Layout';
import { Dashboard } from './components/Dashboard';
import { KanbanBoard } from './components/KanbanBoard';
import { BudgetDashboard } from './components/BudgetDashboard';
import { AuditFeed } from './components/AuditFeed';
import { GovernanceQueue } from './components/GovernanceQueue';
import { OrgChart } from './components/OrgChart';
import { CostCharts } from './components/CostCharts';
import { NotFound } from './components/ui/NotFound';
import { ErrorBoundary } from './components/ui/ErrorBoundary';
import { Skeleton } from './components/ui/Skeleton';

// Heavy routes load on demand — avoids shipping Chart.js / massive components on first paint.
const WorkspacesPage = lazy(() =>
  import('./components/WorkspacesPage').then((m) => ({ default: m.WorkspacesPage })),
);
const EvolutionMonitor = lazy(() =>
  import('./components/EvolutionMonitor').then((m) => ({ default: m.EvolutionMonitor })),
);
const KnowledgeBases = lazy(() =>
  import('./components/KnowledgeBases').then((m) => ({ default: m.KnowledgeBases })),
);
const TasksPage = lazy(() =>
  import('./components/TasksPage').then((m) => ({ default: m.TasksPage })),
);
const OpsPage = lazy(() =>
  import('./components/OpsPage').then((m) => ({ default: m.OpsPage })),
);
const LlmsPage = lazy(() =>
  import('./components/LlmsPage').then((m) => ({ default: m.LlmsPage })),
);
const NotesPage = lazy(() =>
  import('./components/NotesPage').then((m) => ({ default: m.NotesPage })),
);
const WikiPage = lazy(() =>
  import('./components/WikiPage').then((m) => ({ default: m.WikiPage })),
);
const ForgePage = lazy(() =>
  import('./components/ForgePage').then((m) => ({ default: m.ForgePage })),
);
const ForgeToolDetailPage = lazy(() =>
  import('./components/ForgeToolDetailPage').then((m) => ({
    default: m.ForgeToolDetailPage,
  })),
);
const ForgeSettingsPage = lazy(() =>
  import('./components/ForgeSettingsPage').then((m) => ({
    default: m.ForgeSettingsPage,
  })),
);
const ForgeCompositionsPage = lazy(() =>
  import('./components/ForgeCompositionsPage').then((m) => ({
    default: m.ForgeCompositionsPage,
  })),
);
const AffectPage = lazy(() =>
  import('./components/AffectPage').then((m) => ({ default: m.AffectPage })),
);
const EpistemicPage = lazy(() =>
  import('./components/EpistemicPage').then((m) => ({ default: m.EpistemicPage })),
);
const BrainstormPage = lazy(() =>
  import('./components/BrainstormPage').then((m) => ({ default: m.BrainstormPage })),
);
const ChangesPage = lazy(() =>
  import('./components/ChangesPage').then((m) => ({ default: m.ChangesPage })),
);
const CodingSessionsPage = lazy(() =>
  import('./components/CodingSessionsPage').then((m) => ({
    default: m.CodingSessionsPage,
  })),
);

function RouteFallback() {
  return (
    <div className="space-y-4">
      <Skeleton className="h-6 w-48" />
      <Skeleton className="h-64" />
    </div>
  );
}

function LazyRoute({ children }: { children: React.ReactNode }) {
  return (
    <ErrorBoundary>
      <Suspense fallback={<RouteFallback />}>{children}</Suspense>
    </ErrorBoundary>
  );
}

export default function App() {
  return (
    <ProjectProvider>
      <BrowserRouter basename="/cp">
        <Routes>
          <Route element={<Layout />}>
            <Route path="/" element={<ErrorBoundary><Dashboard /></ErrorBoundary>} />
            <Route path="/tickets" element={<ErrorBoundary><KanbanBoard /></ErrorBoundary>} />
            <Route path="/tasks" element={<LazyRoute><TasksPage /></LazyRoute>} />
            <Route path="/ops" element={<LazyRoute><OpsPage /></LazyRoute>} />
            <Route path="/llms" element={<LazyRoute><LlmsPage /></LazyRoute>} />
            <Route path="/notes" element={<LazyRoute><NotesPage /></LazyRoute>} />
            <Route path="/wiki" element={<LazyRoute><WikiPage /></LazyRoute>} />
            <Route path="/budgets" element={<ErrorBoundary><BudgetDashboard /></ErrorBoundary>} />
            <Route path="/audit" element={<ErrorBoundary><AuditFeed /></ErrorBoundary>} />
            <Route path="/governance" element={<ErrorBoundary><GovernanceQueue /></ErrorBoundary>} />
            <Route path="/org-chart" element={<ErrorBoundary><OrgChart /></ErrorBoundary>} />
            <Route path="/costs" element={<ErrorBoundary><CostCharts /></ErrorBoundary>} />
            <Route path="/workspaces" element={<LazyRoute><WorkspacesPage /></LazyRoute>} />
            <Route path="/evolution" element={<LazyRoute><EvolutionMonitor /></LazyRoute>} />
            <Route path="/knowledge" element={<LazyRoute><KnowledgeBases /></LazyRoute>} />
            <Route path="/forge" element={<LazyRoute><ForgePage /></LazyRoute>} />
            <Route path="/forge/settings" element={<LazyRoute><ForgeSettingsPage /></LazyRoute>} />
            <Route path="/forge/compositions" element={<LazyRoute><ForgeCompositionsPage /></LazyRoute>} />
            <Route path="/forge/:id" element={<LazyRoute><ForgeToolDetailPage /></LazyRoute>} />
            <Route path="/affect" element={<LazyRoute><AffectPage /></LazyRoute>} />
            <Route path="/epistemic" element={<LazyRoute><EpistemicPage /></LazyRoute>} />
            <Route path="/brainstorm" element={<LazyRoute><BrainstormPage /></LazyRoute>} />
            <Route path="/changes" element={<LazyRoute><ChangesPage /></LazyRoute>} />
            <Route path="/coding-sessions" element={<LazyRoute><CodingSessionsPage /></LazyRoute>} />
            <Route path="*" element={<NotFound />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </ProjectProvider>
  );
}
