import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { ProjectProvider } from './context/ProjectContext';
import { Layout } from './components/Layout';
import { Dashboard } from './components/Dashboard';
import { KanbanBoard } from './components/KanbanBoard';
import { BudgetDashboard } from './components/BudgetDashboard';
import { AuditFeed } from './components/AuditFeed';
import { GovernanceQueue } from './components/GovernanceQueue';
import { OrgChart } from './components/OrgChart';
import { CostCharts } from './components/CostCharts';
import { WorkspacesPage } from './components/WorkspacesPage';

export default function App() {
  return (
    <ProjectProvider>
      <BrowserRouter basename="/cp">
        <Routes>
          <Route element={<Layout />}>
            <Route path="/" element={<Dashboard />} />
            <Route path="/tickets" element={<KanbanBoard />} />
            <Route path="/budgets" element={<BudgetDashboard />} />
            <Route path="/audit" element={<AuditFeed />} />
            <Route path="/governance" element={<GovernanceQueue />} />
            <Route path="/org-chart" element={<OrgChart />} />
            <Route path="/costs" element={<CostCharts />} />
            <Route path="/workspaces" element={<WorkspacesPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </ProjectProvider>
  );
}
