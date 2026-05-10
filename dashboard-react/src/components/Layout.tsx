import { useState } from 'react';
import { NavLink, Outlet } from 'react-router-dom';
import { ProjectSwitcher } from './ProjectSwitcher';
import { useHealthQuery } from '../api/queries';

const NAV_ITEMS = [
  { to: '/', label: 'Dashboard', icon: '📊', exact: true },
  { to: '/chat', label: 'Chat', icon: '💬', exact: false },
  { to: '/tickets', label: 'Tickets', icon: '🎫', exact: false },
  { to: '/tasks', label: 'Tasks', icon: '⚡', exact: false },
  { to: '/budgets', label: 'Budgets', icon: '💰', exact: false },
  { to: '/audit', label: 'Audit', icon: '📜', exact: false },
  { to: '/governance', label: 'Governance', icon: '⚖️', exact: false },
  { to: '/changes', label: 'Changes', icon: '✏️', exact: false },
  { to: '/architecture-requests', label: 'Architecture', icon: '🏛', exact: false },
  { to: '/action-requests', label: 'Actions', icon: '✉️', exact: false },
  { to: '/threads', label: 'Threads', icon: '🧵', exact: false },
  { to: '/proposals', label: 'Proposals', icon: '📋', exact: false },
  { to: '/org-chart', label: 'Org Chart', icon: '🏢', exact: false },
  { to: '/costs', label: 'Costs', icon: '📈', exact: false },
  { to: '/workspaces', label: 'Workspaces', icon: '🧠', exact: false },
  { to: '/affect', label: 'Affect', icon: '🌡️', exact: false },
  { to: '/epistemic', label: 'Epistemic', icon: '🪞', exact: false },
  { to: '/evolution', label: 'Evolution', icon: '🧬', exact: false },
  { to: '/monitor', label: 'Monitor', icon: '🩺', exact: false },
  { to: '/ops', label: 'Ops', icon: '🛠️', exact: false },
  { to: '/llms', label: 'LLMs', icon: '🤖', exact: false },
  { to: '/knowledge', label: 'Knowledge', icon: '📚', exact: false },
  { to: '/notes', label: 'Notes', icon: '📝', exact: false },
  { to: '/wiki', label: 'Wiki', icon: '📖', exact: false },
  { to: '/brainstorm', label: 'Brainstorm', icon: '💡', exact: false },
  { to: '/forge', label: 'Forge', icon: '🔨', exact: false },
  { to: '/coding-sessions', label: 'Sessions', icon: '🧪', exact: false },
  { to: '/inquiries', label: 'Inquiries', icon: '📜', exact: false },
  { to: '/skills', label: 'Skills', icon: '🪄', exact: false },
  { to: '/files', label: 'Files', icon: '📁', exact: false },
  { to: '/settings', label: 'Settings', icon: '⚙️', exact: false },
];

export function Layout() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const { data: health } = useHealthQuery();

  const statusColor =
    health?.status === 'ok'
      ? 'bg-[#34d399] text-[#0a0e14]'
      : health?.status === 'degraded'
      ? 'bg-[#fbbf24] text-[#0a0e14]'
      : 'bg-[#f87171] text-[#0a0e14]';

  const statusLabel = health?.status ?? 'unknown';

  return (
    <div className="flex h-screen overflow-hidden bg-[#0a0e14]">
      {/* Mobile overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black/60 z-20 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar
          iOS PWA safe-area: when installed as standalone with
          `apple-mobile-web-app-status-bar-style: black-translucent`
          (see public/index.html), the viewport extends UNDER the
          status bar.  Without `pt-[env(safe-area-inset-top)]` here,
          the BotArmy logo block lands behind the clock and signal
          icons.  Same for the home-indicator at the bottom edge —
          `pb-[env(safe-area-inset-bottom)]` keeps the "AndrusAI
          v0.1" footer above it. */}
      <aside
        className={`fixed lg:static inset-y-0 left-0 z-30 w-56 flex flex-col bg-[#111820] border-r border-[#1e2738] transition-transform duration-200 pt-[env(safe-area-inset-top)] pb-[env(safe-area-inset-bottom)] ${
          sidebarOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'
        }`}
      >
        {/* Logo */}
        <div className="flex items-center gap-3 px-4 py-4 border-b border-[#1e2738]">
          <div className="w-8 h-8 rounded-lg bg-[#60a5fa]/20 border border-[#60a5fa]/40 flex items-center justify-center text-sm">
            🤖
          </div>
          <div>
            <div className="text-sm font-semibold text-[#e2e8f0]">BotArmy</div>
            <div className="text-xs text-[#7a8599]">Control Plane</div>
          </div>
        </div>

        {/* Nav */}
        <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.exact}
              onClick={() => setSidebarOpen(false)}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors ${
                  isActive
                    ? 'bg-[#60a5fa]/10 text-[#60a5fa] border border-[#60a5fa]/20'
                    : 'text-[#7a8599] hover:text-[#e2e8f0] hover:bg-[#1e2738]'
                }`
              }
            >
              <span className="text-base leading-none">{item.icon}</span>
              <span>{item.label}</span>
            </NavLink>
          ))}
        </nav>

        {/* Footer */}
        <div className="px-4 py-3 border-t border-[#1e2738]">
          <div className="text-xs text-[#7a8599]">AndrusAI v0.1</div>
        </div>
      </aside>

      {/* Main area */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {/* Header
            iOS PWA safe-area: ``pt-[calc(0.75rem+env(safe-area-inset-top))]``
            preserves the existing 0.75rem (py-3) padding AND adds the
            status-bar inset on top, so the hamburger button + workspace
            dropdown + status pill aren't covered by the iPhone clock /
            signal / battery indicators in standalone-PWA mode. */}
        <header className="flex items-center justify-between px-4 pb-3 pt-[calc(0.75rem+env(safe-area-inset-top))] bg-[#111820] border-b border-[#1e2738] flex-shrink-0">
          <div className="flex items-center gap-3">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="lg:hidden p-1.5 rounded text-[#7a8599] hover:text-[#e2e8f0] hover:bg-[#1e2738]"
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
            <ProjectSwitcher />
          </div>
          <div className="flex items-center gap-3">
            <span className={`text-xs px-2 py-0.5 rounded-full font-medium capitalize ${statusColor}`}>
              {statusLabel}
            </span>
          </div>
        </header>

        {/* Page content
            ``pb-[calc(1rem+env(safe-area-inset-bottom))]`` keeps the
            scroll target above the iOS home indicator on devices with
            no physical home button. */}
        <main className="flex-1 overflow-y-auto p-4 lg:p-6 pb-[calc(1rem+env(safe-area-inset-bottom))]">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
