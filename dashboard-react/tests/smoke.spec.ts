import { test, expect, type Page, type Route } from '@playwright/test';

const BACKEND_PREFIXES = ['/api/', '/config/', '/kb/', '/fiction/', '/philosophy/', '/episteme/', '/experiential/', '/aesthetics/', '/tensions/'];

// Mock backend so tests don't require the FastAPI gateway.
// Anchor on pathname.startsWith(...) so we don't swallow Vite module URLs like /cp/src/api/queryClient.ts.
async function stubBackend(page: Page) {
  await page.route((url) => BACKEND_PREFIXES.some((p) => url.pathname.startsWith(p)), (route: Route) => fulfill(route, { status: 200 }));
}

function fulfill(route: Route, opts: { status: number }) {
  const url = route.request().url();
  let body: unknown = [];
  if (url.includes('/projects')) body = [{ id: 'p1', name: 'Test Project', is_active: true, created_at: new Date().toISOString() }];
  else if (url.includes('/tickets/board')) body = { board: { todo: [], in_progress: [], review: [], done: [], failed: [], blocked: [] }, counts: {}, total: 0 };
  else if (url.includes('/tickets')) body = [];
  else if (url.includes('/budgets')) body = [];
  else if (url.includes('/audit')) body = [];
  else if (url.includes('/governance/pending')) body = [];
  else if (url.includes('/org-chart')) body = [];
  else if (url.includes('/costs/daily')) body = [];
  else if (url.includes('/costs/by-agent')) body = { by_actor: [], total_cost: 0 };
  else if (url.includes('/health')) body = { status: 'ok', tickets_total: 0, audit_entries: 0, governance_pending: 0 };
  else if (url.includes('/consciousness')) body = { latest: {}, history: [], updated_at: null };
  else if (url.includes('/tasks')) body = {
    tasks: [], crews: [], agents: [],
    updated_at: new Date().toISOString(), error: null,
  };
  else if (url.includes('/errors')) body = {
    recent: [], patterns: {}, total_recent: 0,
    updated_at: new Date().toISOString(), error: null,
  };
  else if (url.includes('/anomalies')) body = {
    recent_alerts: [], total: 0,
    updated_at: new Date().toISOString(), error: null,
  };
  else if (url.includes('/deploys')) body = {
    recent: [], auto_deploy_enabled: false,
    updated_at: new Date().toISOString(), error: null,
  };
  else if (url.includes('/tech-radar')) body = {
    discoveries: [], updated_at: new Date().toISOString(), error: null,
  };
  else if (url.includes('/llms/discovery')) body = {
    discovered: [], updated_at: new Date().toISOString(), error: null,
  };
  else if (url.includes('/llms/catalog')) body = {
    models: [], role_assignments: {}, cost_mode: 'balanced',
    updated_at: new Date().toISOString(), error: null,
  };
  else if (url.includes('/llms/roles')) body = {
    assignments: [], updated_at: new Date().toISOString(), error: null,
  };
  else if (url.includes('/evolution/variants')) body = { variants: [], drift_score: 0 };
  else if (url.includes('/notes/roots')) body = {
    roots: [
      { name: 'docs', path: '/tmp/docs' },
      { name: 'wiki', path: '/tmp/wiki' },
    ],
    default_root: 'docs',
  };
  else if (url.includes('/notes/tree')) body = { root: 'docs', tree: { name: 'docs', path: '', type: 'dir', children: [] } };
  else if (url.includes('/notes/graph')) body = { root: 'docs', nodes: [], edges: [], tags: [], updated_at: new Date().toISOString() };
  else if (url.includes('/notes/tags')) body = { root: 'docs', tags: [] };
  else if (url.includes('/notes/search')) body = { query: '', hits: [], total: 0 };
  else if (url.includes('/notes/file')) body = {
    root: 'docs', path: 'x.md', title: 'X', frontmatter: {}, body: '# X', size: 1, mtime: Date.now() / 1000,
    backlinks: [], forward_links: [], tags: [], updated_at: new Date().toISOString(),
  };
  else if (url.includes('/tokens')) body = {
    stats: { hour: [], day: [], week: [], month: [], year: [] },
    request_costs: {
      day: { requests: 0, total_cost_usd: 0, avg_cost_usd: 0, avg_calls: 0, avg_tokens: 0 },
      week: { requests: 0, total_cost_usd: 0, avg_cost_usd: 0, avg_calls: 0, avg_tokens: 0 },
      month: { requests: 0, total_cost_usd: 0, avg_cost_usd: 0, avg_calls: 0, avg_tokens: 0 },
    },
    by_crew: { day: [] },
    projection: { day_cost_usd: 0, mtd_cost_usd: 0, projected_monthly_usd: 0 },
    updated_at: new Date().toISOString(),
  };
  else if (url.includes('/evolution/summary')) body = {
    total_experiments: 0, kept: 0, discarded: 0, crashed: 0, kept_ratio: 0,
    best_score: 0, current_score: 0, score_trend: [],
    current_engine: 'avo', subia_safety: 0.9, engines: {},
  };
  else if (url.includes('/evolution/results')) body = { results: [] };
  else if (url.includes('/evolution/engine')) body = { config_mode: 'auto', selected_engine: 'avo', shinka_available: false };
  else if (url.includes('/api/workspaces/meta')) body = { meta_workspace: {}, by_project: {}, project_count: 0 };
  else if (url.includes('/workspaces') && url.includes('/items')) body = { project_id: 'test', active: [], peripheral: [], capacity: 3, cycle: 0 };
  else if (url.includes('/workspaces')) body = { workspaces: [], count: 0 };
  else if (url.includes('/creative_mode')) body = { creative_run_budget_usd: 1, originality_wiki_weight: 0.6, mem0_weight: 0.4 };
  else if (url.includes('/llm_mode')) body = { mode: 'hybrid', valid_modes: ['local','free','cloud','hybrid','insane','anthropic'] };
  else if (url.includes('/kb/businesses')) body = { businesses: [] };
  else if (url.includes('/kb/status')) body = { collection_name: 'kb', total_chunks: 0 };
  else if (url.includes('/brainstorm/techniques')) body = [
    { name: 'scamper', title: 'SCAMPER', description: 'apply 7 lenses', total_steps: 7 },
    { name: 'six_hats', title: 'Six Thinking Hats', description: 'six frames', total_steps: 7 },
  ];
  else if (url.includes('/brainstorm/sessions/active')) body = { session: null };
  else if (url.includes('/brainstorm/sessions')) body = [];
  else if (url.includes('/stats') || url.includes('/status')) body = { collection_name: 'x', total_chunks: 0 };

  return route.fulfill({
    status: opts.status,
    contentType: 'application/json',
    body: JSON.stringify(body),
  });
}

// Paths are relative to baseURL (http://localhost:5173/cp/). A leading slash
// would strip the /cp/ prefix and land outside the SPA mount point.
const ROUTES: { path: string; heading: string | RegExp }[] = [
  { path: '', heading: /Dashboard/ },
  { path: 'tickets', heading: /Tickets/ },
  { path: 'tasks', heading: /Crew Activity/ },
  { path: 'budgets', heading: /Budgets/ },
  { path: 'audit', heading: /Audit Feed/ },
  { path: 'governance', heading: /Governance Queue/ },
  { path: 'org-chart', heading: /Org Chart/ },
  { path: 'costs', heading: /Cost Analytics/ },
  { path: 'workspaces', heading: /Consciousness Workspaces/ },
  { path: 'evolution', heading: /Evolution Monitor/ },
  { path: 'ops', heading: /Operations/ },
  { path: 'llms', heading: /LLMs/ },
  { path: 'knowledge', heading: /Knowledge Bases/ },
  { path: 'notes', heading: /^Notes$/ },
  { path: 'wiki', heading: /Knowledge Wiki/ },
  { path: 'brainstorm', heading: /^Brainstorm$/ },
];

for (const { path, heading } of ROUTES) {
  test(`renders /${path}`, async ({ page }) => {
    await stubBackend(page);
    await page.goto(path);
    await expect(page.getByRole('heading', { name: heading }).first()).toBeVisible({ timeout: 10_000 });
  });
}

test('unknown route shows NotFound', async ({ page }) => {
  await stubBackend(page);
  await page.goto('does-not-exist');
  await expect(page.getByText('Page not found')).toBeVisible();
});
