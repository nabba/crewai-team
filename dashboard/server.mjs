/**
 * Lightweight dashboard server: static files + API proxy.
 * Serves the React build from serve-root/ and proxies backend routes to the gateway.
 * This eliminates CORS issues (same-origin requests).
 */
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHmac, timingSafeEqual } from 'node:crypto';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const STATIC_ROOT = path.join(__dirname, 'serve-root');
const GATEWAY_HOST = '127.0.0.1';
const GATEWAY_PORT = 8765;
const PORT = parseInt(process.env.PORT || '3100');

// Load GATEWAY_SECRET from the sibling .env if the LaunchAgent / shell
// didn't set it in the process environment. Without it every write to
// /config/* and /budgets/override fails with 401 because the gateway
// expects `Authorization: Bearer <secret>`.
function loadGatewaySecret() {
  if (process.env.GATEWAY_SECRET) return process.env.GATEWAY_SECRET;
  const envPath = path.join(__dirname, '..', '.env');
  try {
    const text = fs.readFileSync(envPath, 'utf8');
    const match = text.match(/^\s*GATEWAY_SECRET\s*=\s*(.+?)\s*$/m);
    if (match) {
      // Strip surrounding quotes if any.
      return match[1].replace(/^(['"])(.*)\1$/, '$2');
    }
  } catch {
    // .env missing or unreadable — fall through to empty string.
  }
  return '';
}

const GATEWAY_SECRET = loadGatewaySecret();

// Optional HTTP Basic Auth in front of the dashboard. When DASHBOARD_USER
// and DASHBOARD_PASS are set (env or sibling .env) every request must
// carry a matching `Authorization: Basic ...` header. When EITHER is
// empty, auth is disabled (preserves laptop-localhost dev ergonomics).
//
// Use case: restricting access to the Tailscale Funnel public HTTPS
// route. Once the browser remembers the credentials per-origin, the
// home-screen PWA inherits them transparently.
function loadEnvVar(name) {
  if (process.env[name]) return process.env[name];
  const envPath = path.join(__dirname, '..', '.env');
  try {
    const text = fs.readFileSync(envPath, 'utf8');
    const re = new RegExp(`^\\s*${name}\\s*=\\s*(.+?)\\s*$`, 'm');
    const match = text.match(re);
    if (match) return match[1].replace(/^(['"])(.*)\1$/, '$2');
  } catch {/* fall through */}
  return '';
}

const DASHBOARD_USER = loadEnvVar('DASHBOARD_USER');
const DASHBOARD_PASS = loadEnvVar('DASHBOARD_PASS');
const AUTH_ENABLED = Boolean(DASHBOARD_USER && DASHBOARD_PASS);

// Hosts that always bypass auth — local development on the same Mac.
const LOCAL_HOSTS = new Set(['localhost', '127.0.0.1', '::1']);

function clientIsLoopback(req) {
  const host = (req.headers.host || '').split(':')[0].replace(/^\[|\]$/g, '');
  if (LOCAL_HOSTS.has(host)) {
    const remote = (req.socket?.remoteAddress || '').replace(/^::ffff:/, '');
    if (LOCAL_HOSTS.has(remote)) return true;
  }
  return false;
}

function checkBasicAuth(req) {
  const header = req.headers.authorization || '';
  if (!header.startsWith('Basic ')) return false;
  let decoded;
  try {
    decoded = Buffer.from(header.slice(6), 'base64').toString('utf8');
  } catch {
    return false;
  }
  const idx = decoded.indexOf(':');
  if (idx < 0) return false;
  const user = decoded.slice(0, idx);
  const pass = decoded.slice(idx + 1);
  // Constant-time compare to defang timing attacks. Buffers must match
  // length, so pre-pad the supplied side to the expected length.
  const ub = Buffer.from(user);
  const pb = Buffer.from(pass);
  const eu = Buffer.from(DASHBOARD_USER);
  const ep = Buffer.from(DASHBOARD_PASS);
  if (ub.length !== eu.length || pb.length !== ep.length) return false;
  return timingSafeEqual(ub, eu) && timingSafeEqual(pb, ep);
}

// Long-lived cookie token derived from DASHBOARD_PASS. Opaque
// (HMAC), regenerable from the same .env, no server-side state.
// Rotating the password invalidates every existing cookie.
const COOKIE_NAME = 'dashboard_auth';
const COOKIE_TOKEN = AUTH_ENABLED
  ? createHmac('sha256', DASHBOARD_PASS).update('dashboard-auth-v1').digest('hex')
  : '';

function parseCookies(header) {
  const out = {};
  if (!header) return out;
  for (const part of header.split(';')) {
    const eq = part.indexOf('=');
    if (eq < 0) continue;
    const k = part.slice(0, eq).trim();
    const v = part.slice(eq + 1).trim();
    if (k) out[k] = decodeURIComponent(v);
  }
  return out;
}

function checkCookieAuth(req) {
  if (!COOKIE_TOKEN) return false;
  const cookies = parseCookies(req.headers.cookie || '');
  const got = cookies[COOKIE_NAME];
  if (!got || got.length !== COOKIE_TOKEN.length) return false;
  return timingSafeEqual(Buffer.from(got), Buffer.from(COOKIE_TOKEN));
}

// One-shot login: GET /cp/login?token=<DASHBOARD_PASS>
//   match    → 302 /cp/ + Set-Cookie (Max-Age=1y, HttpOnly, Secure, SameSite=Lax)
//   mismatch → 401
// Used by iOS Safari standalone PWA where the Basic-Auth keychain
// from the regular browser tab isn't always inherited by the
// home-screen icon.  Visit this URL once; the cookie travels with
// the standalone scope on every subsequent launch.
function handleLogin(req, res) {
  if (!AUTH_ENABLED) {
    res.writeHead(204);
    res.end();
    return;
  }
  const url = new URL(req.url, 'http://x');
  const token = url.searchParams.get('token') || '';
  const ok =
    token.length === DASHBOARD_PASS.length &&
    timingSafeEqual(Buffer.from(token), Buffer.from(DASHBOARD_PASS));
  if (!ok) {
    res.writeHead(401, { 'Content-Type': 'text/plain' });
    res.end('Login token mismatch.\n');
    return;
  }
  const oneYear = 60 * 60 * 24 * 365;
  res.writeHead(302, {
    'Set-Cookie':
      `${COOKIE_NAME}=${COOKIE_TOKEN}; Max-Age=${oneYear}; Path=/; HttpOnly; Secure; SameSite=Lax`,
    Location: '/cp/',
  });
  res.end();
}

function requireAuth(req, res) {
  if (!AUTH_ENABLED) return true;
  if (clientIsLoopback(req)) return true;
  if (checkCookieAuth(req)) return true;
  if (checkBasicAuth(req)) return true;
  res.writeHead(401, {
    'WWW-Authenticate': 'Basic realm="AndrusAI Control Plane", charset="UTF-8"',
    'Content-Type': 'text/plain',
  });
  res.end(
    'Authentication required.\n\n' +
    'iOS PWA users: visit /cp/login?token=YOUR_DASHBOARD_PASS once to set a\n' +
    'long-lived auth cookie that survives standalone-mode launches.\n',
  );
  return false;
}

const BACKEND_PATH_PREFIXES = [
  '/api/',
  '/config/',
  '/kb/',
  '/fiction/',
  '/philosophy/',
  '/episteme/',     // RAG over research KB
  '/epistemic/',    // claim ledger / pushback / overrides — distinct subsystem
  '/experiential/',
  '/aesthetics/',
  '/tensions/',
  '/affect/',       // viability / V/A/C / welfare / reference panel
];

const MIME_TYPES = {
  '.html': 'text/html',
  '.js': 'application/javascript',
  '.css': 'text/css',
  '.json': 'application/json',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.ico': 'image/x-icon',
  '.woff2': 'font/woff2',
  '.woff': 'font/woff',
};

function serveStatic(req, res) {
  let filePath = path.join(STATIC_ROOT, req.url === '/cp/' ? '/cp/index.html' : req.url);

  if (!fs.existsSync(filePath) && req.url.startsWith('/cp/')) {
    filePath = path.join(STATIC_ROOT, 'cp', 'index.html');
  }

  if (!fs.existsSync(filePath)) {
    res.writeHead(404);
    res.end('Not found');
    return;
  }

  const ext = path.extname(filePath);
  const contentType = MIME_TYPES[ext] || 'application/octet-stream';

  // Cache policy:
  //   - index.html MUST always be fresh so browsers pick up new hashed
  //     bundle references on every reload (vite content-hashes the JS
  //     assets into their filenames, so the bundle path changes any time
  //     the content changes — but only if the HTML pointing at it is
  //     re-fetched).
  //   - Content-hashed assets (vite emits e.g. index-B6lzD5Bt.js) are
  //     safe to cache long-term: they're immutable by construction, and
  //     a new build produces a different filename.
  //   - Everything else gets a modest 1-hour cache as a middle ground.
  const isHashed = /\/assets\/[A-Za-z0-9_-]+-[A-Za-z0-9_-]{6,}\.(?:js|css|woff2?|svg|png|ico)$/.test(req.url);
  const isHtml = ext === '.html';
  let cacheControl;
  if (isHtml) {
    cacheControl = 'no-cache, must-revalidate';
  } else if (isHashed) {
    cacheControl = 'public, max-age=31536000, immutable';  // 1 year
  } else {
    cacheControl = 'public, max-age=3600';
  }

  const content = fs.readFileSync(filePath);
  res.writeHead(200, { 'Content-Type': contentType, 'Cache-Control': cacheControl });
  res.end(content);
}

function proxyToGateway(req, res) {
  const headers = { ...req.headers, host: `${GATEWAY_HOST}:${GATEWAY_PORT}` };
  if (GATEWAY_SECRET && !headers.authorization) {
    headers.authorization = `Bearer ${GATEWAY_SECRET}`;
  }

  const options = {
    hostname: GATEWAY_HOST,
    port: GATEWAY_PORT,
    path: req.url,
    method: req.method,
    headers,
  };

  const proxyReq = http.request(options, (proxyRes) => {
    res.writeHead(proxyRes.statusCode, proxyRes.headers);
    proxyRes.pipe(res, { end: true });
  });

  proxyReq.on('error', (err) => {
    console.error(`Proxy error: ${err.message}`);
    res.writeHead(502);
    res.end(JSON.stringify({ error: 'Gateway unavailable' }));
  });

  // Set a timeout (120s for large file uploads up to 20MB)
  proxyReq.setTimeout(120000, () => {
    proxyReq.destroy();
    res.writeHead(504);
    res.end(JSON.stringify({ error: 'Gateway timeout' }));
  });

  req.pipe(proxyReq, { end: true });
}

function isBackendPath(url) {
  for (const prefix of BACKEND_PATH_PREFIXES) {
    if (url.startsWith(prefix)) return true;
  }
  return false;
}

const server = http.createServer((req, res) => {
  // Login route is unauthenticated by design — its only effect is
  // setting the cookie when the right token is supplied.
  const pathOnly = (req.url || '').split('?')[0];
  if (pathOnly === '/cp/login') {
    handleLogin(req, res);
    return;
  }
  if (!requireAuth(req, res)) return;
  if (isBackendPath(req.url)) {
    proxyToGateway(req, res);
    return;
  }
  serveStatic(req, res);
});

server.listen(PORT, () => {
  console.log(`Dashboard server on http://localhost:${PORT}/cp/`);
  console.log(`API proxy → http://${GATEWAY_HOST}:${GATEWAY_PORT} for: ${BACKEND_PATH_PREFIXES.join(', ')}`);
  if (GATEWAY_SECRET) console.log('Gateway secret: injected on outbound proxy requests');
  console.log(
    AUTH_ENABLED
      ? `Basic auth: enabled (loopback bypassed) — user=${DASHBOARD_USER}`
      : 'Basic auth: disabled (set DASHBOARD_USER + DASHBOARD_PASS in .env to enable)',
  );
});
