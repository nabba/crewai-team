/**
 * Lightweight dashboard server: static files + API proxy.
 * Serves the React build from serve-root/ and proxies backend routes to the gateway.
 * This eliminates CORS issues (same-origin requests).
 */
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

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
});
