/* AndrusAI Control Plane — service worker.
 *
 * Two caching strategies:
 *   - cache-first for static assets (CSS / JS bundles, images, fonts) so the
 *     PWA boots offline and a cold reload is instant.
 *   - network-first for API calls under /api/, /config/, /epistemic/, /affect/
 *     so live data wins, with a stale fallback when offline.
 *
 * Push events deliver Web Push notifications and route the click to a
 * deep-link inside the app (default: '/cp/').
 *
 * Versioning: bump CACHE_VERSION whenever we ship a major asset change so
 * old caches get pruned on the next activation.
 */

const CACHE_VERSION = 'v1';
const STATIC_CACHE  = `andrusai-static-${CACHE_VERSION}`;
const RUNTIME_CACHE = `andrusai-runtime-${CACHE_VERSION}`;

// Files to pre-cache on install — keep this minimal so first install is fast.
const PRECACHE_URLS = [
  '/cp/',
  '/cp/manifest.webmanifest',
  '/cp/favicon.svg',
  '/cp/icon-192.png',
  '/cp/icon-512.png',
  '/cp/apple-touch-icon.png',
];

const API_PATHS = [
  '/api/',
  '/config/',
  '/epistemic/',
  '/affect/',
  '/kb/',
  '/fiction/',
  '/philosophy/',
  '/episteme/',
  '/experiential/',
  '/aesthetics/',
  '/tensions/',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(STATIC_CACHE).then((cache) =>
      // Use addAll loosely — if one URL 404s the install still succeeds with the rest.
      Promise.all(PRECACHE_URLS.map((u) =>
        fetch(u, { cache: 'reload' }).then((resp) => {
          if (resp.ok) return cache.put(u, resp);
        }).catch(() => {})
      )),
    ).then(() => self.skipWaiting()),
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    const keys = await caches.keys();
    await Promise.all(
      keys
        .filter((k) => ![STATIC_CACHE, RUNTIME_CACHE].includes(k))
        .map((k) => caches.delete(k)),
    );
    await self.clients.claim();
  })());
});

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;

  // Network-first for API paths
  if (API_PATHS.some((p) => url.pathname.startsWith(p))) {
    event.respondWith(networkFirst(req));
    return;
  }

  // Cache-first for static assets under /cp/
  if (url.pathname.startsWith('/cp/')) {
    event.respondWith(cacheFirst(req));
    return;
  }
});

async function cacheFirst(request) {
  const cache = await caches.open(RUNTIME_CACHE);
  const hit = await cache.match(request);
  if (hit) return hit;
  try {
    const resp = await fetch(request);
    if (resp.ok) cache.put(request, resp.clone()).catch(() => {});
    return resp;
  } catch (err) {
    // If we have nothing, fall back to the precached shell.
    return (await caches.match('/cp/')) || new Response('Offline', { status: 503 });
  }
}

async function networkFirst(request) {
  const cache = await caches.open(RUNTIME_CACHE);
  try {
    const resp = await fetch(request);
    if (resp.ok) cache.put(request, resp.clone()).catch(() => {});
    return resp;
  } catch (err) {
    const stale = await cache.match(request);
    if (stale) return stale;
    return new Response(JSON.stringify({ offline: true }), {
      status: 503,
      headers: { 'Content-Type': 'application/json' },
    });
  }
}

// ── Web Push ──────────────────────────────────────────────────────────────

self.addEventListener('push', (event) => {
  let payload = { title: 'AndrusAI', body: '' };
  if (event.data) {
    try { payload = { ...payload, ...event.data.json() }; }
    catch { payload.body = event.data.text(); }
  }
  const { title, body, url, tag, icon } = payload;
  event.waitUntil(self.registration.showNotification(title || 'AndrusAI', {
    body: body || '',
    icon: icon || '/cp/icon-192.png',
    badge: '/cp/icon-192.png',
    tag: tag || 'andrusai-default',
    data: { url: url || '/cp/' },
    requireInteraction: false,
  }));
});

self.addEventListener('notificationclick', (event) => {
  event.notification.close();
  const targetUrl = (event.notification.data && event.notification.data.url) || '/cp/';
  event.waitUntil((async () => {
    const allClients = await self.clients.matchAll({ type: 'window', includeUncontrolled: true });
    for (const c of allClients) {
      if (c.url.endsWith(targetUrl) && 'focus' in c) return c.focus();
    }
    if (self.clients.openWindow) await self.clients.openWindow(targetUrl);
  })());
});
