// pwa.ts — service worker registration + Web Push subscription helpers.
//
// All functions are idempotent and degrade gracefully when the runtime
// doesn't support the relevant API (older Safari, http-only origins, etc.).

export type PushSubscriptionPayload = {
  endpoint: string;
  keys: { p256dh: string; auth: string };
  userAgent: string;
};

export async function registerServiceWorker(): Promise<ServiceWorkerRegistration | null> {
  if (typeof navigator === 'undefined' || !('serviceWorker' in navigator)) {
    return null;
  }
  try {
    return await navigator.serviceWorker.register('/cp/sw.js', { scope: '/cp/' });
  } catch (err) {
    // Service worker registration is opportunistic — never break the app.
    console.warn('[pwa] service worker registration failed', err);
    return null;
  }
}

export async function getPushSubscription(): Promise<PushSubscription | null> {
  if (typeof navigator === 'undefined' || !('serviceWorker' in navigator)) return null;
  const reg = await navigator.serviceWorker.getRegistration('/cp/');
  return reg?.pushManager.getSubscription() ?? null;
}

export async function subscribeToPush(vapidPublicKey: string): Promise<PushSubscriptionPayload | null> {
  if (typeof navigator === 'undefined' || !('serviceWorker' in navigator)) return null;
  if (typeof Notification === 'undefined' || !('PushManager' in window)) return null;

  const reg = await navigator.serviceWorker.ready;
  // Permission: must be granted before pushManager.subscribe will succeed.
  const perm = await Notification.requestPermission();
  if (perm !== 'granted') return null;

  // Reuse an existing subscription if it already matches the current VAPID key.
  const existing = await reg.pushManager.getSubscription();
  if (existing) await existing.unsubscribe();

  // Cast through BufferSource so TS doesn't complain about the
  // Uint8Array<ArrayBufferLike> vs ArrayBuffer variance.
  const sub = await reg.pushManager.subscribe({
    userVisibleOnly: true,
    applicationServerKey: urlBase64ToUint8Array(vapidPublicKey) as BufferSource,
  });

  return toPayload(sub);
}

export async function unsubscribeFromPush(): Promise<boolean> {
  const sub = await getPushSubscription();
  if (!sub) return false;
  return sub.unsubscribe();
}

function toPayload(sub: PushSubscription): PushSubscriptionPayload {
  const json = sub.toJSON() as { endpoint: string; keys?: Record<string, string> };
  return {
    endpoint: json.endpoint,
    keys: {
      p256dh: json.keys?.p256dh ?? '',
      auth: json.keys?.auth ?? '',
    },
    userAgent: navigator.userAgent,
  };
}

// VAPID keys are base64url-encoded; the browser API needs a Uint8Array.
function urlBase64ToUint8Array(b64: string): Uint8Array {
  const padding = '='.repeat((4 - (b64.length % 4)) % 4);
  const base64 = (b64 + padding).replace(/-/g, '+').replace(/_/g, '/');
  const raw = atob(base64);
  const out = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; ++i) out[i] = raw.charCodeAt(i);
  return out;
}
