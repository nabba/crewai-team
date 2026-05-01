// Same-origin API calls — dashboard server proxies backend paths to the gateway.
// Callers pass the full absolute path (e.g. "/api/cp/tickets", "/kb/status",
// "/config/creative_mode"). The proxy handles forwarding to the FastAPI gateway.
//
// Phase B2: when the gateway is configured to enforce auth (helm sets
// GATEWAY_AUTH_REQUIRED=1), every request must carry
// `Authorization: Bearer <gateway-secret>`. We read the secret from the
// build-time env var VITE_GATEWAY_SECRET; on a laptop dev setup with
// auth enforcement off, the variable is unset and the header is omitted.
// Both the request side AND the gateway side are dev-default permissive,
// so existing local workflows are unaffected.

export class ApiError extends Error {
  status: number;
  body: string;
  constructor(status: number, body: string) {
    super(`API ${status}: ${body.slice(0, 200)}`);
    this.status = status;
    this.body = body;
  }
}

function authHeader(): Record<string, string> {
  // Vite exposes only env vars prefixed with VITE_. The secret is injected
  // at build time by Helm in K8s; on laptop dev it is typically unset.
  const secret = (import.meta as { env?: { VITE_GATEWAY_SECRET?: string } })
    .env?.VITE_GATEWAY_SECRET;
  return secret ? { Authorization: `Bearer ${secret}` } : {};
}

export async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const init: RequestInit = {
    ...options,
    headers: {
      Accept: 'application/json',
      ...(options?.body && !(options.body instanceof FormData)
        ? { 'Content-Type': 'application/json' }
        : {}),
      ...authHeader(),
      ...(options?.headers || {}),
    },
  };
  const res = await fetch(path, init);
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new ApiError(res.status, text);
  }
  if (res.status === 204) return undefined as unknown as T;
  return res.json();
}
