# PWA setup — public HTTPS, dashboard auth, iOS home-screen install

End-to-end recipe to take the React dashboard from "running on
`http://localhost:3100`" to "installed as a home-screen PWA on
your iPhone with Web Push working." Three steps, ~10 minutes.

The chain has hard prerequisites — each step depends on the
previous. iOS Web Push requires a Secure Context, a Secure Context
requires HTTPS (or `localhost`), and HTTPS over the public internet
requires a real cert. Tailscale Funnel handles the cert for free.

---

## Why this is necessary

iOS Safari (and every other modern browser) silently strips
`navigator.serviceWorker` and `window.PushManager` from any page
served over plain HTTP from a non-localhost origin. The Settings
page diagnostic will display:

```
diagnostics: iOS=yes · secure=no · serviceWorker=no · pushManager=no
```

That's not a browser bug; it's the W3C "secure context" rule. The
fix is HTTPS. The cheapest way to get HTTPS for a self-hosted Mac
service is Tailscale Funnel — free Let's-Encrypt certs, public
internet exposure, no port forwarding, no DNS records to manage.

---

## Step 1 — Tailscale Funnel (public HTTPS)

### 1.1 Admin-console toggles (one-time, two clicks)

Sign in at https://login.tailscale.com/admin/ and:

1. **Enable HTTPS Certificates**
   - Open https://login.tailscale.com/admin/dns
   - Confirm **MagicDNS** is on (it usually is — your `*.tail<...>.ts.net` hostname must already resolve).
   - Scroll to **HTTPS Certificates** → click **Enable HTTPS**.
   - Acknowledge the warning that *"your machine names and your tailnet DNS name will be published on a public ledger."*

2. **Enable Funnel for the node**
   - First time only: try the funnel command in step 1.2 below — Tailscale prints a one-shot enable URL like
     `https://login.tailscale.com/f/funnel?node=<id>`. Click it (signed in as the same account); approve.
   - Alternative: edit the policy file at https://login.tailscale.com/admin/acls and add:
     ```json
     "nodeAttrs": [
       { "target": ["autogroup:member"], "attr": ["funnel"] }
     ]
     ```

### 1.2 CLI (run on the Mac that hosts the dashboard)

```bash
# Verify HTTPS works (should print 'Wrote public cert ...').
tailscale cert <your-machine>.<your-tailnet>.ts.net

# Expose the dashboard publicly on https://<machine>.<tailnet>.ts.net/
tailscale funnel --bg --https=443 http://localhost:3100

# Confirm the route is live.
tailscale funnel status
```

The third command should print:

```
# Funnel on:
#     - https://<machine>.<tailnet>.ts.net

https://<machine>.<tailnet>.ts.net (Funnel on)
|-- / proxy http://localhost:3100
```

### 1.3 Smoke test from another machine

```bash
curl -sS -I https://<machine>.<tailnet>.ts.net/cp/
# Should be: HTTP/2 401 (the auth wall — proves HTTPS works)
```

A `401` here is the **expected** result — the auth wall from step 2
is doing its job. If you see `200` instead, dashboard auth isn't
configured (still ok for testing, but don't use Funnel without it).

If you get a connection error, give the cert another minute to
propagate (docs allow up to 10 min).

---

## Step 2 — Dashboard auth (so the Funnel URL isn't fully public)

Tailscale Funnel exposes the URL to the entire internet. Anyone who
guesses the hostname can hit it. Add HTTP Basic Auth + a cookie
login alternative.

### 2.1 Generate credentials

```bash
cd ~/BotArmy/crewai-team
python3 -c "
import secrets, pathlib
p = pathlib.Path('.env')
text = p.read_text() if p.exists() else ''
if 'DASHBOARD_PASS' in text:
    print('already configured')
else:
    pw = secrets.token_urlsafe(24)
    p.write_text(text.rstrip() + f'\nDASHBOARD_USER=andrus\nDASHBOARD_PASS={pw}\n')
    print(f'USER=andrus PASS={pw}')
"
```

Save the printed password — you'll type it once on the iPhone.

### 2.2 Restart the dashboard so the new env loads

```bash
launchctl kickstart -k gui/$(id -u)/com.botarmy.dashboard
```

(If you didn't install the LaunchAgent: kill whatever node is
serving `dashboard/server.mjs` and re-run it.)

### 2.3 Verify the auth wall works

```bash
HOST=https://<machine>.<tailnet>.ts.net
curl -sS -I "$HOST/cp/"                              # should be 401
curl -sS -I -u andrus:<pass> "$HOST/cp/"             # should be 200
curl -sS -I -i "$HOST/cp/login?token=<pass>" | head  # should be 302 + Set-Cookie
```

### What's in front of what

The proxy accepts EITHER auth method:

| Method | Use case |
|---|---|
| `Authorization: Basic ...` | Desktop browsers — Safari saves the dialog credentials in keychain |
| Cookie `dashboard_auth=<HMAC>` | iOS PWA — survives standalone-mode launches that ignore Basic Auth keychain |

`localhost` / `127.0.0.1` / `::1` requests bypass auth entirely so
laptop-localhost dev keeps working.

---

## Step 3 — iPhone home-screen install + Web Push

### 3.1 Set the auth cookie ONCE in Safari

iOS Safari standalone mode shares its cookie jar with regular
Safari, but NOT its Basic-Auth keychain. So we set a cookie first.

In Safari on the iPhone, open:

```
https://<machine>.<tailnet>.ts.net/cp/login?token=<DASHBOARD_PASS>
```

Server responds `302 → /cp/`, sets `dashboard_auth` cookie
(`Max-Age=1y`, `HttpOnly`, `Secure`, `SameSite=Lax`). The dashboard
should load immediately. No Basic-Auth dialog.

### 3.2 Add to Home Screen

From the loaded `/cp/` page in Safari:

1. Tap the **Share** icon (square with up-arrow).
2. Scroll → **Add to Home Screen**.
3. Confirm name (`AndrusAI`).
4. Tap **Add**.

Tap the new home-screen icon. The PWA opens in standalone mode (no
Safari chrome, full-screen). It should load straight to `/cp/` —
the cookie set in 3.1 travels with the standalone scope.

### 3.3 Enable Web Push

1. Inside the PWA, navigate to **Settings**.
2. Find the **PWA notifications** card.
3. The diagnostic line should now read:
   ```
   diagnostics: iOS=yes · secure=yes · standalone=yes · serviceWorker=yes · pushManager=yes
   ```
4. Tap **Enable**. iOS prompts for notification permission.
5. Approve.
6. Tap **Send test** — you should get a notification within 1–2 s.

If any of the diagnostic flags is still `no`, follow the remediation
the Settings page suggests for that specific flag — see "Troubleshooting"
below.

### 3.4 You're done

From now on, every time the gateway notifies the dashboard about a
completed task, an error, or a Signal-routed event, it will ping
your iPhone via Web Push — even if the PWA isn't open.

### 3.5 Dual-device links in Signal approval alerts

Approval-flow Signal alerts (auto-deploy governance requests, Tier-3
amendments) include two clickable links so you can tap the right
one for the device you're holding:

```
📱 iPhone: https://<funnel-host>/cp/...
💻 Mac:    http://<tailnet-host>:3100/cp/...
```

Defaults derive from the Tailscale Funnel hostname that
`app/middleware.py` already trusts in its CORS allowlist, so the
links are clickable out-of-the-box. Override with env vars when
needed:

```bash
export DASHBOARD_PUBLIC_URL="https://andrusai.tail5b289b.ts.net"
export DASHBOARD_MAC_URL="http://plgs-macbook-pro---andrus.tail5b289b.ts.net:3100"
```

Set these in `.env` if your Funnel hostname differs from the default
or your Mac runs the Vite dev server on a non-3100 port.

Same module (`app/dashboard_links.py`) is used for both auto-deploy
alerts and Tier-3 amendment alerts — set once, propagates
everywhere.

---

## Troubleshooting

### `tailscale cert` says "your Tailscale account does not support getting TLS certs"

Step 1.1.1 didn't take. Re-open https://login.tailscale.com/admin/dns,
verify **HTTPS Certificates** is enabled. Wait ~30 seconds and retry.

### `tailscale funnel` says "Funnel is not enabled on your tailnet"

Open the URL Tailscale prints in the same error message — it's a
one-shot Funnel enable URL specific to your node. Click through;
re-run `tailscale funnel --bg --https=443 http://localhost:3100`.

### Dashboard `/cp/` returns 200 from the public URL without auth

`DASHBOARD_USER` / `DASHBOARD_PASS` aren't loaded. Check:

```bash
grep DASHBOARD_ ~/BotArmy/crewai-team/.env
```

If both are present, restart the dashboard:

```bash
launchctl kickstart -k gui/$(id -u)/com.botarmy.dashboard
tail -f ~/.crewai-bridge/dashboard.log
```

The startup log should print `Basic auth: enabled (loopback bypassed)
— user=andrus`.

### Settings page shows `secure=yes` but `standalone=no`

You're in a Safari tab, not the home-screen icon. Close Safari, tap
the AndrusAI icon on the home screen.

### Settings page shows `standalone=yes` but `pushManager=no`

iOS < 16.4. Update iOS in **Settings → General → Software Update**,
then delete + re-add the home-screen icon.

### "Authentication required." plain page on PWA launch

The cookie expired or was wiped. Open Safari (not the home-screen
icon), visit `/cp/login?token=<DASHBOARD_PASS>` once, then re-launch
the home-screen icon.

### Rotating the password

Edit `DASHBOARD_PASS` in `.env`, run:

```bash
launchctl kickstart -k gui/$(id -u)/com.botarmy.dashboard
```

Every existing cookie is invalidated automatically (cookie value is
`HMAC(pass, "dashboard-auth-v1")`). Visit `/cp/login?token=<new pass>`
once on each device to re-mint.

### Cert renewal

Tailscale renews automatically. If you ever need to force-refresh:

```bash
tailscale cert <machine>.<tailnet>.ts.net
```

---

## What touches what (for future maintenance)

```
Internet
  ↓ HTTPS, Let's-Encrypt cert via Tailscale
Tailscale Funnel   ────────────────  no app code, just `tailscale funnel`
  ↓ HTTP, plain
Dashboard server (Node, port 3100) ──  dashboard/server.mjs
  │  Basic Auth OR cookie auth in front
  │  GATEWAY_SECRET injected on outbound
  ↓ HTTP localhost, with auth header
Gateway (uvicorn, port 8765)       ──  app/main.py + control_plane.dashboard_api
  ↓ Postgres / ChromaDB / Neo4j / Redis
```

Three env vars that drive this chain:

| Name | Purpose | File |
|---|---|---|
| `GATEWAY_SECRET` | Bearer token between proxy and gateway | `crewai-team/.env` |
| `DASHBOARD_USER` | HTTP Basic Auth user | `crewai-team/.env` |
| `DASHBOARD_PASS` | HTTP Basic Auth password + cookie HMAC seed | `crewai-team/.env` |

If `DASHBOARD_USER` and `DASHBOARD_PASS` are both empty/unset, the
proxy disables auth entirely (laptop-localhost dev mode). For any
public Funnel deployment, set them.
