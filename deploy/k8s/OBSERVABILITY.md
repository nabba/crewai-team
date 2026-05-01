# BotArmy observability

Hybrid stack — open-source for app metrics (portable across clouds), cloud-native
for infra metrics that come free.

## What gets installed

When `enable_monitoring = true` in the Terraform tfvars (default):

```
┌───────────────────────────────────────────────────────────────┐
│  in-cluster (kube-prometheus-stack)                           │
│                                                               │
│  ┌──────────┐    ┌──────────────┐   ┌──────────────┐         │
│  │ gateway  │───▶│ ServiceMonitor│──▶│  Prometheus  │         │
│  │ /metrics │    │  (BotArmy)    │   │   (7d TSDB)  │         │
│  └──────────┘    └──────────────┘   └──────┬───────┘         │
│                                            ▼                  │
│  ┌──────────────┐                   ┌──────────────┐         │
│  │PrometheusRule│──── alerts ──────▶│ Alertmanager │         │
│  │  (BotArmy)   │                   └──────────────┘         │
│  └──────────────┘                                             │
│                                                               │
│  ┌──────────────────────────────┐    ┌──────────────┐        │
│  │ ConfigMap: BotArmy dashboard │───▶│   Grafana    │        │
│  │ (grafana_dashboard=1 label)  │    │ (auto-loads) │        │
│  └──────────────────────────────┘    └──────────────┘        │
└───────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  cloud-native (free metrics that Prometheus can't see)          │
│                                                                 │
│   AWS:  CloudWatch dashboard                                    │
│         · RDS CPU / connections / IOPS                          │
│         · NAT GW egress bytes                                   │
│                                                                 │
│   GCP:  Cloud Monitoring dashboard                              │
│         · Cloud SQL CPU / connections / disk                    │
│         · GKE container CPU                                     │
└─────────────────────────────────────────────────────────────────┘
```

## What ships in the chart

| Resource | Purpose | Toggle |
| --- | --- | --- |
| `ServiceMonitor` for the gateway | Tells Prometheus to scrape `/metrics` | `monitoring.serviceMonitor.enabled` |
| `PrometheusRule` with 7 alerts | Gateway down, high 5xx rate, OOM, dependencies down, LLM cascade exhausted, p95 latency | `monitoring.prometheusRule.enabled` |
| `ConfigMap` with the BotArmy dashboard | Auto-loaded by Grafana sidecar (`grafana_dashboard=1`) | `monitoring.dashboards.enabled` |

All three default to **off** so a vanilla local install pays no observability cost.
Cloud installs flip them on automatically when `enable_monitoring = true`.

## The dashboard

`dashboards/botarmy-overview.json` — three sections:

1. **Gateway** — pods up, request rate, 5xx rate, p95 latency, memory used / limit
2. **LLM cascade** — requests by tier (local / frontier / premium), p95 latency by tier
3. **Memory backends** — Postgres / Neo4j / ChromaDB up indicators

The queries are written to match the labels Prometheus generates from the
`ServiceMonitor` we ship. If you fork the chart and rename things, you'll need
to update the `job=~".*botarmy.*-..."` patterns.

## Alerts

| Alert | Severity | Trigger | Why |
| --- | --- | --- | --- |
| `BotArmyGatewayDown` | critical | `up == 0` for 2m | The bot is offline |
| `BotArmyGatewayHighErrorRate` | warning | 5xx rate >5% for 5m | LLM upstream or DB blip |
| `BotArmyGatewayHighMemory` | warning | >85% of memory limit for 10m | Possible leak — gateway will OOMKill soon |
| `BotArmyPostgresDown` | critical | Up == 0 for 3m | Mem0 store unreachable; agent has no memory |
| `BotArmyNeo4jDown` | warning | Up == 0 for 3m | Entity graph offline; reasoning degrades |
| `BotArmyChromaDown` | warning | Up == 0 for 3m | RAG retrieval will fail |
| `BotArmyLLMCascadeAllFailing` | critical | Custom counter increments | All LLM tiers refused — recovery loop exhausted |
| `BotArmyLLMP95Latency` | warning | p95 >30s for 10m | Frontier API throttling, or local Ollama overloaded |

## Gateway metrics — what's published

All metrics consumed by the dashboards and alerts are now emitted from the
gateway code. Endpoint: `GET /metrics` (plain-text exposition format).

| Metric | Type | Source |
| --- | --- | --- |
| `up` | gauge | ✓ free — Prometheus auto-generates from scrape success |
| `http_requests_total{method, handler, status}` | counter | `prometheus-fastapi-instrumentator` (auto, registered in `app/main.py`) |
| `http_request_duration_seconds_bucket{method, handler, le}` | histogram | same |
| `process_*`, `python_*` | various | same |
| `container_memory_working_set_bytes` | gauge | ✓ free — cAdvisor via kubelet |
| `container_spec_memory_limit_bytes` | gauge | ✓ free |
| `llm_requests_total{tier, provider, model, status}` | counter | `app/observability/llm_events.py` — single subscriber on CrewAI's event bus, covers every provider (native + LiteLLM-mediated) |
| `llm_request_duration_seconds_bucket{tier, provider, model, le}` | histogram | same — latency computed by pairing Started/Completed events via `call_id` |
| `llm_cascade_all_tiers_failed_total` | counter | `app/recovery/loop.py` — incremented when every alternative strategy is tried and none succeeds |
| `mem0_postgres_connection_errors_total` | counter | `app/memory/mem0_manager.py` — bumped on init failure + per-call exception in store/search/get_all (NOT on async-pool enqueue errors, which aren't connection errors) |

The metric definitions live in `app/observability/metrics.py`. Adding a new
metric is two lines (define the counter/histogram), one call site
(`.inc()` / `.observe(...)`), and an update to the dashboard JSON if you
want it visualised.

### Tier labelling

The `tier` label on `llm_requests_total` resolves through
`app.llm_catalog.get_tier(model)` — same source of truth the cascade selector
uses. Models not yet in the catalog get `tier="unknown"` so the histogram
keeps its label cardinality bounded even for one-off experiments.

## Alert routing

Alertmanager is installed by Terraform (both AWS and GCP modules) but ships
with no destinations by default — alerts show in the Alertmanager UI but
nothing is paged. Three destinations are supported, mix and match as you like:

| Variable | Effect |
| --- | --- |
| `alertmanager_slack_webhook_url` | Slack incoming-webhook URL. Sets `slack_api_url` globally. |
| `alertmanager_slack_channel` | Channel for routed alerts. Default `#botarmy-alerts`. |
| `alertmanager_email_to` | Email destination (critical only). Triggers email_configs. |
| `alertmanager_smtp_smarthost` | SMTP relay (`smtp.sendgrid.net:587`, etc.). Required if email is set. |
| `alertmanager_smtp_from` / `_auth_username` / `_auth_password` | SMTP auth |
| `alertmanager_opsgenie_api_key` | Opsgenie API integration key. Critical → P1, warning → P3. |
| `alertmanager_opsgenie_api_url` | Defaults to `https://api.opsgenie.com/`. Use `.eu.` for EU accounts. |

Routing tree (configured in `observability.tf`):

```
default → silent (visible in UI only)
├── severity=critical → critical receiver (Slack + email if both set)
│      group_wait: 10s · repeat: 1h    ← page sooner
└── severity=warning  → warning receiver (Slack only)
       group_wait: 30s · repeat: 4h
```

Critical alerts repeat every hour until resolved. Warnings repeat every 4 h.
Both send `send_resolved = true` so the channel sees recovery messages too.

### Setting up Slack

1. Create the channel (`#botarmy-alerts` or whatever you set).
2. Open https://api.slack.com/apps → Create New App → From scratch → "BotArmy Alerts".
3. Activate Incoming Webhooks → Add New Webhook to Workspace → pick the channel.
4. Copy the URL (`https://hooks.slack.com/services/T.../B.../...`).
5. Set `alertmanager_slack_webhook_url` in your tfvars + `terraform apply`.

### Setting up email

Bring-your-own SMTP relay — SendGrid, Postmark, Mailgun, AWS SES. Free tiers
on each. Then set the SMTP variables and apply. Alertmanager only sends email
for `severity = critical` (the routing tree filters out warnings) so this is
a low-volume path.

### Setting up Opsgenie

Recommended for production — Slack is fine for awareness, but Opsgenie does
real on-call rotations + escalation policies + mobile pages.

1. Sign up at https://www.atlassian.com/software/opsgenie (5 free users tier
   available; EU instance at https://www.atlassian.com/software/opsgenie?eu).
2. Settings → Integrations → Add integration → **API**.
3. Name it "BotArmy" → Save Integration → copy the API Key.
4. Set `alertmanager_opsgenie_api_key` in your tfvars + `terraform apply`.
5. EU users: also set `alertmanager_opsgenie_api_url = "https://api.eu.opsgenie.com/"`.

Critical alerts arrive as P1 (paged) with `tags=botarmy,critical`. Warnings
arrive as P3 (queued, no escalation) with `tags=botarmy,warning`. Configure
team policies in Opsgenie to filter on these tags if you want different
on-call rotations per severity.

### Verifying alert flow end-to-end

```bash
# Port-forward Alertmanager
kubectl -n monitoring port-forward svc/kube-prometheus-stack-alertmanager 9093:9093

# Send a synthetic alert
curl -XPOST http://localhost:9093/api/v2/alerts -d '[{
  "labels": { "alertname": "TestAlert", "severity": "critical", "service": "botarmy" },
  "annotations": { "summary": "Test alert", "description": "Verifying routing" },
  "startsAt": "'"$(date -u +"%Y-%m-%dT%H:%M:%SZ")"'"
}]' -H "Content-Type: application/json"

# Should appear in Slack within ~10 s
```

## Accessing Grafana

```bash
# AWS
PASSWORD=$(aws secretsmanager get-secret-value \
    --secret-id botarmy-grafana-admin --query SecretString --output text)

# GCP
PASSWORD=$(gcloud secrets versions access latest --secret botarmy-grafana-admin)

# Either cloud — port-forward for now
kubectl -n monitoring port-forward svc/kube-prometheus-stack-grafana 3000:80
open "http://localhost:3000"   # admin / $PASSWORD
```

For permanent access, add an Ingress targeting the Grafana service. The
ALB / GCE Ingress patterns are the same as the gateway. Out of scope for this
chart pass.

## Cost

Roughly $15/mo extra on either cloud. Breakdown:

- Prometheus PVC: 10 GiB ($1)
- Grafana PVC: 5 GiB (~$1)
- Alertmanager PVC: 5 GiB (~$1)
- Pod compute on Autopilot / EKS: ~$10
- Cloud-native dashboards: free

If you want to cut: lower `monitoring_storage_size`, set `prometheus.retention=3d`,
or swap kube-prometheus-stack for a managed service (AMP on AWS, Managed
Prometheus on GCP).

## Hardening (not done in v1)

1. **Remote write to long-term storage** — current setup keeps 7 d in-cluster.
   For longer retention, add `remoteWrite` to AMP (AWS) / Managed Prometheus (GCP).
2. **PagerDuty integration** — Slack, email, and Opsgenie are wired;
   PagerDuty's `pagerduty_configs` block follows the same pattern in
   `observability.tf` if your on-call uses it instead.
3. **Loki for logs** — kube-prometheus-stack does metrics only. Add Loki +
   Promtail (or Cloud Logging on GCP / CloudWatch Logs on AWS via fluent-bit).
4. **OpenTelemetry traces** — wire the gateway to emit OTLP to a collector,
   then Tempo or X-Ray / Cloud Trace.
5. **Silence routing for known-noisy alerts** — when `BotArmyLLMP95Latency`
   fires every time an Opus 4 cold-start hits, you'll want a maintenance window
   silence. UI lives at `/silences` in the port-forwarded Alertmanager.
