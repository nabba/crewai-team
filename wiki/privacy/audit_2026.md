# Annual privacy audit — 2026

_Composed at 2026-05-16T19:49:20.465028+00:00 by `app.privacy.annual_review.compose_review`._

This audit enumerates every data source the system actively
reads, its current on/off state, and where the data lives.
It is observational only — no source is disabled by this
process. The operator reviews and decides.

## Sources by category

### Browse

### Browser-history ingestion

- **Purpose**: interest-signal modality
- **Retention**: workspace/browse/events/<day>.jsonl (canon URLs, no queries/fragments); blocklist applied at read
- **State**: `?` (gated by `browse_ingestion_enabled`)

### Browse LLM topic clustering

- **Purpose**: daily theme extraction from titles
- **Retention**: workspace/browse/topics/<day>.json — clustered themes only; titles never sent to LLM without redaction
- **State**: `?` (gated by `browse_llm_topics_enabled`)

### Google

### Google Workspace (Gmail/Calendar/Docs/Sheets/Slides/Drive)

- **Purpose**: calendar, email, document access
- **Retention**: OAuth refresh token at workspace/google_token.json; content fetched on-demand, not cached
- **State**: `always-on`

### Health

### Apple Health ingestion

- **Purpose**: personal health data (HR, sleep, steps, body mass)
- **Retention**: workspace/health/<kind>.jsonl — per-kind typed records; NEVER leaves the host (no ChromaDB, no LLM over raw records)
- **State**: `?` (gated by `health_ingestion_enabled`)

### Inbox

### Inbox multi-modal ingestion

- **Purpose**: file-drop watcher (PDFs, images, audio, spreadsheets, YouTube links)
- **Retention**: workspace/inbox/ + per-handler outputs (notes/, finance/, etc.)
- **State**: `?` (gated by `inbox_ingestion_enabled`)

### Internal

### Affect trace

- **Purpose**: internal welfare signal
- **Retention**: workspace/affect/trace.jsonl (rolled monthly) — emotional state INFERRED ABOUT THE SYSTEM, not about the operator
- **State**: `always-on`

### Messaging

### Signal messages

- **Purpose**: primary operator command surface
- **Retention**: workspace/audit.log (request_received + response_sent rows; ts + sender + length only, NEVER content)
- **State**: `always-on`

### Conversation history

- **Purpose**: context retention for follow-up questions
- **Retention**: workspace/conversation_store.db (SQLite); content stored; retention policy operator-managed
- **State**: `always-on`

### Person

### Person correlation L1 (presence)

- **Purpose**: track how often people appear in operator's inputs
- **Retention**: workspace/companion/person_model/* — counts only, no message bodies
- **State**: `DISABLED` (gated by `person_correlation_enabled`)

### Person correlation L2-L4 (centrality, social graph)

- **Purpose**: social-graph features + suggestions
- **Retention**: workspace/companion/person_*.json — opt-in cascade; typed-phrase gates
- **State**: `DISABLED` (gated by `person_correlation_social_graph_enabled`)

### Travel

### Travel (TripIt iCal)

- **Purpose**: flight/ferry/hotel awareness for briefing
- **Retention**: workspace/life_companion/travel_state.json — segment summaries only
- **State**: `always-on`

### Voice

### Voice transcripts (Signal audio attachments)

- **Purpose**: voice-mode input
- **Retention**: ephemeral — STT result becomes text on the audit surface; raw audio not persisted past the request
- **State**: `ENABLED` (gated by `voice_mode`)

## Year delta

**New sources since previous audit:**
  - `Affect trace`
  - `Apple Health ingestion`
  - `Browse LLM topic clustering`
  - `Browser-history ingestion`
  - `Conversation history`
  - `Google Workspace (Gmail/Calendar/Docs/Sheets/Slides/Drive)`
  - `Inbox multi-modal ingestion`
  - `Person correlation L1 (presence)`
  - `Person correlation L2-L4 (centrality, social graph)`
  - `Signal messages`
  - `Travel (TripIt iCal)`
  - `Voice transcripts (Signal audio attachments)`

## Policy events this year

No continuity-ledger ``*_policy`` events were emitted this year. (This usually means no master switches were flipped — review the per-source states above for the current envelope.)

## Operator next steps

  1. Confirm every ENABLED source above is still serving a purpose.
  2. For any source you no longer need, flip its master switch and (if applicable) run the source's `forget` path.
  3. Save this file to git so the year-over-year delta on the next audit is meaningful.
