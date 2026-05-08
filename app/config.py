import re
from pydantic_settings import BaseSettings
from pydantic import ConfigDict, Field, SecretStr, field_validator
from functools import cache

# Hard caps — prevent a misconfigured or tampered .env from granting runaway resources
_MAX_SANDBOX_MEMORY_MB = 2048   # 2 GB ceiling
_MIN_SANDBOX_MEMORY_MB = 32
_MAX_SANDBOX_CPU = 4.0
_MIN_SANDBOX_CPU = 0.05
_MAX_SANDBOX_TIMEOUT = 120      # seconds


class Settings(BaseSettings):
    anthropic_api_key: SecretStr
    brave_api_key: SecretStr

    signal_bot_number: str
    signal_owner_number: str
    signal_cli_path: str = "/opt/homebrew/bin/signal-cli"
    signal_socket_path: str = "/tmp/signal-cli.sock"
    # HTTP endpoint for signal-cli (preferred in Docker — Unix sockets don't work
    # across the Docker Desktop for Mac VM boundary).
    # Start signal-cli with: signal-cli daemon --http 127.0.0.1:7583 --receive-mode on-start
    # Then set SIGNAL_HTTP_URL=http://host.docker.internal:7583
    signal_http_url: str = ""
    # Path where signal-cli stores downloaded attachments.
    # On macOS: ~/.local/share/signal-cli/attachments
    # Mounted read-only into Docker at /app/attachments
    signal_attachment_path: str = ""
    # Host-side path to the workspace directory (for translating Docker paths
    # to host paths when sending file attachments via signal-cli).
    # Docker writes to /app/workspace/output/, signal-cli needs the host path.
    # Example: /Users/andrus/BotArmy/crewai-team/workspace
    workspace_host_path: str = ""

    gateway_secret: SecretStr
    gateway_port: int = 8765
    gateway_bind: str = "127.0.0.1"

    commander_model: str = "claude-opus-4-6"
    specialist_model: str = "claude-sonnet-4-6"

    # ── Multi-tier LLM configuration ──────────────────────────────────
    # Cost mode controls model selection across all roles.
    # "budget"   = minimize cost (DeepSeek V3.2 everywhere, Sonnet for commander)
    # "balanced" = best cost/quality trade-off (mixed models per role)
    # "quality"  = maximize quality (Kimi/Gemini for specialists, Opus for vetting)
    cost_mode: str = "balanced"

    # API tier — frontier models via OpenRouter (DeepSeek, MiniMax, Kimi, GLM, Gemini)
    # Get your key at https://openrouter.ai/keys
    api_tier_enabled: bool = True
    openrouter_api_key: SecretStr = SecretStr("")

    # External LLM rankings — blend third-party leaderboards into the
    # selector's benchmark scores (app/llm_external_ranks.py). The AA
    # fetcher activates only when the key is set; OpenRouter + HF are
    # always available.
    external_ranks_enabled: bool = True
    external_ranks_weight: float = 0.3  # 0.0 = ignore, 1.0 = replace internal
    artificial_analysis_api_key: SecretStr = SecretStr("")

    # LLM mode — unified runtime-mode vocabulary. See app.llm_catalog.RUNTIME_MODES.
    # Accepts ``free`` / ``budget`` / ``balanced`` (default) / ``quality`` /
    # ``insane`` / ``anthropic`` plus legacy aliases (``hybrid``/``local``/
    # ``cloud``) which are normalised at set_mode() time.
    llm_mode: str = "balanced"

    sandbox_image: str = "crewai-sandbox:latest"
    sandbox_timeout_seconds: int = 30
    sandbox_memory_limit: str = "512m"
    sandbox_cpu_limit: float = 0.5

    self_improve_cron: str = "0 3 * * *"
    self_improve_topic_file: str = "workspace/skills/learning_queue.md"

    # Meta-cognitive loop — retrospective analysis and benchmarking
    retrospective_cron: str = "0 4 * * *"   # daily at 4 AM (after self-improvement)
    benchmark_cron: str = "0 5 * * *"        # daily at 5 AM

    # Parallelism — controls how many crews/sub-agents can run concurrently.
    max_parallel_crews: int = 3   # max crews commander can dispatch at once
    max_sub_agents: int = 4       # max sub-agents a single crew can spawn
    thread_pool_size: int = 6     # shared thread pool size (caps total API calls)

    # Cloud backup — set to a GitHub/git remote URL to enable workspace sync.
    # Use an HTTPS URL with a PAT for simplicity:
    #   https://<PAT>@github.com/you/crewai-memory-backup.git
    # Leave empty to disable (the system works fine without backups).
    workspace_backup_repo: str = ""
    workspace_sync_cron: str = "0 * * * *"  # default: every hour

    # How many recent user+assistant exchanges to include in each new request
    # so the LLM understands short/contextual replies.
    conversation_history_turns: int = 10

    # Evolution loop — autoresearch-style continuous improvement
    evolution_iterations: int = 5         # experiments per evolution session
    evolution_deep_iterations: int = 15   # experiments for "evolve deep" command

    # Auto-deploy: when true, code mutations that pass ALL safety checks
    # + composite_score improvement are deployed automatically without human
    # approval. Post-deploy monitoring auto-rollbacks on error spike.
    # TIER_GATED files additionally require canary deployment pass.
    evolution_auto_deploy: bool = True

    # Evolution engine: "auto" (dynamic selection), "avo" (5-phase AVO pipeline),
    # or "shinka" (ShinkaEvolve island archive). "auto" picks the best engine
    # per session based on stagnation, SUBIA safety, and recent performance.
    evolution_engine: str = "auto"

    # ── Self-improving feedback loop ─────────────────────────────────────
    feedback_enabled: bool = True          # collect feedback signals
    modification_enabled: bool = True      # allow modification engine to run
    modification_tier1_auto: bool = True   # autonomous Tier 1 modifications
    safety_auto_rollback: bool = True      # auto-rollback on negative feedback
    safety_max_negative_before_rollback: int = 2   # negative reactions before rollback

    # ── Fast deployment infrastructure ───────────────────────────────────
    # ── Canary deployment (synthetic eval before promotion) ────────────
    canary_deploy_enabled: bool = True       # route auto-deploys through canary eval
    canary_regression_tolerance: float = 0.05  # 5% regression allowed

    sandbox_evolution_enabled: bool = True    # use Docker sandbox for code mutations
    sandbox_parallel_count: int = 2           # max parallel sandbox instances
    health_monitor_enabled: bool = True       # continuous health monitoring
    self_healing_enabled: bool = True         # auto-remediation on health alerts
    version_manifest_enabled: bool = True     # composite version tracking

    # ── Host Bridge (controlled external resource access) ──────────────
    bridge_enabled: bool = False         # disabled by default — start bridge first
    bridge_host: str = "host.docker.internal"
    bridge_port: int = 9100

    # ── MCP (Model Context Protocol) ────────────────────────────────────
    # Client-side consumption of external MCP servers (e.g. filesystem,
    # github, gdrive). Servers are declared in MCP_SERVERS (JSON array) or
    # /app/workspace/mcp_servers.json. Each server's tools become available
    # to every agent via the base_crew tool plugin registry.
    mcp_client_enabled: bool = True
    mcp_servers_json: str = ""  # optional inline JSON override

    # ── Agent Zero amendments ───────────────────────────────────────────
    history_compression_enabled: bool = True   # 3-tier conversation compression
    lifecycle_hooks_enabled: bool = True       # ordered execution hooks
    tool_self_correction_enabled: bool = True  # LLM-guided tool error correction
    project_isolation_enabled: bool = True     # per-venture memory namespacing

    # ── SubIA integration (Phase 16a) ───────────────────────────────────
    # All default OFF. Each flag is independently toggleable so we can
    # activate SubIA in stages without big-bang deployments.
    #   subia_live_enabled:      registers the CIL lifecycle hooks so every
    #                            crew task runs through the Phase 4 loop.
    #   subia_grounding_enabled: routes chat responses through the Phase 15
    #                            grounding pipeline (ingress correction
    #                            capture + egress fact-checking).
    #   subia_idle_jobs_enabled: registers TSAL + Phase 12 idle jobs with
    #                            the production idle_scheduler.
    subia_live_enabled: bool = Field(
        default=False, validation_alias="SUBIA_FEATURE_FLAG_LIVE",
    )
    subia_grounding_enabled: bool = Field(
        default=False, validation_alias="SUBIA_GROUNDING_ENABLED",
    )
    subia_idle_jobs_enabled: bool = Field(
        default=False, validation_alias="SUBIA_IDLE_JOBS_ENABLED",
    )
    # Phase 17 — self-introspection routing. When enabled, user messages
    # asking AndrusAI about its own state (frustration, mood, attention,
    # etc.) get a system-prompt prefix injected with the live homeostasis
    # snapshot, so the LLM grounds its answer in actual data instead of
    # falling back to "I have no feelings".
    subia_introspection_enabled: bool = Field(
        default=False, validation_alias="SUBIA_INTROSPECTION_ENABLED",
    )

    # ── Gateway HTTP auth enforcement (Phase B3) ───────────────────────
    # When True, every dashboard / epistemic mutating route requires
    # `Authorization: Bearer <gateway_secret>`. Default False preserves
    # the laptop developer experience (localhost-only). Helm sets it to
    # True for K8s deployments, where the perimeter is no longer the OS.
    # See app/control_plane/auth_dep.py for the dependency implementation.
    # NOTE: this field is also readable via os.environ.get directly so the
    # dependency module does not have to construct a Settings instance.
    gateway_auth_required: bool = Field(
        default=False, validation_alias="GATEWAY_AUTH_REQUIRED",
    )

    # ── Control Plane ─────────────────────────────────────────────────
    control_plane_enabled: bool = True      # enable control plane (tickets, budgets, audit)
    budget_enforcement_enabled: bool = True  # pre-call budget checks in llm_factory
    ticket_system_enabled: bool = True      # create tickets from Signal messages
    default_budget_per_agent_usd: float = 50.0  # monthly budget per agent
    autonomous_mode: bool = False           # 24/7 heartbeat-driven ticket processing
    load_shed_threshold: int = 0            # 0 = auto (max_parallel_crews + 1)

    # ── Creative Mode (multi-agent divergent-discussion-convergence) ─────
    # Hard cap in USD on a single creative run. If the run's accumulated
    # LLM cost exceeds this value, the orchestrator aborts mid-phase and
    # returns the best output so far. Adjustable via dashboard.
    creative_run_budget_usd: float = 0.10
    # Weight for wiki-based vs Mem0-based originality distance in Torrance scoring.
    # Higher = originality judged more against shared corpus; lower = more against
    # agent's own prior utterances.
    creative_originality_wiki_weight: float = 0.6  # Mem0 weight = 1 - this

    # ── ATLAS: Autonomous Tool-Learning & Adaptive Skills ──────────────
    atlas_enabled: bool = True           # enable ATLAS capability layer
    atlas_api_scout_enabled: bool = True  # autonomous API discovery
    atlas_video_learning_enabled: bool = True  # YouTube knowledge extraction
    atlas_code_forge_enabled: bool = True  # grounded code generation
    atlas_competence_tracking: bool = True  # competence gap detection

    # ── Local LLM (Native Ollama + Metal GPU) ─────────────────────────────
    # Uses native Ollama installation for Metal GPU acceleration.
    # All roles default to qwen3.5:35b-a3b-q4_K_M (MoE, ~20GB on disk,
    # 35B total / 3B active per token — speed of 30B-A3B with stronger
    # tools support that fixes the mem0 function-calling gap).
    local_llm_enabled: bool = True
    local_llm_base_url: str = "http://host.docker.internal:11434"
    ollama_base_url: str = "http://localhost:11434"  # native Ollama on host
    ollama_max_concurrent_crews: int = 2  # max crews hitting Ollama at once (semaphore)

    # Role → model mapping (Ollama model names, auto-pulled on first use).
    # 2026-04-25: swapped qwen3:30b-a3b → qwen3.5:35b-a3b-q4_K_M for
    # better tools support + vision + thinking modes.  Same memory class.
    local_model_coding: str = "qwen3.5:35b-a3b-q4_K_M"         # MoE — fast, excellent coding
    local_model_architecture: str = "qwen3.5:35b-a3b-q4_K_M"   # MoE — strong reasoning
    local_model_research: str = "qwen3.5:35b-a3b-q4_K_M"       # web research + synthesis
    local_model_writing: str = "qwen3.5:35b-a3b-q4_K_M"        # docs, summaries, reports
    local_model_default: str = "qwen3.5:35b-a3b-q4_K_M"        # fallback for unspecified

    # Vetting — Claude reviews local LLM output before sending to user
    vetting_enabled: bool = True
    vetting_model: str = "claude-sonnet-4.6"  # Sonnet 4.6: #1 GDPval-AA, 5x cheaper than Opus

    # ── Mem0 persistent memory ──────────────────────────────────────
    # Adds cross-session fact extraction (LLM-based) and graph memory (Neo4j).
    # Coexists with ChromaDB — Mem0 for persistent knowledge, ChromaDB for
    # real-time operational state (beliefs, policies, self-reports).
    mem0_enabled: bool = True
    mem0_postgres_host: str = "postgres"
    mem0_postgres_port: int = 5432
    mem0_postgres_user: str = "mem0"
    mem0_postgres_password: SecretStr = SecretStr("")  # MUST be set via MEM0_POSTGRES_PASSWORD env var
    mem0_postgres_db: str = "mem0"
    mem0_neo4j_url: str = "bolt://neo4j:7687"
    mem0_neo4j_user: str = "neo4j"
    mem0_neo4j_password: SecretStr = SecretStr("")  # MUST be set via MEM0_NEO4J_PASSWORD env var
    mem0_llm_model: str = "ollama/qwen3.5:35b-a3b-q4_K_M"  # local model for fact extraction (Qwen3.5 has tools support)
    mem0_embedder_model: str = "nomic-ai/nomic-embed-text-v1.5"  # 768-dim, matches ChromaDB
    mem0_user_id: str = "owner"  # single-user system

    @property
    def mem0_postgres_url(self) -> str:
        """Build postgres URL from components — password never hardcoded.

        H6: URL contains plaintext password — callers must NEVER log this value.
        The __repr__ of Settings already masks SecretStr fields, but this property
        returns a plain str so it must be handled with care.
        """
        pw = self.mem0_postgres_password.get_secret_value()
        if not pw:
            return ""
        return f"postgresql://{self.mem0_postgres_user}:{pw}@{self.mem0_postgres_host}:{self.mem0_postgres_port}/{self.mem0_postgres_db}"

    @property
    def mem0_postgres_url_safe(self) -> str:
        """Redacted version safe for logging."""
        pw = self.mem0_postgres_password.get_secret_value()
        if not pw:
            return ""
        return f"postgresql://{self.mem0_postgres_user}:***@{self.mem0_postgres_host}:{self.mem0_postgres_port}/{self.mem0_postgres_db}"

    # Firebase — service account for Firestore dashboard writes
    firebase_service_account_json: str = ""

    # Temporal context — default location for seasonal/astronomical context
    default_latitude: float = 60.17      # Helsinki
    default_longitude: float = 24.94
    default_timezone: str = "Europe/Helsinki"

    # Embedding — pinned dimension (768 = Ollama nomic-embed-text)
    embedding_dimension: int = 768
    embedding_refuse_fallback: bool = True  # Legacy — CPU fallback removed entirely

    # ── Email (IMAP/SMTP) — zero external deps, Python stdlib ────
    email_enabled: bool = False
    email_imap_host: str = ""           # e.g. "imap.gmail.com"
    email_imap_port: int = 993
    email_smtp_host: str = ""           # e.g. "smtp.gmail.com"
    email_smtp_port: int = 587
    email_address: str = ""
    email_password: SecretStr = SecretStr("")  # Gmail: use App Password

    # Importance ranking (rank_emails tool): comma-separated list of
    # senders to upweight. Each entry can be a full address
    # (alice@example.com), a domain (@example.com), or a name fragment
    # (Alice). Match is case-insensitive substring on the From header.
    # Empty = no upweight overlay (heuristic still ranks personal mail
    # above bulk/marketing via header analysis).
    email_important_senders: str = ""

    # ── SEC EDGAR (financial filings API — free, no key) ──────
    sec_edgar_user_agent: str = "BotArmy/1.0 (contact@example.com)"

    # Structured logging
    structured_log_path: str = "/app/workspace/logs/errors.jsonl"
    structured_log_max_mb: int = 50

    # Workspace versioning
    workspace_lock_timeout_s: int = 30

    # Idle scheduler tuning
    idle_lightweight_workers: int = 3
    idle_heavy_time_cap_s: int = 600
    idle_training_interval_s: int = 3600

    # Consciousness indicators (Butlin et al. 2025)
    consciousness_enabled: bool = True
    workspace_capacity: int = 5
    belief_store_enabled: bool = True

    # CrewAI per-crew verbose logging. Env: CREW_VERBOSE=1 to enable.
    crew_verbose: bool = False

    # Speed upgrade Stage 4.3 — cascade race for short prompts. When enabled,
    # hybrid mode fires Ollama + API-tier concurrently on prompts <800 tokens
    # and uses the first non-error response. Adds a small API cost — enable
    # only after watching cost dashboard. Default OFF.
    cascade_race_short: bool = False
    cascade_race_token_threshold: int = 800   # ~3200 chars — short prompts only
    cascade_race_timeout_s: float = 4.0

    # ── Trajectory-informed memory (arXiv:2603.10600) ─────────────────
    # Five independently toggleable flags that stage in the trajectory
    # attribution → tip synthesis → task-conditional retrieval loop.
    # All now ENABLED — the full paper-derived loop is live in production.
    #
    #   trajectory_enabled:
    #       Capture per-crew execution trajectories (types.TrajectoryStep)
    #       alongside existing telemetry. No LLM calls. Pure data capture.
    #   attribution_enabled (requires trajectory_enabled):
    #       After failed/retried/slow/recovered runs, an infrastructure
    #       AttributionAnalyzer runs a short LLM call to identify the
    #       causal decision + failure mode, emitting a LearningGap.
    #   tip_synthesis_enabled (requires attribution_enabled):
    #       Self-Improver idle job picks up trajectory-attribution gaps
    #       and synthesises strategy/recovery/optimization tips via the
    #       existing Integrator → KB pipeline.
    #   task_conditional_retrieval_enabled:
    #       Commander retrieves tips using a metadata where_filter
    #       keyed on (agent_role, tip_type, predicted_failure_mode).
    #   observer_calibration_enabled (requires attribution_enabled):
    #       Close the loop — compare Observer pre-predictions against
    #       Attribution post-labels and emit OBSERVER_MIS_PREDICTION
    #       gaps when a failure mode is chronically mis-called.
    trajectory_enabled: bool = True
    attribution_enabled: bool = True
    tip_synthesis_enabled: bool = True
    task_conditional_retrieval_enabled: bool = True
    observer_calibration_enabled: bool = True

    # ── Transfer Insight Layer (Phase 17, arXiv:2606.21099-style MTL) ──
    # Cross-domain meta-memory compiled from healing/evo/grounding/gap.
    # Drafts always land in shadow_drafts.jsonl + KBs at status="shadow".
    #
    #   transfer_memory_shadow_logging_enabled:
    #       Log what WOULD be retrieved on each dispatch (without
    #       injecting). Captures predicted-vs-actual usefulness data
    #       for the operator to review before flipping retrieval on.
    #   transfer_memory_retrieval_enabled:
    #       Inject the <transfer_memory> block into the dispatch
    #       prompt. Default OFF — flip to True only after shadow data
    #       suggests positive transfer. Per-domain rollout via
    #       ``transfer_memory_enabled_domains`` (comma-sep, "" = all).
    #   transfer_memory_auto_promote_enabled:
    #       Allow the promotion idle job to flip shadow records to
    #       active automatically once they meet the effectiveness
    #       threshold. Default OFF — promotion stays in operator hands
    #       until measured data justifies automation.
    transfer_memory_shadow_logging_enabled: bool = True
    transfer_memory_retrieval_enabled: bool = True
    # Auto-promote ON: the transfer-promotion idle job (≥6h cadence) will
    # flip shadow records to active once they pass the deterministic
    # eligibility check (age ≥7d + surface_count ≥3 + not blacklisted +
    # zero negative-transfer log entries). Manual promotion via the
    # dashboard endpoint still works for ad-hoc operator action.
    transfer_memory_auto_promote_enabled: bool = True
    # Phase 17c per-domain rollout — coding + grounding go live first
    # (paper's strongest evidence + AndrusAI's grounding correction
    # substrate). Empty string = all domains; widen by editing this
    # value once the per-domain shadow data validates further rollout.
    transfer_memory_enabled_domains: str = "coding,grounding"

    # ── Personal-agent surface (Phase 0 — May 2026) ────────────────────
    # Voice mode: "off" disables both STT and TTS; "local" uses the
    # whisper.cpp + Piper binaries on the host; "cloud" uses Groq Whisper
    # for STT and Google Cloud Neural2 for TTS. Switchable at runtime
    # via /config/voice_mode (file-backed in app.runtime_settings).
    voice_mode: str = "off"
    # Vision-driven computer use (Anthropic Haiku 4.5). Disabled by default;
    # gated by a monthly USD cap that the dashboard can raise/lower at runtime.
    vision_cu_enabled: bool = False
    vision_cu_monthly_cap_usd: float = 10.0
    # Concierge persona — wraps Commander's terse output in a warmer voice
    # for direct chat (Signal DMs + /cp/chat). Bypassed for tool output
    # and structured /cp API consumers.
    concierge_persona_enabled: bool = False

    # ── Voice provider keys ────────────────────────────────────────────
    groq_api_key: SecretStr = SecretStr("")              # STT (cloud mode)
    google_cloud_tts_key: SecretStr = SecretStr("")      # TTS (cloud mode)

    # ── Google Workspace OAuth (Gmail, Calendar, Docs, Sheets, Slides) ──
    # Installed-app flow; bootstrap with `python -m app.google_workspace.bootstrap`.
    # Refresh token persists in workspace/google_token.json (chmod 600).
    google_oauth_client_id: SecretStr = SecretStr("")
    google_oauth_client_secret: SecretStr = SecretStr("")

    model_config = ConfigDict(env_file=".env", extra="ignore")

    @field_validator("sandbox_memory_limit")
    @classmethod
    def validate_memory_limit(cls, v: str) -> str:
        """Accept Docker memory strings like 128m, 1g and clamp to safe bounds."""
        m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([kmgKMG]?)", v.strip())
        if not m:
            raise ValueError(f"Invalid sandbox_memory_limit format: {v!r}. Use e.g. '512m' or '1g'.")
        amount, unit = float(m.group(1)), m.group(2).lower()
        mb = amount * {"k": 1/1024, "m": 1, "g": 1024, "": 1}[unit]
        if mb < _MIN_SANDBOX_MEMORY_MB:
            raise ValueError(f"sandbox_memory_limit {v!r} is below minimum {_MIN_SANDBOX_MEMORY_MB}m.")
        if mb > _MAX_SANDBOX_MEMORY_MB:
            raise ValueError(f"sandbox_memory_limit {v!r} exceeds maximum {_MAX_SANDBOX_MEMORY_MB}m.")
        return v

    @field_validator("sandbox_cpu_limit")
    @classmethod
    def validate_cpu_limit(cls, v: float) -> float:
        if v < _MIN_SANDBOX_CPU or v > _MAX_SANDBOX_CPU:
            raise ValueError(
                f"sandbox_cpu_limit {v} out of range [{_MIN_SANDBOX_CPU}, {_MAX_SANDBOX_CPU}]."
            )
        return v

    @field_validator("sandbox_timeout_seconds")
    @classmethod
    def validate_timeout(cls, v: int) -> int:
        if v < 1 or v > _MAX_SANDBOX_TIMEOUT:
            raise ValueError(
                f"sandbox_timeout_seconds {v} out of range [1, {_MAX_SANDBOX_TIMEOUT}]."
            )
        return v

    @field_validator("voice_mode")
    @classmethod
    def validate_voice_mode(cls, v: str) -> str:
        allowed = ("off", "local", "cloud")
        v = v.strip().lower()
        if v not in allowed:
            raise ValueError(f"voice_mode must be one of {allowed}, got {v!r}")
        return v

    @field_validator("vision_cu_monthly_cap_usd")
    @classmethod
    def validate_vision_cu_cap(cls, v: float) -> float:
        if v < 0.0:
            raise ValueError("vision_cu_monthly_cap_usd must be non-negative")
        if v > 1000.0:
            raise ValueError("vision_cu_monthly_cap_usd exceeds sanity cap of $1000/mo")
        return float(v)

    @field_validator("cost_mode")
    @classmethod
    def validate_cost_mode(cls, v: str) -> str:
        """Accept the unified 6-mode vocabulary (and legacy 3-mode values).

        ``cost_mode`` is retained for back-compat with deployment configs
        written against the pre-unification vocabulary. New deployments
        should set ``llm_mode`` instead — the two fields now share a
        vocabulary and the factory reads ``llm_mode`` via ``get_mode()``.
        """
        allowed = (
            "free", "budget", "balanced", "quality", "insane", "anthropic",
            # Legacy aliases — tolerated at config-load time so a stale
            # .env doesn't brick startup.
            "hybrid", "local", "cloud",
        )
        if v not in allowed:
            raise ValueError(f"cost_mode must be one of {allowed}, got {v!r}")
        return v


@cache
def get_settings() -> Settings:
    return Settings()


# --- Scoped accessors so agents only see what they need ---

def get_anthropic_api_key() -> str:
    """Return the Anthropic API key (for LLM agents only)."""
    return get_settings().anthropic_api_key.get_secret_value()


def get_brave_api_key() -> str:
    """Return the Brave Search API key (for web search tool only)."""
    return get_settings().brave_api_key.get_secret_value()


def get_gateway_secret() -> str:
    """Return the gateway secret (for HTTP auth only)."""
    return get_settings().gateway_secret.get_secret_value()


def get_openrouter_api_key() -> str:
    """Return the OpenRouter API key (for API-tier LLMs only)."""
    return get_settings().openrouter_api_key.get_secret_value()


def get_groq_api_key() -> str:
    """Return the Groq API key (for cloud-mode Whisper STT only)."""
    return get_settings().groq_api_key.get_secret_value()


def get_google_cloud_tts_key() -> str:
    """Return the Google Cloud TTS API key (for cloud-mode Neural2 TTS only)."""
    return get_settings().google_cloud_tts_key.get_secret_value()


def get_google_oauth_client() -> tuple[str, str]:
    """Return (client_id, client_secret) for Google Workspace OAuth."""
    s = get_settings()
    return (
        s.google_oauth_client_id.get_secret_value(),
        s.google_oauth_client_secret.get_secret_value(),
    )
