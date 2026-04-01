import re
from pydantic_settings import BaseSettings
from pydantic import ConfigDict, SecretStr, field_validator
from functools import lru_cache

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

    # LLM mode — "local", "cloud", or "hybrid" (initial value; changes at runtime)
    llm_mode: str = "hybrid"

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
    # Default: false (human approval via Signal required for code changes).
    evolution_auto_deploy: bool = False

    # ── Self-improving feedback loop ─────────────────────────────────────
    feedback_enabled: bool = True          # collect feedback signals
    modification_enabled: bool = True      # allow modification engine to run
    modification_tier1_auto: bool = True   # autonomous Tier 1 modifications
    safety_auto_rollback: bool = True      # auto-rollback on negative feedback
    safety_max_negative_before_rollback: int = 2   # negative reactions before rollback

    # ── Local LLM (Native Ollama + Metal GPU) ─────────────────────────────
    # Uses native Ollama installation for Metal GPU acceleration.
    # All roles default to qwen3:30b-a3b (MoE, ~20GB, 15-22 tok/s on M4 Max).
    local_llm_enabled: bool = True
    local_llm_base_url: str = "http://host.docker.internal:11434"
    ollama_base_url: str = "http://localhost:11434"  # native Ollama on host
    ollama_max_concurrent_crews: int = 2  # max crews hitting Ollama at once (semaphore)

    # Role → model mapping (Ollama model names, auto-pulled on first use)
    local_model_coding: str = "qwen3:30b-a3b"         # MoE — fast, excellent coding
    local_model_architecture: str = "qwen3:30b-a3b"   # MoE — strong reasoning
    local_model_research: str = "qwen3:30b-a3b"      # web research + synthesis
    local_model_writing: str = "qwen3:30b-a3b"       # docs, summaries, reports
    local_model_default: str = "qwen3:30b-a3b"       # fallback for unspecified

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
    mem0_llm_model: str = "ollama/qwen3:30b-a3b"  # local model for fact extraction
    mem0_embedder_model: str = "all-MiniLM-L6-v2"  # same as ChromaDB
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

    model_config = ConfigDict(env_file=".env")

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

    @field_validator("cost_mode")
    @classmethod
    def validate_cost_mode(cls, v: str) -> str:
        allowed = ("budget", "balanced", "quality")
        if v not in allowed:
            raise ValueError(f"cost_mode must be one of {allowed}, got {v!r}")
        return v


@lru_cache
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
