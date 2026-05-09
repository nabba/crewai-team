"""Hand-curated catalogue of every Signal slash / natural-language
command the dispatcher in ``commands.py`` recognises.

This is the single source of truth surfaced by
``GET /api/cp/signal-commands`` and rendered in the React /cp/chat
page sidebar. Hand-curated rather than introspected because the
dispatcher uses many different idioms (``lower in (...)``,
``lower.startswith(...)``, regex parsing inside helper functions),
and a syntactic scan misses the human-readable shape of each.

Adding a new command? Add a row here AND add the dispatcher branch
to ``try_command()``. The two stay paired by convention, not by
machinery.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SignalCommand:
    command: str
    aliases: tuple[str, ...]
    syntax: str
    description: str
    category: str


_CATEGORIES = (
    "Help & status",
    "Skills",
    "Self-improvement",
    "Routing & workspaces",
    "Crews & evolution",
    "Tickets & governance",
    "Budgets & costs",
    "LLM catalog",
    "Knowledge bases",
    "Tools & integrations",
    "Audit & ops",
    "Personal life",
    "Brainstorm",
    "Sessions",
)


SIGNAL_COMMANDS: tuple[SignalCommand, ...] = (
    # ── Help & status ─────────────────────────────────────────────
    SignalCommand("/help", ("help", "?"), "/help",
                  "Compact list of the most useful Signal commands.",
                  "Help & status"),
    SignalCommand("/status", ("status",), "/status",
                  "Uptime, voice mode, scheduled tasks, push devices, last error.",
                  "Help & status"),
    SignalCommand("/usage", ("usage",), "/usage",
                  "Per-period token usage and cost summary.",
                  "Help & status"),
    SignalCommand("/compress", ("compress",), "/compress",
                  "Manually compress current conversation history.",
                  "Help & status"),

    # ── Skills ────────────────────────────────────────────────────
    SignalCommand("/skill save", (), "/skill save <name>: <task template>",
                  "Save a new skill. Without a colon, uses your last user message as the template.",
                  "Skills"),
    SignalCommand("/skill list", (), "/skill list",
                  "List every saved skill.",
                  "Skills"),
    SignalCommand("/skill show", (), "/skill show <name>",
                  "Show a skill's template + usage counters.",
                  "Skills"),
    SignalCommand("/skill run", (), "/skill run <name> [k=v ...]",
                  "Substitute args into the template and dispatch.",
                  "Skills"),
    SignalCommand("/skill delete", (), "/skill delete <name>",
                  "Remove a saved skill.",
                  "Skills"),
    SignalCommand("skills", ("list skills", "show skills"), "skills",
                  "Plain-language alias for `/skill list`.",
                  "Skills"),

    # ── Self-improvement ─────────────────────────────────────────
    SignalCommand("learn", (), "learn <topic>",
                  "Queue a topic for self-improvement (researched in the next idle window).",
                  "Self-improvement"),
    SignalCommand("please learn", (), "please learn <topic>",
                  "Queue + immediately run a self-improvement cycle on the topic.",
                  "Self-improvement"),
    SignalCommand("show learning queue", (), "show learning queue",
                  "Print pending self-improvement topics.",
                  "Self-improvement"),
    SignalCommand("run self-improvement now", (), "run self-improvement now",
                  "Trigger a learning-queue drain immediately.",
                  "Self-improvement"),
    SignalCommand("watch", (), "watch <youtube url>",
                  "Distill a YouTube video transcript into a skill + team memory entry.",
                  "Self-improvement"),
    SignalCommand("improve", (), "improve",
                  "Run a one-shot improvement scan and surface proposals.",
                  "Self-improvement"),
    SignalCommand("retrospective", ("run retrospective",), "retrospective",
                  "Run the retrospective crew across recent traces and update policies.",
                  "Self-improvement"),
    SignalCommand("policies", ("show policies",), "policies",
                  "List current behavioural policies (output of the retrospective crew).",
                  "Self-improvement"),

    # ── Routing & workspaces ─────────────────────────────────────
    SignalCommand("force this", ("try harder",),
                  "force this / try harder",
                  "Re-run the last refusal through the recovery loop with a tool-first nudge.",
                  "Routing & workspaces"),
    SignalCommand("switch workspace to", (),
                  "switch workspace to <name>",
                  "Change the active workspace (PLG, Archibal, KaiCart, Eesti mets, default).",
                  "Routing & workspaces"),
    SignalCommand("workspaces", (), "workspaces",
                  "List every workspace and which is active.",
                  "Routing & workspaces"),

    # ── Crews & evolution ────────────────────────────────────────
    SignalCommand("evolve", (), "evolve",
                  "Trigger one evolution iteration (mutate + score + maybe keep).",
                  "Crews & evolution"),
    SignalCommand("evolve deep", (), "evolve deep",
                  "Trigger a deep evolution batch (multiple iterations + ensemble).",
                  "Crews & evolution"),
    SignalCommand("experiments", ("show experiments",), "experiments",
                  "Show recent evolution experiments + outcomes.",
                  "Crews & evolution"),
    SignalCommand("results", ("show results",), "results",
                  "Show evolution results table.",
                  "Crews & evolution"),
    SignalCommand("metrics", ("show metrics",), "metrics",
                  "Composite scores + per-metric breakdown.",
                  "Crews & evolution"),
    SignalCommand("variants", ("archive", "genealogy"), "variants",
                  "List archived prompt variants + their lineage.",
                  "Crews & evolution"),
    SignalCommand("benchmarks", ("show benchmarks",), "benchmarks",
                  "Print the latest LLM benchmark snapshot.",
                  "Crews & evolution"),

    # ── Tickets & governance ─────────────────────────────────────
    SignalCommand("tickets", ("ticket list", "kanban"), "tickets",
                  "List recent tickets (todo + in_progress) for the active project.",
                  "Tickets & governance"),
    SignalCommand("pending", ("governance", "governance pending"), "pending",
                  "Show pending governance approval requests.",
                  "Tickets & governance"),
    SignalCommand("approve", (), "approve <governance-id>",
                  "Approve a pending governance request.",
                  "Tickets & governance"),
    SignalCommand("reject", (), "reject <governance-id>",
                  "Reject a pending governance request.",
                  "Tickets & governance"),
    SignalCommand("proposals", ("show proposals",), "proposals",
                  "List active improvement proposals from the self-improver.",
                  "Tickets & governance"),
    SignalCommand("org chart", ("org", "team"), "org chart",
                  "Show the org chart (commander → user crews + internal agents).",
                  "Tickets & governance"),

    # ── Budgets & costs ──────────────────────────────────────────
    SignalCommand("budget", ("budget status", "budgets"), "budget",
                  "Per-agent budget status for the active project (limit / spent / paused).",
                  "Budgets & costs"),
    SignalCommand("tokens", ("token usage",), "tokens [period]",
                  "Token usage; optional period: hour / day / week / month / quarter / year.",
                  "Budgets & costs"),
    SignalCommand("auto deploy on", (), "auto deploy on",
                  "Enable auto-deployment of self-improver-approved patches.",
                  "Budgets & costs"),
    SignalCommand("auto deploy off", (), "auto deploy off",
                  "Disable auto-deployment.",
                  "Budgets & costs"),
    SignalCommand("auto deploy", (), "auto deploy",
                  "Show current auto-deploy status.",
                  "Budgets & costs"),
    SignalCommand("deploys", ("deploy log",), "deploys",
                  "Show recent self-deploy attempts + outcomes.",
                  "Budgets & costs"),
    SignalCommand("rollback", (), "rollback <deploy-id>",
                  "Roll back a previously-applied deploy.",
                  "Budgets & costs"),
    SignalCommand("diff", (), "diff <deploy-id>",
                  "Show the diff a self-deploy applied.",
                  "Budgets & costs"),

    # ── LLM catalog ──────────────────────────────────────────────
    SignalCommand("mode", (), "mode [free|budget|balanced|quality|insane|anthropic]",
                  "Show or set the runtime LLM tier mode.",
                  "LLM catalog"),
    SignalCommand("catalog", ("show catalog",), "catalog",
                  "Print the LLM catalog (every model + tier + recent benchmark).",
                  "LLM catalog"),
    SignalCommand("llm status", ("llm",), "llm status",
                  "Show the active models per role + last refresh time.",
                  "LLM catalog"),
    SignalCommand("llm ranks", (), "llm ranks [tier]",
                  "External-rank table for a tier (free / budget / mid / premium).",
                  "LLM catalog"),
    SignalCommand("refresh ranks", ("ranks refresh",), "refresh ranks",
                  "Re-fetch external ranks (livebench, openrouter rankings, …).",
                  "LLM catalog"),
    SignalCommand("refresh catalog", ("catalog refresh",), "refresh catalog",
                  "Re-fetch the LLM catalog from upstream sources.",
                  "LLM catalog"),
    SignalCommand("rebenchmark", (), "rebenchmark <model>",
                  "Re-run the benchmark on a specific model.",
                  "LLM catalog"),
    SignalCommand("discover models", ("discover", "model discovery"),
                  "discover models",
                  "Run a discovery cycle now (find new models + benchmark them).",
                  "LLM catalog"),
    SignalCommand("discovered", ("models discovered",), "discovered",
                  "Show recently-discovered models awaiting promotion.",
                  "LLM catalog"),
    SignalCommand("promote", (), "promote <model> <tier>",
                  "Promote a model to a tier (locks it as a candidate for that tier's role).",
                  "LLM catalog"),
    SignalCommand("demote", (), "demote <model>",
                  "Reverse a promotion.",
                  "LLM catalog"),
    SignalCommand("promoted", ("list promoted",), "promoted",
                  "List currently-promoted models.",
                  "LLM catalog"),
    SignalCommand("pin", (), "pin <role> <model>",
                  "Pin a specific model to an agent role (overrides selector).",
                  "LLM catalog"),
    SignalCommand("unpin", (), "unpin <role>",
                  "Clear a role pin.",
                  "LLM catalog"),
    SignalCommand("pinned", ("list pins", "show pins"), "pinned",
                  "Show all current role pins.",
                  "LLM catalog"),
    SignalCommand("fleet", ("models",), "fleet",
                  "Show local Ollama fleet (downloaded models + RAM headroom).",
                  "LLM catalog"),
    SignalCommand("fleet stop all", (), "fleet stop all",
                  "Unload every running Ollama model.",
                  "LLM catalog"),
    SignalCommand("fleet pull", (), "fleet pull <model>",
                  "Pull a specific model into the local Ollama fleet.",
                  "LLM catalog"),

    # ── Knowledge bases ──────────────────────────────────────────
    SignalCommand("kb status", ("kb", "knowledge base"), "kb status",
                  "Per-KB stats (kb / fiction / philosophy / episteme / experiential / aesthetics / tensions).",
                  "Knowledge bases"),
    SignalCommand("kb list", (), "kb list",
                  "List every KB document.",
                  "Knowledge bases"),
    SignalCommand("kb add", (), "kb add <text>",
                  "Add a snippet to the default KB. Use kb add to <name> ... to target a specific KB.",
                  "Knowledge bases"),
    SignalCommand("kb remove", (), "kb remove <id>",
                  "Remove a KB document by id.",
                  "Knowledge bases"),
    SignalCommand("kb search", (), "kb search <query>",
                  "Semantic search over the default KB.",
                  "Knowledge bases"),
    SignalCommand("kb reset", (), "kb reset",
                  "Wipe and re-ingest the default KB. Destructive.",
                  "Knowledge bases"),

    # ── Tools & integrations ─────────────────────────────────────
    SignalCommand("composio", ("composio status", "integrations"), "composio",
                  "Show Composio integration status.",
                  "Tools & integrations"),
    SignalCommand("composio apps", ("composio connected",), "composio apps",
                  "List connected Composio apps + tool counts.",
                  "Tools & integrations"),
    SignalCommand("bridge", ("bridge status",), "bridge",
                  "Show host-bridge status (sandbox limits, recent activity).",
                  "Tools & integrations"),
    SignalCommand("mcp", ("mcp status", "mcp servers"), "mcp",
                  "Show MCP server status (connected / available tools).",
                  "Tools & integrations"),

    # ── Audit & ops ──────────────────────────────────────────────
    SignalCommand("errors", ("show errors",), "errors",
                  "Recent error patterns + last few errors.",
                  "Audit & ops"),
    SignalCommand("anomalies", ("alerts",), "anomalies",
                  "Recent statistical anomalies from the detector.",
                  "Audit & ops"),
    SignalCommand("audit", ("run audit", "code audit"), "audit",
                  "Run an audit pass over the codebase.",
                  "Audit & ops"),
    SignalCommand("fix errors", ("resolve errors",), "fix errors",
                  "Trigger the auditor's fix-errors pass.",
                  "Audit & ops"),
    SignalCommand("audit status", ("auditor",), "audit status",
                  "Show auditor status + last run.",
                  "Audit & ops"),
    SignalCommand("tech radar", ("tech", "radar", "discoveries"), "tech radar",
                  "Show recent tech-radar discoveries.",
                  "Audit & ops"),
    SignalCommand("training", ("training status",), "training",
                  "Show training-curation status.",
                  "Audit & ops"),
    SignalCommand("train now", (), "train now",
                  "Force a training-pipeline run now.",
                  "Audit & ops"),
    SignalCommand("schedules", ("show schedules", "list schedules"), "schedules",
                  "List scheduled tasks.",
                  "Audit & ops"),
    SignalCommand("jobs", ("list jobs", "show jobs"), "jobs",
                  "List background idle jobs + last run / next run.",
                  "Audit & ops"),
    SignalCommand("program", ("show program",), "program",
                  "Print the program plan (PROGRAM.md) summary.",
                  "Audit & ops"),

    # ── Personal life ────────────────────────────────────────────
    SignalCommand("check email", ("email", "inbox"), "check email",
                  "Show top urgent unread emails (Personal-Agent surface).",
                  "Personal life"),
    SignalCommand("calendar", ("schedule", "events", "today"), "calendar",
                  "Show today's calendar events.",
                  "Personal life"),
    SignalCommand("tasks", ("todo", "task list"), "tasks",
                  "List your personal tasks.",
                  "Personal life"),

    # ── Brainstorm ───────────────────────────────────────────────
    SignalCommand("/brainstorm", (), "/brainstorm <topic>",
                  "Interactive ideation: SCAMPER, Six Hats, How-Might-We, Reverse, Crazy-8s, Rapid Ideation, Starbursting.",
                  "Brainstorm"),
)


def to_payload() -> list[dict]:
    """Serialise the registry for the dashboard endpoint."""
    return [asdict(c) for c in SIGNAL_COMMANDS]


def categories() -> list[str]:
    """Insertion-order category list (mirrors _CATEGORIES)."""
    seen: list[str] = []
    for c in SIGNAL_COMMANDS:
        if c.category not in seen:
            seen.append(c.category)
    return seen
