import logging
import re
from pathlib import Path
from app.config import get_settings
from app.llm_factory import is_using_local

settings = get_settings()
logger = logging.getLogger(__name__)


def try_command(user_input: str, sender: str, commander) -> str | None:
    """Try to handle user_input as a special command. Returns response string or None."""
    lower = user_input.lower().strip()

    # ── Brainstorm subsystem ───────────────────────────────────────────
    # Claims /brainstorm slash commands AND any plain message from a sender
    # that already has an active brainstorm session (so multi-turn Q/A
    # works without the user re-prefixing every reply).
    try:
        from app.brainstorm.signal_handler import try_handle as _brainstorm_try
        _b = _brainstorm_try(user_input, sender)
        if _b is not None:
            return _b
    except Exception as _bs_exc:
        logger.warning("brainstorm: routing failed (%s) — falling through", _bs_exc)

    # Access the shared thread pool from orchestrator
    from app.agents.commander.orchestrator import _ctx_pool

    # ── Phase 4 mobile-surface slash commands (May 2026) ───────────
    # /help: list the most useful Signal commands at a glance.
    # /status: live system summary (uptime, voice mode, scheduled tasks,
    # last error, recent push count) — same data the React /cp/ops page
    # shows, but condensed for a quick mobile glance.
    if lower in ("/help", "help", "?"):
        return _signal_help()
    if lower in ("/status", "status"):
        return _signal_status()

    # ── Phase F #10 (2026-05-09) — long-arc commitment management ─────
    # /commitment list                              show active commitments
    # /commitment fulfilled <id>                    mark fulfilled
    # /commitment broken <id>                       mark broken
    # /commitment deferred <id>                     mark deferred (+mute nudges)
    # /commitment unmute <id>                       resume long_arc nudges
    if lower.startswith("/commitment") or lower.startswith("commitment "):
        sub = _handle_commitment_command(user_input)
        if sub is not None:
            return sub

    # ── Phase G #3 (2026-05-09) — topic dormancy mute/unmute ──────────
    # /topic mute <name>                            silence dormancy nudges
    # /topic unmute <name>                          resume nudges
    if lower.startswith("/topic") or lower.startswith("topic "):
        sub = _handle_topic_command(user_input)
        if sub is not None:
            return sub

    # ── Q4#16 (PROGRAM §41, 2026-05-11) — companion tensions ──────────
    # /tensions                                     list open tensions
    # /tensions list                                same
    # /tensions add <question>                      manually file a tension
    # /tensions resolve <id> <resolution>           mark RESOLVED
    if lower.startswith("/tensions") or lower.startswith("tensions "):
        sub = _handle_tensions_command(user_input)
        if sub is not None:
            return sub

    # ── Q4.2 (PROGRAM §42) — person correlation ───────────────────────
    # /person                                       list tracked people
    # /person mute <email>                          mute from surfaces
    # /person unmute <email>                        unmute
    # /person forget <email>                        delete profile
    # /person forget-all                            nuke everything
    # /person forget-graph                          delete graph only
    # /person mute-suggestions <email>              suppress nudges
    # /person opt-out-of-paths <email>              exclude as conduit
    # /person path-to <email>                       L4.1 path query
    if lower.startswith("/person") or lower.startswith("person "):
        sub = _handle_person_command(user_input)
        if sub is not None:
            return sub

    # ── Q8.1 (PROGRAM §46.1) — long-horizon threads ──────────────────
    # /thread                                       list open threads
    # /thread start <title>                         create a new thread
    # /thread status [id]                           show thread detail
    # /thread note <id> <text>                      append a note
    # /thread subq <id> <text>                      add a sub-question
    # /thread done <id> <subq_id> [resolution]      resolve a sub-question
    # /thread block <id> <reason>                   file a blocker
    # /thread hint <id> <text>                      append "what might unblock"
    # /thread unblock <id>                          clear blockers
    # /thread resolve <id> [summary]                close as RESOLVED
    # /thread abandon <id> <reason>                 close as ABANDONED
    if (
        lower.startswith("/thread")
        or lower.startswith("thread ")
        or lower == "thread"
    ):
        sub = _handle_thread_command(user_input)
        if sub is not None:
            return sub

    # ── Phase 5 skill registry slash commands ──────────────────────
    # /skill save <name>: <task template>           save a new skill
    # /skill save <name>                            save using last user message
    # /skill list                                   list all saved skills
    # /skill show <name>                            show a skill's template + counters
    # /skill run <name> [k=v ...]                   substitute args + dispatch
    # /skill delete <name>                          remove a skill
    if lower.startswith("/skill") or lower.startswith("skill "):
        sub = _handle_skill_command(user_input, sender, commander)
        if sub is not None:
            return sub

    # ── force-recover (2026-04-28) ─────────────────────────────────
    # Triggered when the user explicitly says "force this" / "try
    # harder" / "force recover" right after a refusal-shaped answer.
    # Looks up the previous (user_input, agent_response) from
    # conversation history and re-runs the recovery loop with
    # force=True (bypasses the auto-detector's confidence threshold
    # AND the policy guard).
    _FORCE_PATTERNS = (
        "force this", "force recover", "force-recover",
        "try harder", "try alternative", "try another way",
        "find another way",
    )
    if lower in _FORCE_PATTERNS or any(lower.startswith(p) for p in _FORCE_PATTERNS):
        return _handle_force_recover(sender, commander)


    # "please learn <topic>" / "start learning <topic>" — add to queue AND run now
    _learn_now_match = re.match(
        r"^(?:please\s+)?(?:learn|start\s+learn(?:ing)?)\s+(.+)",
        lower,
    )
    if _learn_now_match:
        topic = _learn_now_match.group(1).strip()[:200]
        topic = re.sub(r'[^a-zA-Z0-9 _\-,.]', '', topic).strip()
        if not topic:
            return "Please provide a valid topic to learn."
        _QUEUE_ROOT = Path("/app/workspace")
        queue_file = Path(settings.self_improve_topic_file).resolve()
        try:
            queue_file.relative_to(_QUEUE_ROOT)
        except ValueError:
            return "Configuration error: learning queue path is outside workspace."
        queue_file.parent.mkdir(parents=True, exist_ok=True)
        with open(queue_file, "a") as f:
            f.write(f"\n{topic}")
        # If user said "please learn" or "start learning", run immediately
        if lower.startswith("please") or "start" in lower:
            try:
                from app.crews.self_improvement_crew import SelfImprovementCrew
                SelfImprovementCrew().run()
                try:
                    from app.memory.system_chronicle import generate_and_save
                    _ctx_pool.submit(generate_and_save)
                except Exception:
                    pass
                return f"Learned about: {topic}. Skill files updated."
            except Exception as e:
                return f"Added '{topic}' to queue but learning failed: {str(e)[:200]}"
        return f"Added to learning queue: {topic}"

    if lower in ("skills", "list skills", "show skills"):
        skills_dir = Path("/app/workspace/skills")
        if not skills_dir.exists():
            return "No skill files yet. Use 'learn <topic>' to start learning."
        files = sorted(skills_dir.glob("*.md"), key=lambda f: f.stat().st_mtime, reverse=True)
        if not files:
            return "No skill files yet."
        total = len(files)
        # Show most recent 20 with domain grouping
        lines = [f"Skill Files: {total} total\n"]
        for f in files[:20]:
            name = f.stem.replace("_", " ").replace("-", " ")
            lines.append(f"  - {name}")
        if total > 20:
            lines.append(f"\n  ...and {total - 20} more. Use 'skills' via Signal for the full list.")
        return "\n".join(lines)

    if lower == "show learning queue":
        _QUEUE_ROOT = Path("/app/workspace")
        queue_file = Path(settings.self_improve_topic_file).resolve()
        try:
            queue_file.relative_to(_QUEUE_ROOT)
        except ValueError:
            return "Configuration error: learning queue path is outside workspace."
        if queue_file.exists():
            content = queue_file.read_text().strip()
            return f"Learning Queue:\n{content}" if content else "Learning queue is empty."
        return "Learning queue is empty."

    if lower == "run self-improvement now":
        from app.crews.self_improvement_crew import SelfImprovementCrew
        SelfImprovementCrew().run()
        try:
            from app.memory.system_chronicle import generate_and_save
            _ctx_pool.submit(generate_and_save)
        except Exception:
            pass
        return "Self-improvement run completed."

    # "watch <youtube_url>" — extract transcript, distill into skill + memory
    if lower.startswith("watch "):
        url = user_input[6:].strip()[:200]
        if "youtu" not in url:
            return "Please provide a YouTube URL. Usage: watch https://youtube.com/watch?v=..."
        from app.crews.self_improvement_crew import SelfImprovementCrew
        return SelfImprovementCrew().learn_from_youtube(url)

    if lower == "improve":
        from app.crews.self_improvement_crew import SelfImprovementCrew
        SelfImprovementCrew().run_improvement_scan()
        return "Improvement scan completed. Use 'proposals' to see results."

    if lower in ("fleet", "models"):
        from app.ollama_native import format_fleet_status
        from app.llm_catalog import format_catalog
        from app.llm_benchmarks import get_summary
        return (
            f"{format_fleet_status()}\n\n"
            f"{format_catalog()}\n\n"
            f"{get_summary()}"
        )

    if lower == "fleet stop all":
        from app.ollama_native import stop_all
        stop_all()
        return "All models unloaded from GPU."

    if lower.startswith("fleet pull "):
        model = user_input[11:].strip()[:60]
        if not model:
            return "Usage: fleet pull <model_name> (e.g. fleet pull gemma3:27b)"
        from app.ollama_native import spawn_model
        try:
            url = spawn_model(model)
            return f"Model {model} pulled and ready at {url}"
        except Exception as exc:
            return f"Failed to pull {model}: {str(exc)[:200]}"

    if lower in ("retrospective", "run retrospective"):
        from app.crews.retrospective_crew import RetrospectiveCrew
        return RetrospectiveCrew().run()

    if lower in ("benchmarks", "show benchmarks"):
        from app.benchmarks import format_benchmarks_for_display
        return format_benchmarks_for_display()

    if lower in ("policies", "show policies"):
        from app.policies.policy_loader import format_policies_for_display, get_policy_stats
        display = format_policies_for_display()
        stats = get_policy_stats()
        if stats:
            display += f"\n\n📊 Stats: {stats.get('count', 0)} policies"
            if stats.get('oldest'):
                display += f", oldest: {stats['oldest'][:10]}"
        return display

    if lower == "evolve":
        from app.evolution import run_evolution_session
        result = run_evolution_session(max_iterations=settings.evolution_iterations)
        try:
            from app.memory.system_chronicle import generate_and_save
            _ctx_pool.submit(generate_and_save)
        except Exception:
            pass
        return f"Evolution session completed:\n{result}"

    if lower == "evolve deep":
        from app.evolution import run_evolution_session
        result = run_evolution_session(max_iterations=settings.evolution_deep_iterations)
        try:
            from app.memory.system_chronicle import generate_and_save
            _ctx_pool.submit(generate_and_save)
        except Exception:
            pass
        return f"Deep evolution session completed:\n{result}"

    if lower in ("experiments", "show experiments"):
        from app.evolution import get_journal_summary
        return f"Experiment History:\n\n{get_journal_summary(15)}"

    if lower in ("results", "show results"):
        from app.results_ledger import format_ledger
        return f"Results Ledger:\n\n{format_ledger(20)}"

    if lower in ("metrics", "show metrics"):
        from app.metrics import compute_metrics, format_metrics
        return f"System Metrics:\n\n{format_metrics(compute_metrics())}"

    # ── LLM mode switching ─────────────────────────────────────────
    if lower.startswith("mode "):
        new_mode = user_input[5:].strip().lower()
        from app.llm_mode import VALID_MODES, set_mode
        if new_mode not in VALID_MODES:
            return f"Invalid mode. Use: {' | '.join('mode ' + m for m in VALID_MODES)}"
        set_mode(new_mode)
        from app.firebase_reporter import report_llm_mode
        report_llm_mode(new_mode)
        return f"LLM mode switched to: {new_mode.upper()}"

    if lower == "mode":
        from app.llm_mode import VALID_MODES, get_mode
        mode = get_mode()
        options = ", ".join(f"'mode {m}'" for m in VALID_MODES)
        return f"Current LLM mode: {mode.upper()}\n\nUse {options} to switch."

    # ── Token usage ───────────────────────────────────────────────────
    if lower in ("tokens", "token usage"):
        from app.llm_benchmarks import format_token_stats
        return format_token_stats("day")

    if lower.startswith("tokens "):
        period = user_input[7:].strip().lower()
        valid_periods = ("hour", "day", "week", "month", "quarter", "year")
        if period not in valid_periods:
            return f"Invalid period. Use: {', '.join(valid_periods)}"
        from app.llm_benchmarks import format_token_stats
        return format_token_stats(period)

    if lower in ("catalog", "show catalog"):
        from app.llm_catalog import format_catalog, format_role_assignments
        return f"{format_catalog()}\n\n{format_role_assignments(settings.cost_mode)}"

    if lower in ("program", "show program"):
        program_path = Path("/app/workspace/program.md")
        if program_path.exists():
            content = program_path.read_text().strip()
            # Truncate for Signal message limits
            if len(content) > 1400:
                content = content[:1400] + "\n\n[truncated]"
            return f"Evolution Program:\n\n{content}"
        return "No program.md found. Create workspace/program.md to guide evolution."

    if lower in ("errors", "show errors"):
        from app.healing.error_diagnosis import get_recent_errors, get_error_patterns
        errors = get_recent_errors(5)
        if not errors:
            return "No errors recorded. System is healthy."
        patterns = get_error_patterns()
        lines = ["Recent Errors:\n"]
        for e in errors:
            status = "fixed" if e.get("diagnosed") else "pending"
            lines.append(
                f"[{e['ts'][:16]}] {e['crew']}: {e['error_type']} — "
                f"{e['error_msg'][:80]} ({status})"
            )
        if patterns:
            lines.append(f"\nPatterns: {', '.join(f'{k}({v}x)' for k,v in list(patterns.items())[:5])}")
        return "\n".join(lines)

    if lower in ("audit", "run audit", "code audit"):
        from app.auditor import run_code_audit
        result = run_code_audit()
        try:
            from app.memory.system_chronicle import generate_and_save
            _ctx_pool.submit(generate_and_save)
        except Exception:
            pass
        return result

    if lower in ("fix errors", "resolve errors"):
        from app.auditor import run_error_resolution
        return run_error_resolution()

    if lower in ("audit status", "auditor"):
        from app.auditor import get_audit_summary, get_error_resolution_status
        from app.auto_deployer import get_deploy_log
        return (
            f"Audit Activity:\n{get_audit_summary(5)}\n\n"
            f"{get_error_resolution_status()}\n\n"
            f"Recent Deploys:\n{get_deploy_log(5)}"
        )

    if lower in ("deploys", "deploy log"):
        from app.auto_deployer import get_deploy_log
        return f"Deploy Log:\n{get_deploy_log(10)}"

    if lower == "auto deploy on":
        import os
        os.environ["EVOLUTION_AUTO_DEPLOY"] = "true"
        return ("✅ Auto-deploy ENABLED. Code mutations that pass all safety checks + "
                "composite_score improvement will deploy automatically with 60s monitoring.\n"
                "Send 'auto deploy off' to disable.")

    if lower == "auto deploy off":
        import os
        os.environ["EVOLUTION_AUTO_DEPLOY"] = "false"
        return "🔒 Auto-deploy DISABLED. Code proposals require human approval."

    if lower == "auto deploy":
        import os
        state = os.environ.get("EVOLUTION_AUTO_DEPLOY", "false")
        return f"Auto-deploy is {'ENABLED ✅' if state == 'true' else 'DISABLED 🔒'}.\nSend 'auto deploy on' or 'auto deploy off' to change."

    # Step 9: diff and rollback commands for governance
    if lower.startswith("diff "):
        try:
            pid = int(user_input.split()[1])
        except (IndexError, ValueError):
            return "Usage: diff <proposal_id>"
        from app.proposals import get_proposal
        p = get_proposal(pid)
        if not p:
            return f"Proposal #{pid} not found."
        lines = [f"Proposal #{pid}: {p.get('title', '')}", f"Type: {p.get('type', '')}", f"Status: {p.get('status', '')}"]
        if p.get("description"):
            lines.append(f"\n{p['description'][:800]}")
        if p.get("files"):
            for fpath, content in p["files"].items():
                lines.append(f"\n--- {fpath} ---\n{content[:500]}")
        return "\n".join(lines)

    if lower.startswith("rollback "):
        try:
            pid = int(user_input.split()[1])
        except (IndexError, ValueError):
            return "Usage: rollback <proposal_id>"
        from app.proposals import get_proposal
        p = get_proposal(pid)
        if not p or p.get("status") != "approved":
            return f"Proposal #{pid} not found or not approved."
        # Check for backup
        from app.auto_deployer import BACKUP_DIR
        backups = sorted(BACKUP_DIR.iterdir()) if BACKUP_DIR.exists() else []
        if not backups:
            return "No backups available for rollback."
        latest_backup = backups[-1]
        # Restore from backup
        import shutil
        restored = []
        for f in latest_backup.rglob("*.py"):
            rel = f.relative_to(latest_backup)
            dest = Path("/app") / rel
            try:
                shutil.copy2(f, dest)
                restored.append(str(rel))
            except OSError as exc:
                return f"Rollback failed: {exc}"
        if restored:
            return f"Rolled back {len(restored)} files: {', '.join(restored[:5])}"
        return "No files found in backup to restore."

    # Step 9: Tech radar command
    if lower in ("tech radar", "tech", "radar", "discoveries"):
        from app.crews.tech_radar_crew import get_recent_discoveries
        discoveries = get_recent_discoveries(10)
        if not discoveries:
            return "No tech discoveries yet. The tech radar runs during idle time."
        lines = ["Recent Tech Discoveries:\n"]
        for d in discoveries:
            lines.append(f"  • {d[:150]}")
        return "\n".join(lines)

    # Step 1: Anomaly alerts command
    if lower in ("anomalies", "alerts"):
        from app.anomaly_detector import get_recent_alerts
        alerts = get_recent_alerts(10)
        if not alerts:
            return "No anomalies detected. System metrics are within normal ranges."
        lines = ["Recent Anomaly Alerts:\n"]
        for a in alerts:
            lines.append(f"  [{a['ts'][:16]}] {a['type']}: {a['metric']}={a['value']} ({a['sigma']}σ {a['direction']})")
        return "\n".join(lines)

    # Step 2: Variant archive command
    if lower in ("variants", "archive", "genealogy"):
        from app.variant_archive import format_archive_context
        return format_archive_context(15)

    if lower in ("proposals", "show proposals"):
        from app.proposals import list_proposals
        pending = list_proposals("pending")
        if not pending:
            return "No pending improvement proposals."
        lines = ["Pending Improvement Proposals:\n"]
        for p in pending:
            lines.append(
                f"#{p['id']} [{p['type']}] {p['title']}\n"
                f"  Created: {p['created_at'][:10]}"
            )
        lines.append("\nReply 'approve <id>' or 'reject <id>'.")
        return "\n".join(lines)

    if lower.startswith("approve "):
        try:
            pid = int(user_input.split()[1])
        except (IndexError, ValueError):
            return "Usage: approve <proposal_id>"
        from app.proposals import approve_proposal
        return approve_proposal(pid)

    if lower.startswith("reject "):
        try:
            pid = int(user_input.split()[1])
        except (IndexError, ValueError):
            return "Usage: reject <proposal_id>"
        from app.proposals import reject_proposal
        return reject_proposal(pid)

    if lower == "status":
        from app.proposals import list_proposals
        from app.metrics import composite_score
        pending = list_proposals("pending")
        pending_str = f" | {len(pending)} pending proposals" if pending else ""
        try:
            score = composite_score()
            score_str = f" | Score: {score:.4f}"
        except Exception:
            score_str = ""
        local_str = " | LLM: local (Ollama)" if is_using_local() else " | LLM: Claude API"
        return f"System is running. All services operational.{pending_str}{score_str}{local_str}"

    if lower in ("llm status", "llm"):
        from app.llm_mode import get_mode
        from app.llm_factory import get_last_model, get_last_tier
        from app.llm_catalog import format_role_assignments
        mode = get_mode()
        last_model = get_last_model() or "none"
        last_tier = get_last_tier() or "none"
        lines = [
            f"LLM Mode: {mode.upper()}",
            f"Cost Mode: {settings.cost_mode}",
            f"Last Model: {last_model} (tier: {last_tier})",
            f"Commander: {settings.commander_model}",
            f"Vetting: {settings.vetting_model} ({'ON' if settings.vetting_enabled else 'OFF'})",
            f"API Tier: {'ON' if settings.api_tier_enabled and settings.openrouter_api_key.get_secret_value() else 'OFF'}",
            f"Local Ollama: {'ON' if settings.local_llm_enabled else 'OFF'}",
            "",
            format_role_assignments(settings.cost_mode),
        ]
        return "\n".join(lines)

    # ── Knowledge base commands ───────────────────────────────────────
    if lower in ("kb", "kb status", "knowledge base"):
        try:
            from app.knowledge_base.vectorstore import KnowledgeStore
            store = KnowledgeStore()
            stats = store.stats()
            lines = [
                f"Knowledge Base: {stats['total_documents']} docs, "
                f"{stats['total_chunks']} chunks, "
                f"~{stats['estimated_tokens']:,} tokens",
            ]
            if stats["categories"]:
                cats = ", ".join(f"{c}({n})" for c, n in sorted(stats["categories"].items()))
                lines.append(f"Categories: {cats}")
            return "\n".join(lines)
        except Exception as exc:
            return f"Knowledge base error: {str(exc)[:200]}"

    if lower == "kb list":
        try:
            from app.knowledge_base.vectorstore import KnowledgeStore
            store = KnowledgeStore()
            docs = store.list_documents()
            if not docs:
                return "Knowledge base is empty."
            lines = [f"Knowledge Base ({len(docs)} documents):\n"]
            for d in docs[:20]:
                lines.append(
                    f"  {d['source']} ({d['format']}) | "
                    f"{d['category']} | {d['total_chunks']} chunks"
                )
            return "\n".join(lines)
        except Exception as exc:
            return f"Knowledge base error: {str(exc)[:200]}"

    if lower.startswith("kb remove "):
        source_path = user_input[10:].strip()
        if not source_path:
            return "Usage: kb remove <source_path>"
        try:
            from app.knowledge_base.vectorstore import KnowledgeStore
            store = KnowledgeStore()
            count = store.remove_document(source_path)
            if count:
                return f"Removed {count} chunks from '{source_path}'"
            return f"No document found: '{source_path}'"
        except Exception as exc:
            return f"Knowledge base error: {str(exc)[:200]}"

    if lower.startswith("kb add"):
        # "kb add" with attachments → ingest each attachment
        # "kb add <url> [category]" → ingest a URL
        source_text = user_input[6:].strip()
        category = "general"
        try:
            from app.knowledge_base.vectorstore import KnowledgeStore
            store = KnowledgeStore()

            # If attachments are present, ingest them into the KB
            # Note: attachments are passed via commander.handle() — we access
            # them indirectly through the commander instance's last call context.
            # However, for the command handler, attachments are passed as a
            # parameter to handle() and we need to check the original attachments.
            # Since try_command doesn't receive attachments directly, we check
            # if there's attachment context by looking at the source_text.
            # For attachment-based kb add, the caller should handle it.

            # No attachments path — treat as URL/path
            if not source_text:
                return (
                    "Usage:\n"
                    "  kb add <url> [category] — ingest a URL\n"
                    "  Send file + 'kb add [category]' — ingest attachment"
                )
            parts = source_text.split(None, 1)
            url_or_path = parts[0]
            category = parts[1] if len(parts) > 1 else "general"
            result = store.add_document(source=url_or_path, category=category)
            if result.success:
                return (
                    f"Ingested '{result.source}': "
                    f"{result.chunks_created} chunks, "
                    f"{result.total_characters:,} chars ({category})"
                )
            return f"Failed: {result.error}"
        except Exception as exc:
            return f"Ingestion error: {str(exc)[:200]}"

    if lower == "kb reset":
        try:
            from app.knowledge_base.vectorstore import KnowledgeStore
            store = KnowledgeStore()
            store.reset()
            return "Knowledge base has been reset."
        except Exception as exc:
            return f"Knowledge base error: {str(exc)[:200]}"

    if lower.startswith("kb search "):
        query = user_input[10:].strip()
        if not query:
            return "Usage: kb search <question>"
        try:
            from app.knowledge_base.vectorstore import KnowledgeStore
            store = KnowledgeStore()
            results = store.query(question=query, top_k=5)
            if not results:
                return f"No results found for: '{query}'"
            lines = [f"Found {len(results)} results:\n"]
            for i, r in enumerate(results, 1):
                text_preview = r["text"][:200].replace("\n", " ")
                lines.append(
                    f"{i}. [{r['score']:.0%}] {r['source']} ({r['category']})\n"
                    f"   {text_preview}..."
                )
            return "\n".join(lines)
        except Exception as exc:
            return f"Knowledge base error: {str(exc)[:200]}"

    # ── Control Plane commands ───────────────────────────────────────────
    if settings.control_plane_enabled:

        # ── Project / workspace management ───────────────────────────────
        # The user-facing dashboard calls these "Workspaces" while the
        # control-plane DB schema calls them "Projects" — accept both
        # terms verbatim so the user doesn't have to remember which side
        # of the system they're talking to.
        if lower in (
            "project list", "projects",
            "workspace list", "workspaces",
        ):
            try:
                from app.control_plane.projects import get_projects
                return get_projects().format_list()
            except Exception as exc:
                return f"Error: {str(exc)[:200]}"

        # Match every reasonable phrasing the user might type:
        #   project switch X         / workspace switch X
        #   switch project X         / switch workspace X
        #   switch to project X      / switch to workspace X
        #   switch project to X      / switch workspace to X   ← user's case
        # Project name capture is greedy (.+) so multi-word names like
        # "eesti mets" survive — the earlier regex used \S+ which
        # truncated to the first token. Match against the original
        # (non-lowercased) input so display-case names round-trip
        # correctly via get_by_name (case-insensitive lookup) →
        # canonical_name. The trailing optional "to " strips a connector
        # word the user likely typed between the noun and the name.
        _proj_switch = re.match(
            r"^(?:"
                r"(?:project|workspace)\s+switch"
                r"|"
                r"switch\s+(?:to\s+)?(?:project|workspace)"
            r")\s+(?:to\s+)?(.+)",
            user_input.strip(), re.IGNORECASE,
        )
        if _proj_switch:
            try:
                from app.control_plane.projects import get_projects
                name = _proj_switch.group(1).strip().strip(".,!?")
                result = get_projects().switch(name)
                if result:
                    return f"Switched to project: {result.get('name')} — {result.get('mission', '')[:100]}"
                return (
                    f"Project '{name}' not found. "
                    f"Use `project list` (or `workspaces`) to see available ones."
                )
            except Exception as exc:
                return f"Error: {str(exc)[:200]}"

        # Project / workspace status. Matches both the canonical short
        # forms ("project", "workspace status") AND natural-language
        # questions ("what is the current active workspace", "which
        # project am I on", "where am I"). The natural-language patterns
        # were added 2026-04-30 after the agent answered "what is the
        # current active workspace" by routing to the research crew —
        # which has no project-introspection tool and produced a
        # "missing tool" recovery message instead of a 1-line answer.
        _stripped = lower.rstrip("?.! ").strip()
        _is_status_q = (
            _stripped in (
                "project status", "project",
                "workspace status", "workspace",
                "where am i",
            )
            or bool(re.match(
                r"^(?:"
                    # "what (is) (the) (current|active) (project|workspace)"
                    r"what(?:'s| is)?(?:\s+the)?(?:\s+(?:current|active))*\s+(?:project|workspace)"
                    r"|"
                    # "which (project|workspace) (am I in|am I on|...)"
                    r"which\s+(?:project|workspace)"
                    r"|"
                    # "current (project|workspace)" / "active (project|workspace)"
                    r"(?:current|active)\s+(?:project|workspace)"
                    r"|"
                    # "what (project|workspace) am I (on|in|using|working on)"
                    r"what\s+(?:project|workspace)\s+am\s+i\s+(?:on|in|using|working)"
                r")\b",
                _stripped,
            ))
        )
        if _is_status_q:
            try:
                from app.control_plane.projects import get_projects
                pm = get_projects()
                pid = pm.get_active_project_id()
                status = pm.get_status(pid)
                proj = status.get("project", {})
                tickets = status.get("tickets", {})
                lines = [
                    f"📋 Project: {proj.get('name', '?')}",
                    f"   Mission: {proj.get('mission', '—')[:100]}",
                    f"   Tickets: {tickets.get('todo', 0)} todo, {tickets.get('in_progress', 0)} in progress, "
                    f"{tickets.get('done', 0)} done, {tickets.get('failed', 0)} failed",
                    "",
                    "Switch with `switch workspace to <name>` or list with `workspaces`.",
                ]
                return "\n".join(lines)
            except Exception as exc:
                return f"Error: {str(exc)[:200]}"

        # Definitional: "what is a workspace?" / "what is a project?".
        # Matches the agent's first question that got hallucinated direct-
        # route reply about "UI only". Returns a short factual answer
        # plus the actual command surface so the user knows it's chat-
        # accessible.
        _is_definitional_q = bool(re.match(
            r"^what(?:'s| is)?(?:\s+a)?\s+(?:project|workspace)\b",
            _stripped,
        ))
        if _is_definitional_q:
            try:
                from app.control_plane.projects import get_projects
                pm = get_projects()
                pid = pm.get_active_project_id()
                proj = pm.get_by_id(pid) or {}
                active_name = proj.get("name", "default")
            except Exception:
                active_name = "(unknown)"
            return (
                "A **workspace** (also called a project in the DB schema) is a "
                "named context that scopes tickets, budgets, KB content, audit "
                "logs, and agent memory. Each workspace has isolated state — "
                "switching workspaces changes which data the agents see.\n\n"
                f"You're currently on: **{active_name}**.\n\n"
                "Commands:\n"
                "  • `workspaces` — list available\n"
                "  • `workspace` — show current + ticket counts\n"
                "  • `switch workspace to <name>` — change active workspace"
            )

        # ── Ticket management ─────────────────────────────────────────────
        if lower in ("tickets", "ticket list", "kanban"):
            try:
                from app.control_plane.tickets import get_tickets
                from app.control_plane.projects import get_projects
                pid = get_projects().get_active_project_id()
                board = get_tickets().get_board(pid)
                counts = board.get("counts", {})
                lines = ["🎫 Tickets:"]
                for status_name in ["todo", "in_progress", "review", "done", "failed", "blocked"]:
                    items = board.get("board", {}).get(status_name, [])
                    if items:
                        lines.append(f"\n  {status_name.upper()} ({len(items)}):")
                        for t in items[:5]:
                            lines.append(f"    #{str(t['id'])[:8]} {t.get('title', '—')[:60]}")
                return "\n".join(lines) if len(lines) > 1 else "No tickets yet."
            except Exception as exc:
                return f"Error: {str(exc)[:200]}"

        _ticket_detail = re.match(r"^ticket\s+([0-9a-f-]+)", lower)
        if _ticket_detail:
            try:
                from app.control_plane.tickets import get_tickets
                ticket = get_tickets().get(_ticket_detail.group(1))
                if not ticket:
                    return "Ticket not found."
                lines = [
                    f"🎫 #{str(ticket['id'])[:8]}: {ticket.get('title', '—')[:100]}",
                    f"   Status: {ticket.get('status')} | Priority: {ticket.get('priority')}",
                    f"   Crew: {ticket.get('assigned_crew', '—')} | Agent: {ticket.get('assigned_agent', '—')}",
                    f"   Cost: ${float(ticket.get('cost_usd', 0)):.4f} | Tokens: {ticket.get('tokens_used', 0)}",
                ]
                comments = ticket.get("comments", [])
                if comments:
                    lines.append(f"\n   Comments ({len(comments)}):")
                    for c in comments[-5:]:
                        lines.append(f"   [{c.get('author')}] {str(c.get('content', ''))[:80]}")
                return "\n".join(lines)
            except Exception as exc:
                return f"Error: {str(exc)[:200]}"

        # ── Budget management ─────────────────────────────────────────────
        if lower in ("budget", "budget status", "budgets"):
            try:
                from app.control_plane.budgets import get_budget_enforcer
                from app.control_plane.projects import get_projects
                pid = get_projects().get_active_project_id()
                return get_budget_enforcer().format_status(pid)
            except Exception as exc:
                return f"Error: {str(exc)[:200]}"

        _budget_set = re.match(r"^budget\s+set\s+(\S+)\s+([\d.]+)", lower)
        if _budget_set:
            try:
                from app.control_plane.budgets import get_budget_enforcer
                from app.control_plane.projects import get_projects
                role = _budget_set.group(1)
                amount = float(_budget_set.group(2))
                pid = get_projects().get_active_project_id()
                get_budget_enforcer().set_budget(pid, role, amount)
                return f"Budget set: {role} → ${amount:.2f}/month"
            except Exception as exc:
                return f"Error: {str(exc)[:200]}"

        _budget_override = re.match(r"^budget\s+override\s+(\S+)\s+([\d.]+)", lower)
        if _budget_override:
            try:
                from app.control_plane.budgets import get_budget_enforcer
                from app.control_plane.projects import get_projects
                role = _budget_override.group(1)
                amount = float(_budget_override.group(2))
                pid = get_projects().get_active_project_id()
                get_budget_enforcer().override_budget(pid, role, amount)
                return f"Budget overridden: {role} → ${amount:.2f}/month (unpaused)"
            except Exception as exc:
                return f"Error: {str(exc)[:200]}"

        # ── Governance ────────────────────────────────────────────────────
        if lower in ("pending", "governance", "governance pending"):
            try:
                from app.control_plane.governance import get_governance
                return get_governance().format_pending()
            except Exception as exc:
                return f"Error: {str(exc)[:200]}"

        # ── Audit trail ───────────────────────────────────────────────────
        _audit_cmd = re.match(r"^audit(?:\s+(\S+))?", lower)
        if _audit_cmd and lower.startswith("audit"):
            try:
                from app.control_plane.audit import get_audit
                actor_filter = _audit_cmd.group(1)
                if actor_filter == "costs":
                    from app.control_plane.projects import get_projects
                    pid = get_projects().get_active_project_id()
                    summary = get_audit().cost_summary(pid)
                    lines = ["💰 Cost Audit:"]
                    for row in summary.get("by_actor", [])[:10]:
                        lines.append(f"  {row.get('actor')}: ${float(row.get('total_cost') or 0):.4f} "
                                     f"({row.get('calls')} calls, {row.get('total_tokens') or 0} tokens)")
                    lines.append(f"\n  Total: ${summary.get('total_cost', 0):.4f}")
                    return "\n".join(lines)
                entries = get_audit().query(actor=actor_filter, limit=15)
                if not entries:
                    return "No audit entries found."
                lines = ["📜 Audit Log:"]
                for e in entries:
                    ts = str(e.get("timestamp", ""))[:19]
                    lines.append(f"  {ts} [{e.get('actor')}] {e.get('action')} "
                                 f"{e.get('resource_type', '')}/{str(e.get('resource_id', ''))[:8]}")
                return "\n".join(lines)
            except Exception as exc:
                return f"Error: {str(exc)[:200]}"

        # ── Org chart ─────────────────────────────────────────────────────
        if lower in ("org chart", "org", "team"):
            try:
                from app.control_plane.org_chart import format_org_chart
                return format_org_chart()
            except Exception as exc:
                return f"Error: {str(exc)[:200]}"

    # ── Firecrawl web scraping commands ──────────────────────────────────
    _scrape_match = re.match(r"^scrape\s+(https?://\S+)", lower)
    if _scrape_match:
        try:
            from app.tools.firecrawl_tools import firecrawl_scrape
            url = _scrape_match.group(1)
            return firecrawl_scrape(url)
        except Exception as exc:
            return f"Scrape error: {str(exc)[:200]}"

    _ingest_match = re.match(r"^ingest\s+(https?://\S+)(?:\s+(\S+))?", lower)
    if _ingest_match:
        try:
            from app.tools.firecrawl_tools import ingest_url_to_chromadb
            url = _ingest_match.group(1)
            category = _ingest_match.group(2) or "general"
            result = ingest_url_to_chromadb(url, tags={"category": category})
            if result.get("error"):
                return f"Ingest error: {result['error']}"
            return (
                f"✅ Ingested: {result.get('page_title', url)}\n"
                f"   Chunks: {result.get('chunks_ingested', 0)}\n"
                f"   Hash: {result.get('content_hash', '?')}\n"
                f"   Collection: web_knowledge"
            )
        except Exception as exc:
            return f"Ingest error: {str(exc)[:200]}"

    _crawl_match = re.match(r"^crawl\s+(https?://\S+)(?:\s+(\d+))?", lower)
    if _crawl_match:
        try:
            from app.tools.firecrawl_tools import ingest_crawl_to_chromadb
            url = _crawl_match.group(1)
            max_pages = int(_crawl_match.group(2) or "20")
            if max_pages > 50:
                return "Max pages capped at 50 for safety."
            result = ingest_crawl_to_chromadb(url, max_pages=max_pages)
            if result.get("error"):
                return f"Crawl error: {result['error']}"
            return (
                f"✅ Crawled + ingested: {url}\n"
                f"   Pages: {result.get('pages_ingested', 0)}\n"
                f"   Total chunks: {result.get('total_chunks', 0)}\n"
                f"   Collection: web_knowledge"
            )
        except Exception as exc:
            return f"Crawl error: {str(exc)[:200]}"

    _map_match = re.match(r"^map\s+(https?://\S+)", lower)
    if _map_match:
        try:
            from app.tools.firecrawl_tools import firecrawl_map
            return firecrawl_map(_map_match.group(1))
        except Exception as exc:
            return f"Map error: {str(exc)[:200]}"

    # ── Composio SaaS integration commands ──────────────────────────────
    if lower in ("composio", "composio status", "integrations"):
        try:
            from app.tools.composio_tool import format_status
            return format_status()
        except Exception as exc:
            return f"Composio error: {str(exc)[:200]}"

    if lower in ("composio apps", "composio connected"):
        try:
            from app.tools.composio_tool import list_connected_apps
            info = list_connected_apps()
            if not info.get("connected"):
                return "No apps connected. Visit https://app.composio.dev/connections"
            lines = ["🔌 Connected apps:"]
            for app in info["connected"]:
                lines.append(f"  {app.get('app', '?')} — {app.get('status', '?')}")
            return "\n".join(lines)
        except Exception as exc:
            return f"Error: {str(exc)[:200]}"

    # ── Host Bridge commands ─────────────────────────────────────────────
    if lower in ("bridge", "bridge status"):
        try:
            from app.bridge_client import get_bridge
            bridge = get_bridge("commander")
            if not bridge:
                return "Host Bridge: not configured (no token for commander)"
            if bridge.is_available():
                status = bridge.status()
                return (
                    f"🌉 Host Bridge: online\n"
                    f"  Host: {status.get('hostname', '?')}\n"
                    f"  OS: {status.get('os', '?')} {status.get('arch', '?')}\n"
                    f"  Python: {status.get('python', '?')}"
                )
            return "🌉 Host Bridge: offline (port 9100 unreachable)"
        except Exception as exc:
            return f"Bridge error: {str(exc)[:200]}"

    # ── LLM Discovery commands ──────────────────────────────────────────
    if lower in ("discover models", "discover", "model discovery"):
        try:
            from app.llm_discovery import run_discovery_cycle
            result = run_discovery_cycle(max_benchmarks=2)
            return (
                f"🔍 LLM Discovery:\n"
                f"  Scanned: {result.get('scanned', 0)} models\n"
                f"  New found: {result.get('new_found', 0)}\n"
                f"  Benchmarked: {result.get('benchmarked', 0)}\n"
                f"  Promoted: {result.get('promoted', 0)}\n"
                f"  Pending approval: {result.get('proposals', 0)}"
            )
        except Exception as exc:
            return f"Discovery error: {str(exc)[:200]}"

    if lower in ("discovered", "discovered models", "models discovered"):
        try:
            from app.llm_discovery import format_discovery_report
            return format_discovery_report()
        except Exception as exc:
            return f"Error: {str(exc)[:200]}"

    # "llm ranks <model>" — show external ranking breakdown
    if lower.startswith("llm ranks"):
        model_key = content.split(None, 2)[2].strip() if len(content.split(None, 2)) > 2 else ""
        if not model_key:
            return "Usage: llm ranks <catalog-key>   e.g. 'llm ranks deepseek-v3.2'"
        try:
            from app.llm_external_ranks import format_ranks
            return format_ranks(model_key)
        except Exception as exc:
            return f"Error: {str(exc)[:200]}"

    if lower in ("refresh ranks", "llm refresh ranks", "ranks refresh"):
        try:
            from app.llm_external_ranks import refresh_all
            summary = refresh_all(force=True)
            return (
                "🔄 External ranks refreshed:\n"
                f"  OpenRouter: {summary.get('openrouter', 0)} rows\n"
                f"  HuggingFace: {summary.get('huggingface', 0)} rows\n"
                f"  Artificial Analysis: {summary.get('artificial_analysis', 0)} rows"
            )
        except Exception as exc:
            return f"Error: {str(exc)[:200]}"

    # "refresh catalog" — force-rebuild the live LLM catalog from AA /
    # OpenRouter / Ollama. Useful after a new model launches and you
    # don't want to wait for the 24h idle-scheduler tick.
    if lower in ("refresh catalog", "catalog refresh", "llm refresh catalog"):
        try:
            from app.llm_catalog_builder import refresh, format_refresh_summary
            summary = refresh(force=True)
            return format_refresh_summary(summary)
        except Exception as exc:
            return f"Error: {str(exc)[:200]}"

    # "rebenchmark <model>" — force a full multi-role rebenchmark and
    # report drift vs. the prior strengths columns.
    if lower.startswith("rebenchmark"):
        parts = user_input.split(None, 1)
        model_key = parts[1].strip() if len(parts) > 1 else ""
        if not model_key:
            return "Usage: rebenchmark <catalog-key>  e.g. 'rebenchmark claude-sonnet-4.6'"
        try:
            from app.llm_discovery import rebenchmark_incumbent
            summary = rebenchmark_incumbent(model_key)
            if summary.get("error"):
                return f"Rebenchmark error: {summary['error']}"
            lines = [f"🔁 Rebenchmark {summary['model']}:"]
            for role, new in summary["new_scores"].items():
                old = summary["old_scores"].get(role, 0.0)
                drift = summary["drift"].get(role, 0.0)
                lines.append(f"  {role:<14} {old:.2f} → {new:.2f} ({drift:+.2f})")
            if summary.get("alerted"):
                lines.append("⚠️  Drift alert filed to governance.")
            return "\n".join(lines)
        except Exception as exc:
            return f"Error: {str(exc)[:200]}"

    # ── LLM promotions (global "first choice" flag) ─────────────────
    if lower.startswith("promote "):
        model_key = user_input[len("promote "):].strip()
        if not model_key:
            return "Usage: promote <catalog-key>   e.g. 'promote kimi-k2.6'"
        try:
            from app.llm_promotions import promote
            ok = promote(model_key, promoted_by=f"signal:{sender or 'user'}",
                         reason="signal promote command")
            if not ok:
                return f"❌ {model_key!r} not in live catalog. Run `refresh catalog` first."
            return f"🚀 Promoted {model_key}. It'll be first-choice wherever it fits."
        except Exception as exc:
            return f"Error: {str(exc)[:200]}"

    if lower.startswith("demote "):
        model_key = user_input[len("demote "):].strip()
        if not model_key:
            return "Usage: demote <catalog-key>"
        try:
            from app.llm_promotions import demote
            demote(model_key)
            return f"Demoted {model_key}. Back in the regular pool."
        except Exception as exc:
            return f"Error: {str(exc)[:200]}"

    if lower in ("promoted", "list promoted", "show promoted"):
        try:
            from app.llm_promotions import format_promotions
            return format_promotions()
        except Exception as exc:
            return f"Error: {str(exc)[:200]}"

    # ── Hand pins (hard override per role × cost_mode) ──────────────
    if lower.startswith("pin "):
        # Syntax: pin <role> <cost_mode> <model>
        parts = user_input.split(None, 3)
        if len(parts) < 4:
            return "Usage: pin <role> <cost_mode> <model>   e.g. 'pin commander balanced claude-opus-4.7'"
        _, role, cost_mode, model_key = parts
        try:
            from app.llm_role_assignments import pin_role
            ok = pin_role(role.lower(), cost_mode.lower(), model_key.strip(),
                          assigned_by=f"signal:{sender or 'user'}",
                          reason="signal pin command")
            if not ok:
                return f"❌ Pin rejected — {model_key!r} not in live catalog."
            return (f"📌 Pinned {role} [{cost_mode}] → {model_key}. "
                    f"Resolver will return this model for that pair until unpinned.")
        except Exception as exc:
            return f"Error: {str(exc)[:200]}"

    if lower.startswith("unpin "):
        # Syntax: unpin <role> <cost_mode>
        parts = user_input.split()
        if len(parts) < 3:
            return "Usage: unpin <role> <cost_mode>   e.g. 'unpin commander balanced'"
        _, role, cost_mode = parts[:3]
        try:
            from app.llm_role_assignments import unpin_role
            n = unpin_role(role.lower(), cost_mode.lower())
            if n == 0:
                return f"No active hand-pins for {role} [{cost_mode}]."
            return f"Unpinned {role} [{cost_mode}] ({n} row{'s' if n!=1 else ''} retired). Resolver takes back over."
        except Exception as exc:
            return f"Error: {str(exc)[:200]}"

    if lower in ("pinned", "list pins", "show pins"):
        try:
            from app.llm_role_assignments import format_pins
            return format_pins()
        except Exception as exc:
            return f"Error: {str(exc)[:200]}"

    # ── PIM shortcut commands ──────────────────────────────────────
    if lower in ("check email", "email", "inbox"):
        try:
            from app.crews.pim_crew import PIMCrew
            return PIMCrew().run("Check my inbox for unread emails and summarize the most important ones")
        except Exception as exc:
            return f"Error: {str(exc)[:200]}"

    if lower in ("calendar", "schedule", "events", "today"):
        try:
            from app.crews.pim_crew import PIMCrew
            return PIMCrew().run("Show me my calendar events for today and tomorrow")
        except Exception as exc:
            return f"Error: {str(exc)[:200]}"

    if lower in ("tasks", "todo", "task list"):
        try:
            from app.tools.task_tools import create_task_tools
            tools = create_task_tools("commander")
            for t in tools:
                if t.name == "list_tasks":
                    return t._run(status="active")
            return "Task tools not available."
        except Exception as exc:
            return f"Error: {str(exc)[:200]}"

    if lower in ("schedules", "show schedules", "list schedules"):
        try:
            from app.tools.schedule_manager_tools import create_schedule_tools
            tools = create_schedule_tools("commander")
            for t in tools:
                if t.name == "list_schedules":
                    return t._run()
            return "Schedule tools not available."
        except Exception as exc:
            return f"Error: {str(exc)[:200]}"

    # ── Natural-language scheduling (T2-6) ───────────────────────────────
    # Usage: "schedule <task> <natural-language-time>"
    #        e.g. "schedule daily briefing every weekday at 7am"
    _schedule_match = re.match(r"^schedule\s+(.+?)\s+((?:every|each|on|at|daily|hourly|weekdays?|weekends?|mon|tue|wed|thu|fri|sat|sun).*)$", lower)
    if _schedule_match:
        task = _schedule_match.group(1).strip()[:200]
        when = _schedule_match.group(2).strip()[:200]
        try:
            from app.cron.nl_parser import nl_to_cron, describe_cron
            cron_expr = nl_to_cron(when)
            if not cron_expr:
                return f"Could not parse schedule: {when!r}. Try 'every day at 7am' or 'weekdays at 9:30'."
            from apscheduler.triggers.cron import CronTrigger
            import uuid
            job_id = f"nl_{uuid.uuid4().hex[:8]}"
            from app.main import scheduler as _sched

            def _run_nl_job(_task=task, _sender=sender):
                try:
                    commander.handle(_task, _sender, [])
                except Exception:
                    logger.exception(f"NL scheduled job failed: {_task[:60]}")

            _sched.add_job(
                _run_nl_job,
                CronTrigger.from_crontab(cron_expr),
                id=job_id,
                name=task,
                replace_existing=True,
            )
            # Persist so schedules survive restart
            try:
                _persist_nl_job(job_id, task, cron_expr, sender)
            except Exception:
                logger.debug("Failed to persist NL job", exc_info=True)
            return (
                f"✅ Scheduled job `{job_id}`: {task}\n"
                f"   Cron: `{cron_expr}` ({describe_cron(cron_expr)})\n"
                f"   Use `jobs` to list, `cancel {job_id}` to remove."
            )
        except Exception as exc:
            return f"Schedule error: {str(exc)[:200]}"

    if lower in ("jobs", "list jobs", "show jobs"):
        try:
            from app.main import scheduler as _sched
            jobs = _sched.get_jobs()
            if not jobs:
                return "No scheduled jobs."
            lines = [f"🗓️  Scheduled jobs ({len(jobs)}):"]
            for j in jobs:
                next_run = j.next_run_time.isoformat() if j.next_run_time else "—"
                lines.append(f"  - {j.id}: {j.name or j.id} → next {next_run}")
            return "\n".join(lines)
        except Exception as exc:
            return f"Jobs error: {str(exc)[:200]}"

    _cancel_match = re.match(r"^cancel\s+([A-Za-z0-9_\-]+)$", lower)
    if _cancel_match:
        job_id = _cancel_match.group(1)
        try:
            from app.main import scheduler as _sched
            _sched.remove_job(job_id)
            _delete_nl_job(job_id)
            return f"✅ Cancelled job `{job_id}`."
        except Exception as exc:
            return f"Cancel error: {str(exc)[:200]}"

    # ── /compress and /usage (T3-11) ─────────────────────────────────────
    if lower in ("/compress", "compress"):
        try:
            from app.history_compression import get_history
            from app.security import _sender_hash
            h = get_history(_sender_hash(sender))
            before = h.total_tokens
            h.compress()
            after = h.total_tokens
            stats = h.get_stats()
            return (
                f"🗜️  Compression ran.\n"
                f"   Tokens: {before} → {after}\n"
                f"   Bulks: {stats['bulks']}, topics: {stats['topics']}, "
                f"current: {stats['current_messages']} msgs\n"
                f"   Utilization: {stats['utilization']}"
            )
        except Exception as exc:
            return f"Compress error: {str(exc)[:200]}"

    if lower in ("/usage", "usage"):
        try:
            from app.history_compression import get_history
            from app.security import _sender_hash
            h = get_history(_sender_hash(sender))
            s = h.get_stats()
            lines = [
                "📊 Session usage:",
                f"  Tokens: {s['total_tokens']} / {s['max_tokens']} ({s['utilization']})",
                f"  Bulks: {s['bulks']}, topics: {s['topics']}, current: {s['current_messages']} msgs",
                f"  Needs compression: {'yes' if s['needs_compression'] else 'no'}",
            ]
            # Day's token usage from tracker
            try:
                from app.rate_throttle import format_token_stats
                lines.append("")
                lines.append(format_token_stats("day"))
            except Exception:
                pass
            return "\n".join(lines)
        except Exception as exc:
            return f"Usage error: {str(exc)[:200]}"

    # ── MCP status (T1-1) ────────────────────────────────────────────────
    if lower in ("mcp", "mcp status", "mcp servers"):
        try:
            from app.mcp.registry import format_status
            return format_status()
        except Exception as exc:
            return f"MCP error: {str(exc)[:200]}"

    # ── Training controls (T4-14) ────────────────────────────────────────
    if lower in ("training", "training status"):
        try:
            from app.training_pipeline import get_orchestrator
            return get_orchestrator().format_report()
        except Exception as exc:
            return f"Training: {str(exc)[:200]}"

    if lower == "train now":
        import threading as _th

        def _bg_train():
            try:
                from app.training_pipeline import run_training_cycle
                result = run_training_cycle()
                logger.info(f"Manual training: {result.get('status')}")
            except Exception:
                logger.error("Manual training failed", exc_info=True)

        _th.Thread(target=_bg_train, daemon=True, name="manual-train").start()
        return "🎓 Training started in background. Check 'training status' later."

    _export_match = re.match(r"^export training\s+(\w+)", lower)
    if _export_match:
        fmt = _export_match.group(1)
        try:
            from app.training_collector import get_pipeline
            result = get_pipeline().export_format(fmt)
            if result.get("error"):
                return f"Export error: {result['error']}"
            return f"✅ Exported {result['exported']} examples ({fmt})\n   Path: {result['path']}"
        except Exception as exc:
            return f"Export error: {str(exc)[:200]}"

    # No command matched
    return None


# ── NL job persistence (T2-6) ────────────────────────────────────────────────

_NL_JOBS_FILE = Path("/app/workspace/nl_jobs.json")


def _persist_nl_job(job_id: str, task: str, cron_expr: str, sender: str) -> None:
    import json as _json
    jobs = _read_nl_jobs()
    jobs[job_id] = {"task": task, "cron": cron_expr, "sender": sender}
    _NL_JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _NL_JOBS_FILE.write_text(_json.dumps(jobs, indent=2))


def _delete_nl_job(job_id: str) -> None:
    jobs = _read_nl_jobs()
    if job_id in jobs:
        import json as _json
        jobs.pop(job_id, None)
        _NL_JOBS_FILE.write_text(_json.dumps(jobs, indent=2))


def _read_nl_jobs() -> dict:
    if not _NL_JOBS_FILE.exists():
        return {}
    try:
        import json as _json
        return _json.loads(_NL_JOBS_FILE.read_text())
    except Exception:
        return {}


def restore_nl_jobs(scheduler, commander) -> int:
    """Re-register persisted NL jobs on startup. Returns count restored."""
    from apscheduler.triggers.cron import CronTrigger
    jobs = _read_nl_jobs()
    restored = 0
    for job_id, data in jobs.items():
        task = data.get("task", "")
        cron_expr = data.get("cron", "")
        sender = data.get("sender", "")
        if not (task and cron_expr and sender):
            continue

        def _make_runner(_task, _sender):
            def _run():
                try:
                    commander.handle(_task, _sender, [])
                except Exception:
                    logger.exception(f"NL scheduled job failed: {_task[:60]}")
            return _run

        try:
            scheduler.add_job(
                _make_runner(task, sender),
                CronTrigger.from_crontab(cron_expr),
                id=job_id, name=task, replace_existing=True,
            )
            restored += 1
        except Exception:
            logger.debug(f"Failed to restore NL job {job_id}", exc_info=True)
    return restored


# ══════════════════════════════════════════════════════════════════════
# force-recover handler (2026-04-28)
# ══════════════════════════════════════════════════════════════════════

def _handle_force_recover(sender: str, commander) -> str:
    """User explicitly asks to retry the previous refusal.

    Walks back through recent conversation history to find the last
    (user message, agent response) pair, then re-runs the recovery
    loop with force=True so the auto-detector's confidence threshold
    is bypassed.

    Returns the recovered text, or a clear "I tried but couldn't
    find anything to recover" message.
    """
    try:
        from app.recovery import maybe_recover, is_enabled
    except Exception:
        return "Recovery loop module not available."

    if not is_enabled():
        return (
            "Recovery loop is disabled. Set RECOVERY_LOOP_ENABLED=true "
            "in .env and restart the gateway."
        )

    # Look up the last user message + last agent response from
    # the conversation store. We need both: the agent's last response
    # is the refusal-shaped text we're trying to recover from; the
    # user's last actual question (skipping the "force this" message
    # itself) is the task we want to re-run.
    try:
        from app.conversation_store import get_history
        history = get_history(sender, limit=20)
    except Exception as exc:
        return f"Could not load conversation history: {exc}"

    # History is a list of {role, content, ts} ordered chronologically.
    # We want the last assistant response and the user message that
    # preceded it (NOT the current "force this" message).
    last_user_question = None
    last_assistant_response = None
    # Walk backward; skip the most recent user message (which is the
    # force-recover command itself).
    seen_force_msg = False
    for entry in reversed(history):
        role = entry.get("role", "")
        content = entry.get("content", "") or entry.get("text", "")
        if not content:
            continue
        if role == "user" and not seen_force_msg:
            seen_force_msg = True
            continue
        if role == "assistant" and last_assistant_response is None:
            last_assistant_response = content
            continue
        if role == "user" and last_user_question is None:
            last_user_question = content
            break

    if not last_assistant_response or not last_user_question:
        return (
            "Couldn't find a previous question + response to recover. "
            "Send your question fresh and use 'force this' if it refuses."
        )

    logger.info(
        "force-recover: replaying last task=%r against response of length %d",
        last_user_question[:80], len(last_assistant_response),
    )

    rec = maybe_recover(
        last_assistant_response,
        last_user_question,
        crew_used="research",   # we don't know which crew — research is
                                # the most permissive default
        commander=commander,
        difficulty=6,
        used_tier="unknown",
        force=True,
    )

    if not rec.triggered:
        return (
            "Force-recover ran but couldn't find anything to recover "
            "from the previous response."
        )
    if rec.success and rec.text:
        suffix = f"\n\n_(force-recover: {', '.join(rec.strategies_tried)})_"
        return rec.text + suffix
    return (
        f"Tried {len(rec.strategies_tried)} alternative routes "
        f"({', '.join(rec.strategies_tried)}) but none produced a "
        f"better answer. The original response stands."
    )


# ── Phase 4 mobile-surface slash commands ────────────────────────────────

def _signal_help() -> str:
    """Compact list of the most useful Signal commands."""
    return (
        "AndrusAI — Signal commands\n"
        "\n"
        "Status & info:\n"
        "  /help                       this list\n"
        "  /status                     uptime, voice mode, scheduled tasks, last error\n"
        "  skills                      list known skill files\n"
        "  show learning queue         pending topics\n"
        "  workspaces                  list workspaces\n"
        "\n"
        "Operations:\n"
        "  learn <topic>               queue a topic for self-improvement\n"
        "  please learn <topic>        queue + run self-improvement now\n"
        "  switch workspace to <name>  change active workspace\n"
        "  watch <youtube url>         distill into skill + memory\n"
        "  force this / try harder     re-run last refusal via the recovery loop\n"
        "\n"
        "Brainstorm:\n"
        "  /brainstorm <topic>         interactive ideation (SCAMPER, Six Hats, …)\n"
        "\n"
        "Voice notes (Settings → Voice mode in /cp/settings):\n"
        "  send a voice note → transcribed; reply comes back as voice if mode != off"
    )


def _signal_status() -> str:
    """Live system status — uptime, voice mode, scheduled tasks, last error.

    Pulls from the same data the React /cp/ops page uses but condensed for a
    quick mobile glance. Designed to fit in a single Signal message bubble.
    """
    lines: list[str] = ["AndrusAI status"]

    # Voice mode
    try:
        from app.runtime_settings import snapshot as rt_snapshot
        rt = rt_snapshot()
        lines.append(f"  voice: {rt['voice_mode']}")
        if rt.get("vision_cu_enabled"):
            cap = rt.get("vision_cu_monthly_cap_usd", 10.0)
            lines.append(f"  vision-cu: on (cap ${cap:.2f}/mo)")
        if rt.get("concierge_persona_enabled"):
            lines.append("  concierge: on")
    except Exception:
        pass

    # Scheduled jobs
    try:
        from main import scheduler
        jobs = scheduler.get_jobs()
        if jobs:
            lines.append(f"  scheduled: {len(jobs)} jobs")
            # First three with their next-run time, abbreviated
            for j in jobs[:3]:
                nrt = j.next_run_time
                when = nrt.strftime("%a %H:%M") if nrt else "—"
                lines.append(f"    · {j.name or j.id}: {when}")
    except Exception:
        pass

    # Web push device count
    try:
        from app.web_push import list_subscriptions, is_configured as wp_configured
        n = len(list_subscriptions())
        if wp_configured():
            lines.append(f"  push: {n} device{'s' if n != 1 else ''}")
        else:
            lines.append("  push: not configured")
    except Exception:
        pass

    # Last error from the structured log (best-effort, last line only)
    try:
        from app.config import get_settings as _gs
        from pathlib import Path
        log_path = Path(_gs().structured_log_path)
        if log_path.exists():
            with log_path.open("rb") as fp:
                fp.seek(0, 2)
                size = fp.tell()
                # Read last ~4 KB and grab the final non-empty line.
                fp.seek(max(0, size - 4096))
                tail = fp.read().decode("utf-8", errors="replace").strip().splitlines()
                if tail:
                    last = tail[-1]
                    import json as _json
                    try:
                        rec = _json.loads(last)
                        msg = rec.get("event") or rec.get("message") or last[:80]
                        lines.append(f"  last error: {str(msg)[:80]}")
                    except (ValueError, TypeError):
                        lines.append(f"  last error: {last[:80]}")
    except Exception:
        pass

    if len(lines) == 1:
        lines.append("  (no metrics available — gateway just booted?)")
    return "\n".join(lines)


# ── Phase 5 skill registry dispatcher ────────────────────────────────────

def _handle_skill_command(user_input: str, sender: str, commander) -> str | None:
    """Dispatch ``/skill ...`` and ``skill ...`` subcommands. Returns a
    response string when the input is claimed, None otherwise."""
    raw = user_input.strip()
    # Strip leading "/skill" or "skill" + whitespace.
    rest = raw[len("/skill"):] if raw.lower().startswith("/skill") else raw[len("skill"):]
    rest = rest.strip()
    if not rest:
        return _skill_help()

    # First word is the subcommand.
    parts = rest.split(None, 1)
    sub = parts[0].lower()
    tail = parts[1] if len(parts) > 1 else ""

    if sub in ("help", "?"):
        return _skill_help()
    if sub in ("list", "ls"):
        return _skill_list()
    if sub == "show":
        return _skill_show(tail)
    if sub == "save":
        return _skill_save(tail, sender)
    if sub == "run":
        return _skill_run(tail, sender, commander)
    if sub in ("delete", "rm", "remove"):
        return _skill_delete(tail)
    return f"Unknown skill subcommand {sub!r}. Try /skill help."


def _skill_help() -> str:
    return (
        "Skill registry — save tasks you run repeatedly.\n"
        "\n"
        "  /skill save <name>: <task template>   save a new skill (use {placeholder} for args)\n"
        "  /skill save <name>                     save using your last user message\n"
        "  /skill list                            list saved skills\n"
        "  /skill show <name>                     show a skill's template + run counters\n"
        "  /skill run <name> [k=v ...]            substitute args and run via commander\n"
        "  /skill delete <name>                   remove a skill\n"
        "\n"
        "Example:\n"
        "  /skill save weekly: Summarize my Q{quarter} week with focus on {topic}\n"
        "  /skill run weekly quarter=2 topic=growth"
    )


def _skill_list() -> str:
    from app.skills import list_skills
    skills = list_skills()
    if not skills:
        return "No skills saved yet. Use `/skill save <name>: <task>` to add one."
    lines = [f"Skills ({len(skills)} total):"]
    for s in skills:
        rate = ""
        if s.run_count:
            pct = 100.0 * s.success_count / s.run_count
            rate = f" — {s.success_count}/{s.run_count} ({pct:.0f}% ok)"
        args_hint = f" args: {', '.join(s.args_schema)}" if s.args_schema else ""
        lines.append(f"  · {s.name}{args_hint}{rate}")
    return "\n".join(lines)


def _skill_show(name: str) -> str:
    from app.skills import get_skill
    if not name:
        return "Usage: /skill show <name>"
    s = get_skill(name)
    if not s:
        return f"No skill named {name!r}. Try /skill list."
    out = [
        f"Skill: {s.name}",
        f"  description: {s.description or '(none)'}",
        f"  template: {s.task_template}",
    ]
    if s.args_schema:
        out.append(f"  args: {', '.join(s.args_schema)}")
    if s.task_hint:
        out.append(f"  hint: {s.task_hint}")
    if s.run_count:
        pct = 100.0 * s.success_count / s.run_count
        out.append(f"  runs: {s.run_count} ({pct:.0f}% ok), last {s.last_run_at}")
    return "\n".join(out)


def _skill_save(tail: str, sender: str) -> str:
    """``save <name>: <template>`` or ``save <name>`` (uses last user message)."""
    from app.skills import save_skill
    if not tail.strip():
        return "Usage: /skill save <name>[: <task template>]"
    name, sep, template = tail.partition(":")
    name = name.strip()
    template = template.strip()
    if not template:
        # Pull the most recent user message before this command from history.
        try:
            from app.conversation_store import get_recent_messages
            history = get_recent_messages(sender, limit=10) or []
        except Exception:
            history = []
        # ``get_recent_messages`` returns newest-first — walk forward and
        # skip the /skill save command itself.
        for entry in history:
            if entry.get("role") != "user":
                continue
            content = (entry.get("content") or "").strip()
            if not content:
                continue
            lc = content.lower()
            if lc.startswith("/skill") or lc.startswith("skill save") or lc.startswith("skill run"):
                continue
            template = content
            break
    if not template:
        return ("Couldn't find a task to save. Run the task once first, "
                "then `/skill save <name>` — or use `/skill save <name>: "
                "<task>` to provide it inline.")
    try:
        skill = save_skill(name=name, task_template=template)
    except ValueError as exc:
        return f"Save failed: {exc}"

    # Audit so the operator can see who saved what.
    try:
        import json as _json
        from app.audit import log_security_event
        log_security_event(
            "skill_save",
            _json.dumps({"name": skill.name, "args": skill.args_schema}),
        )
    except Exception:
        pass

    args_hint = f" — args: {', '.join(skill.args_schema)}" if skill.args_schema else ""
    return f"Saved skill {skill.name!r}{args_hint}.\nRun it with `/skill run {skill.name}`."


def _skill_run(tail: str, sender: str, commander) -> str:
    from app.skills import run_skill
    if not tail.strip():
        return "Usage: /skill run <name> [k=v ...]"

    name, args = _parse_skill_run_args(tail)
    if not name:
        return "Usage: /skill run <name> [k=v ...]"

    try:
        result = run_skill(name, args, sender, commander)
    except KeyError:
        return f"No skill named {name!r}. Try /skill list."
    except ValueError as exc:
        # Missing-args path — the runner returns exactly the placeholder names.
        return f"Skill error: {exc}"

    try:
        import json as _json
        from app.audit import log_security_event
        log_security_event(
            "skill_run",
            _json.dumps({"name": name, "args": list(args.keys())}),
        )
    except Exception:
        pass
    return result


def _parse_skill_run_args(tail: str) -> tuple[str, dict[str, str]]:
    """Split ``"name k1=v1 k2=v2"`` into (name, {k1: v1, k2: v2}).

    Values may be quoted with double quotes if they contain spaces:
        weekly quarter=2 topic="growth and ops"
    """
    import shlex
    try:
        tokens = shlex.split(tail)
    except ValueError:
        tokens = tail.split()
    if not tokens:
        return "", {}
    # Name runs from the start until the first token containing '='.
    name_parts: list[str] = []
    args: dict[str, str] = {}
    arg_started = False
    for tok in tokens:
        if "=" in tok and not arg_started and not tok.startswith("="):
            arg_started = True
        if not arg_started:
            name_parts.append(tok)
            continue
        if "=" not in tok:
            # Stray positional after args started — append to last value.
            if args:
                last_key = list(args.keys())[-1]
                args[last_key] = args[last_key] + " " + tok
            continue
        k, _, v = tok.partition("=")
        args[k.strip()] = v.strip()
    return " ".join(name_parts), args


def _skill_delete(name: str) -> str:
    from app.skills import delete_skill
    if not name:
        return "Usage: /skill delete <name>"
    if delete_skill(name):
        return f"Deleted skill {name!r}."
    return f"No skill named {name!r}."


# ── Long-arc commitment management (Phase F #10) ─────────────────────────


def _commitment_help() -> str:
    return (
        "Commitment commands:\n"
        "  /commitment list                show active commitments\n"
        "  /commitment fulfilled <id>      mark fulfilled (terminal)\n"
        "  /commitment broken <id>         mark broken (terminal)\n"
        "  /commitment deferred <id>       defer + mute nudges\n"
        "  /commitment unmute <id>         resume nudges\n"
    )


def _commitment_list() -> str:
    """List active commitments with their current status + age."""
    try:
        from app.subia.persistence import load_kernel_state
        from datetime import datetime, timezone
    except Exception:
        return "Commitments unavailable: kernel persistence not loaded."
    try:
        kernel = load_kernel_state()
    except Exception:
        return "Commitments unavailable: kernel state could not be read."
    commitments = (
        getattr(getattr(kernel, "self_state", None), "active_commitments", None)
        or []
    )
    if not commitments:
        return "No active commitments."
    lines = ["📋 Active commitments:"]
    now = datetime.now(timezone.utc)
    for c in commitments:
        if hasattr(c, "__dict__"):
            d = dict(c.__dict__)
        elif isinstance(c, dict):
            d = dict(c)
        else:
            continue
        cid = d.get("id", "?")
        desc = (d.get("description") or "")[:80]
        venture = d.get("venture", "?")
        status = d.get("status", "active")
        deadline = d.get("deadline") or "open"
        lines.append(f"  • [{cid}] ({venture}) {desc} — {status}, due {deadline}")
    return "\n".join(lines)


def _commitment_set_status(
    commitment_id: str, new_status: str, *, mute: bool = False,
) -> str:
    """Update status on a commitment by id. Persists kernel + sets mute flag."""
    if not commitment_id:
        return f"Usage: /commitment {new_status} <id>"
    try:
        from app.subia.persistence import load_kernel_state, save_kernel_state
    except Exception:
        return "Commitments unavailable: kernel persistence not loaded."
    try:
        kernel = load_kernel_state()
    except Exception:
        return "Commitments unavailable: kernel state could not be read."

    target = None
    for c in (
        getattr(getattr(kernel, "self_state", None), "active_commitments", None)
        or []
    ):
        cid = (
            getattr(c, "id", None)
            if not isinstance(c, dict) else c.get("id")
        )
        if str(cid) == commitment_id:
            target = c
            break
    if target is None:
        return f"No commitment with id {commitment_id!r}."

    if hasattr(target, "status"):
        target.status = new_status
    elif isinstance(target, dict):
        target["status"] = new_status

    try:
        save_kernel_state(kernel)
    except Exception:
        return "Updated in memory but failed to persist kernel state."

    # Mute the long_arc_follow_up nudges via its state file. The
    # follow-up module owns the state schema; we just flip the flag.
    if mute:
        try:
            from app.life_companion._common import (
                read_state_json, write_state_json,
            )
            state = read_state_json("long_arc_follow_up.json", {})
            by_id = state.setdefault("by_commitment", {})
            entry = by_id.setdefault(commitment_id, {})
            entry["muted"] = True
            write_state_json("long_arc_follow_up.json", state)
        except Exception:
            logger.debug("commitment: mute write failed", exc_info=True)
    return f"Commitment {commitment_id!r} marked {new_status}."


def _commitment_unmute(commitment_id: str) -> str:
    if not commitment_id:
        return "Usage: /commitment unmute <id>"
    try:
        from app.life_companion._common import (
            read_state_json, write_state_json,
        )
        state = read_state_json("long_arc_follow_up.json", {})
        by_id = state.setdefault("by_commitment", {})
        entry = by_id.setdefault(commitment_id, {})
        entry["muted"] = False
        write_state_json("long_arc_follow_up.json", state)
    except Exception:
        return "Could not update mute state."
    return f"Resumed long-arc nudges for {commitment_id!r}."


def _handle_commitment_command(user_input: str) -> str | None:
    """Dispatch ``/commitment <sub> [id]``. Returns response string or
    None if the subcommand wasn't recognised."""
    text = user_input.strip()
    if text.lower().startswith("/commitment"):
        text = text[len("/commitment"):].strip()
    elif text.lower().startswith("commitment "):
        text = text[len("commitment "):].strip()
    if not text or text.lower() in ("help", "?", "/help"):
        return _commitment_help()
    parts = text.split(maxsplit=1)
    sub = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""
    if sub == "list":
        return _commitment_list()
    if sub == "fulfilled":
        return _commitment_set_status(arg, "fulfilled", mute=True)
    if sub == "broken":
        return _commitment_set_status(arg, "broken", mute=True)
    if sub == "deferred":
        return _commitment_set_status(arg, "deferred", mute=True)
    if sub == "unmute":
        return _commitment_unmute(arg)
    return _commitment_help()


# ── Topic dormancy mute/unmute (Phase G #3) ──────────────────────────────


def _topic_help() -> str:
    return (
        "Topic dormancy commands:\n"
        "  /topic mute <name>      silence dormancy nudges for <name>\n"
        "  /topic unmute <name>    resume nudges\n"
    )


def _handle_topic_command(user_input: str) -> str | None:
    text = user_input.strip()
    if text.lower().startswith("/topic"):
        text = text[len("/topic"):].strip()
    elif text.lower().startswith("topic "):
        text = text[len("topic "):].strip()
    if not text or text.lower() in ("help", "?", "/help"):
        return _topic_help()
    parts = text.split(maxsplit=1)
    sub = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""
    if not arg:
        return _topic_help()
    if sub == "mute":
        try:
            from app.life_companion.topic_dormancy import mute as _mute
            _mute(arg)
            return f"Muted dormancy nudges for topic {arg!r}."
        except Exception:
            return "Could not update mute state."
    if sub == "unmute":
        try:
            from app.life_companion.topic_dormancy import unmute as _unmute
            was = _unmute(arg)
            return (
                f"Resumed dormancy nudges for topic {arg!r}."
                if was else f"Topic {arg!r} was not muted."
            )
        except Exception:
            return "Could not update mute state."
    return _topic_help()


# ── Q4#16 (PROGRAM §41) — companion tensions Signal commands ───────


def _tensions_help() -> str:
    return (
        "/tensions               — list open tensions you've left with me\n"
        "/tensions add <q>       — manually file a tension\n"
        "/tensions resolve <id> <text>  — mark RESOLVED with a note"
    )


def _handle_tensions_command(user_input: str) -> str | None:
    text = user_input.strip()
    if text.lower().startswith("/tensions"):
        text = text[len("/tensions"):].strip()
    elif text.lower().startswith("tensions "):
        text = text[len("tensions "):].strip()
    # Empty body = list (default action)
    if not text or text.lower() in ("list", "help", "?"):
        if text.lower() in ("help", "?"):
            return _tensions_help()
        try:
            from app.companion.tensions import list_tensions, STATUS_OPEN
            tensions = list_tensions(status=STATUS_OPEN, min_freshness=0.0) or []
        except Exception:
            return "Could not read tensions."
        if not tensions:
            return "No open tensions. Use /tensions add <question> to file one."
        lines = [f"📋 {len(tensions)} open tension(s):"]
        for t in tensions[:10]:
            lines.append(f"  {t.id}  {t.question[:80]}")
        if len(tensions) > 10:
            lines.append(f"  …and {len(tensions) - 10} more")
        return "\n".join(lines)

    parts = text.split(maxsplit=1)
    sub = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if sub == "add":
        if not arg or len(arg) < 8:
            return "Question too short (need ≥8 chars). Try: /tensions add <question>"
        try:
            from app.companion.tensions import create_tension
            t = create_tension(
                question=arg, detection_source="manual:signal",
            )
        except Exception:
            return "Could not create tension."
        if t is None:
            return (
                "Tension rejected — either OPEN cap (30) reached, or "
                "question length out of bounds. Resolve some existing "
                "tensions first."
            )
        return f"✅ Filed tension {t.id}: {t.question[:100]}"

    if sub == "resolve":
        parts2 = arg.split(maxsplit=1)
        if len(parts2) < 2:
            return "Usage: /tensions resolve <id> <resolution>"
        tid, resolution = parts2[0], parts2[1]
        try:
            from app.companion.tensions import resolve_tension
            t = resolve_tension(tid, resolution)
        except Exception:
            return "Could not resolve tension."
        if t is None:
            return f"Tension {tid!r} not found."
        return f"✅ Resolved {t.id}: {t.question[:80]}"

    return _tensions_help()


# ── Q4.2 (PROGRAM §42) — /person Signal commands ──────────────────


def _person_help() -> str:
    return (
        "/person                                — list tracked people\n"
        "/person mute <email>                   — exclude from surfaces\n"
        "/person unmute <email>                 — re-include\n"
        "/person forget <email>                 — delete profile\n"
        "/person forget-all                     — nuke all person data\n"
        "/person forget-graph                   — delete graph only\n"
        "/person mute-suggestions <email>       — suppress nudges\n"
        "/person opt-out-of-paths <email>       — exclude as conduit\n"
        "/person path-to <email>                — shortest-path query (L4.1)\n"
        "Enable/disable + master switches at /cp/settings."
    )


def _handle_person_command(user_input: str) -> str | None:
    text = user_input.strip()
    if text.lower().startswith("/person"):
        text = text[len("/person"):].strip()
    elif text.lower().startswith("person "):
        text = text[len("person "):].strip()

    if not text or text.lower() in ("list", "help", "?"):
        if text.lower() in ("help", "?"):
            return _person_help()
        try:
            from app.companion.person_model import current_profile
            prof = current_profile()
        except Exception:
            return "Could not read person profile."
        if not prof.get("enabled"):
            return ("Person correlation is disabled. "
                    "Enable in /cp/settings (and read docs/PERSON_CORRELATION.md).")
        people = prof.get("people") or []
        if not people:
            return "No tracked people yet."
        lines = [f"📋 {len(people)} tracked person(s):"]
        for p in people[:15]:
            display = (p.get("display_names") or [""])[0] or p.get("person_id", "?")
            total = p.get("total_occurrences", 0)
            mods = p.get("modality_count", 0)
            lines.append(f"  {display[:40]:<40s}  {total} hits / {mods} modalities")
        if len(people) > 15:
            lines.append(f"  …and {len(people) - 15} more")
        return "\n".join(lines)

    parts = text.split(maxsplit=1)
    sub = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if sub == "mute" and arg:
        from app.companion.person_model import mute
        return f"✅ Muted {arg}" if mute(arg) else f"Already muted: {arg}"
    if sub == "unmute" and arg:
        from app.companion.person_model import unmute
        return f"✅ Unmuted {arg}" if unmute(arg) else f"Was not muted: {arg}"
    if sub == "forget" and arg:
        from app.companion.person_model import forget
        return f"✅ Forgotten {arg}" if forget(arg) else f"Not tracked: {arg}"
    if sub == "forget-all":
        from app.companion.person_model import forget_all
        return f"✅ Forgot {forget_all()} person(s)."
    if sub == "forget-graph":
        try:
            from app.companion.social_graph import forget_graph
            return f"✅ Forgot social graph ({forget_graph()} edges)."
        except Exception:
            return "Could not forget graph."
    if sub == "mute-suggestions" and arg:
        from app.companion.person_suggestions import mute_suggestions_for
        return f"✅ Suggestions muted for {arg}" if mute_suggestions_for(arg) else f"Already muted: {arg}"
    if sub == "opt-out-of-paths" and arg:
        try:
            from app.companion.social_graph import opt_out_of_paths
            return (f"✅ {arg} excluded as path intermediary"
                    if opt_out_of_paths(arg) else f"Already opted out: {arg}")
        except Exception:
            return "Could not opt out (L4 master may be off)."
    if sub == "path-to" and arg:
        try:
            from app.companion.graph_features.shortest_path import find_path
            result = find_path(source="andrus", target=arg)
        except Exception:
            return "Path query failed."
        if not result.get("ok"):
            return f"No path: {result.get('error') or 'unknown'}"
        path = result.get("path") or []
        hops = result.get("hops", 0)
        if hops == 0:
            return "You ARE the target."
        return f"Path ({hops} hops): " + " → ".join(path)


# ── Q8.1 (PROGRAM §46.1) — /thread Signal commands ─────────────────


def _thread_help() -> str:
    return (
        "/thread                                — list open threads\n"
        "/thread start <title>                   — create a new thread\n"
        "/thread status [id]                     — show thread (last open if no id)\n"
        "/thread list                            — same as bare /thread\n"
        "/thread note <id> <text>                — append a note\n"
        "/thread subq <id> <text>                — add a sub-question\n"
        "/thread done <id> <subq_id> [res]       — resolve a sub-question\n"
        "/thread block <id> <reason>             — file a blocker (status → BLOCKED)\n"
        "/thread unblock <id>                    — clear all blockers + IN_PROGRESS\n"
        "/thread hint <id> <text>                — append an unblock hypothesis\n"
        "/thread resolve <id> [summary]          — close as RESOLVED\n"
        "/thread abandon <id> <reason>           — close as ABANDONED\n"
        "Full surface at /cp/threads (React) and /api/cp/threads (REST)."
    )


def _format_thread_brief(t) -> str:
    """One-line summary for list views."""
    status_icon = {
        "open": "🟢", "in_progress": "🔵",
        "blocked": "🟡", "resolved": "✅", "abandoned": "⚫",
    }.get(getattr(t.status, "value", str(t.status)), "•")
    short_id = t.id[:8]
    n_open = len(t.open_subquestions)
    n_blocked = len(t.blockers)
    parts = [f"{status_icon} {short_id}  {t.title[:60]}"]
    suf: list[str] = []
    if n_open:
        suf.append(f"{n_open} open")
    if n_blocked:
        suf.append(f"{n_blocked} blocked")
    if suf:
        parts.append(f"  ({', '.join(suf)})")
    return "".join(parts)


def _resolve_thread_arg(thread_id_or_short: str):
    """Look up by full id OR by 8-char prefix (the short form shown
    in list views). Returns the Thread or None."""
    from app.threads import get as get_thread, list_all
    if not thread_id_or_short:
        return None
    full = get_thread(thread_id_or_short)
    if full is not None:
        return full
    short = thread_id_or_short.strip().lower()
    if not short:
        return None
    for t in list_all(limit=500):
        if t.id.startswith(short):
            return t
    return None


def _handle_thread_command(user_input: str) -> str | None:
    text = user_input.strip()
    if text.lower().startswith("/thread"):
        text = text[len("/thread"):].strip()
    elif text.lower().startswith("thread "):
        text = text[len("thread "):].strip()
    elif text.lower() == "thread":
        text = ""
    else:
        return None

    # Empty body or "list" = list open
    if not text or text.lower() in ("list", "help", "?"):
        if text.lower() in ("help", "?"):
            return _thread_help()
        try:
            from app.threads import list_open
            threads = list_open(limit=10) or []
        except Exception:
            return "Could not read threads."
        if not threads:
            return "No open threads. Use /thread start <title> to file one."
        lines = [f"📋 {len(threads)} open thread(s):"]
        for t in threads:
            lines.append("  " + _format_thread_brief(t))
        lines.append("Use /thread status <id> for detail.")
        return "\n".join(lines)

    parts = text.split(maxsplit=1)
    sub = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if sub == "start":
        if not arg or len(arg) < 4:
            return "Title too short (need ≥4 chars). Try: /thread start <title>"
        try:
            from app.threads import create_thread
            t = create_thread(title=arg)
        except ValueError as exc:
            return f"Cannot create thread: {exc}"
        except Exception:
            return "Could not create thread."
        return f"✅ Filed thread {t.id[:8]}: {t.title[:80]}"

    if sub == "status":
        # Without an id, surface the most recently touched open thread.
        if not arg:
            try:
                from app.threads import list_open
                threads = list_open(limit=1) or []
            except Exception:
                return "Could not read threads."
            if not threads:
                return "No open threads."
            t = threads[0]
        else:
            t = _resolve_thread_arg(arg.split()[0])
            if t is None:
                return f"Thread {arg!r} not found."
        lines = [
            f"📋 {t.title}",
            f"   id: {t.id}",
            f"   status: {t.status.value}",
            f"   created: {t.created_at[:16]}  last touched: {t.last_touched_at[:16]}",
        ]
        if t.sub_questions:
            lines.append(f"   sub-questions ({len(t.resolved_subquestions)}/{len(t.sub_questions)} resolved):")
            for sq in t.sub_questions[:5]:
                mark = "✅" if sq.resolved else "  "
                lines.append(f"     {mark} {sq.id[:8]}  {sq.text[:60]}")
        if t.blockers:
            lines.append(f"   🚧 blockers ({len(t.blockers)}):")
            for b in t.blockers[:5]:
                lines.append(f"     • {b[:80]}")
        if t.unblock_hints:
            lines.append(f"   💡 unblock hints ({len(t.unblock_hints)}):")
            for h in t.unblock_hints[:5]:
                lines.append(f"     • {h[:80]}")
        if t.notes:
            lines.append(f"   📝 notes ({len(t.notes)}):")
            for n in t.notes[-3:]:
                lines.append(f"     • {n[:80]}")
        return "\n".join(lines)

    # Two-arg helpers — split id + body
    if sub in ("note", "block", "hint", "abandon"):
        parts2 = arg.split(maxsplit=1)
        if len(parts2) < 2:
            return f"Usage: /thread {sub} <id> <text>"
        tid, body = parts2[0], parts2[1]
        t = _resolve_thread_arg(tid)
        if t is None:
            return f"Thread {tid!r} not found."
        try:
            if sub == "note":
                from app.threads import record_note
                t2 = record_note(t.id, body)
                return f"✅ Noted on {t2.id[:8]}."
            if sub == "block":
                from app.threads import mark_blocked
                t2 = mark_blocked(t.id, blocker=body)
                return f"🚧 Blocked {t2.id[:8]}: {body[:80]}"
            if sub == "hint":
                from app.threads import add_unblock_hint
                t2 = add_unblock_hint(t.id, body)
                return f"💡 Hint added to {t2.id[:8]}."
            if sub == "abandon":
                from app.threads import abandon_thread
                t2 = abandon_thread(t.id, reason=body)
                return f"⚫ Abandoned {t2.id[:8]}: {body[:80]}"
        except Exception as exc:
            return f"Could not {sub}: {exc}"

    if sub == "subq":
        parts2 = arg.split(maxsplit=1)
        if len(parts2) < 2:
            return "Usage: /thread subq <id> <question>"
        tid, body = parts2[0], parts2[1]
        t = _resolve_thread_arg(tid)
        if t is None:
            return f"Thread {tid!r} not found."
        try:
            from app.threads import add_subquestion
            t2 = add_subquestion(t.id, body)
        except Exception as exc:
            return f"Could not add sub-question: {exc}"
        sq = t2.sub_questions[-1]
        return f"✅ Sub-question {sq.id[:8]} added to {t2.id[:8]}."

    if sub == "done":
        # /thread done <thread_id> <subq_id> [resolution]
        tokens = arg.split(maxsplit=2)
        if len(tokens) < 2:
            return "Usage: /thread done <thread_id> <subq_id> [resolution]"
        tid, sqid = tokens[0], tokens[1]
        res = tokens[2] if len(tokens) > 2 else ""
        t = _resolve_thread_arg(tid)
        if t is None:
            return f"Thread {tid!r} not found."
        # Resolve subq by full id OR prefix
        target_sq_id = None
        for sq in t.sub_questions:
            if sq.id == sqid or sq.id.startswith(sqid.lower()):
                target_sq_id = sq.id
                break
        if target_sq_id is None:
            return f"Sub-question {sqid!r} not found in {t.id[:8]}."
        try:
            from app.threads import resolve_subquestion
            t2 = resolve_subquestion(t.id, target_sq_id, res)
        except Exception as exc:
            return f"Could not resolve sub-question: {exc}"
        return f"✅ Resolved sub-question {target_sq_id[:8]} on {t2.id[:8]}."

    if sub == "unblock":
        if not arg:
            return "Usage: /thread unblock <id>"
        t = _resolve_thread_arg(arg.split()[0])
        if t is None:
            return f"Thread {arg!r} not found."
        try:
            from app.threads import clear_blockers
            t2 = clear_blockers(t.id)
        except Exception:
            return "Could not clear blockers."
        return f"✅ Unblocked {t2.id[:8]} (status → {t2.status.value})."

    if sub == "resolve":
        # /thread resolve <id> [summary]
        tokens = arg.split(maxsplit=1)
        if not tokens:
            return "Usage: /thread resolve <id> [summary]"
        tid = tokens[0]
        summary = tokens[1] if len(tokens) > 1 else ""
        t = _resolve_thread_arg(tid)
        if t is None:
            return f"Thread {tid!r} not found."
        try:
            from app.threads import resolve_thread
            t2 = resolve_thread(t.id, summary=summary)
        except Exception as exc:
            return f"Could not resolve thread: {exc}"
        return f"✅ Resolved {t2.id[:8]}: {t2.title[:80]}"

    return _thread_help()

    return _person_help()
