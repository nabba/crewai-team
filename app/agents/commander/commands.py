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

    # Access the shared thread pool from orchestrator
    from app.agents.commander.orchestrator import _ctx_pool

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
        if new_mode not in ("local", "cloud", "hybrid", "insane"):
            return "Invalid mode. Use: mode local, mode cloud, mode hybrid, or mode insane"
        from app.llm_mode import set_mode
        set_mode(new_mode)
        from app.firebase_reporter import report_llm_mode
        report_llm_mode(new_mode)
        return f"LLM mode switched to: {new_mode.upper()}"

    if lower == "mode":
        from app.llm_mode import get_mode
        mode = get_mode()
        return f"Current LLM mode: {mode.upper()}\n\nUse 'mode local', 'mode cloud', or 'mode hybrid' to switch."

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
        from app.self_heal import get_recent_errors, get_error_patterns
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

    # No command matched
    return None
