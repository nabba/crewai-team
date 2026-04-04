"""
firebase.listeners — Firestore real-time listeners and queue pollers.

Handles mode listener (on_snapshot + polling fallback), KB queue poller,
philosophy queue poller, fiction queue poller, chat inbox poller, and
Firestore mode reading.
"""

import logging
import threading
from typing import Optional

from app.firebase.infra import _get_db, _fire, _now_iso

logger = logging.getLogger(__name__)


# ── LLM mode reading ────────────────────────────────────────────────────────

def read_llm_mode_from_firestore() -> str | None:
    """Read LLM mode from Firestore (dashboard-set value). Returns None if unavailable."""
    db = _get_db()
    if not db:
        return None
    try:
        doc = db.collection("config").document("llm").get()
        if doc.exists:
            mode = doc.to_dict().get("mode")
            if mode in ("local", "cloud", "hybrid", "insane"):
                return mode
    except Exception:
        logger.debug("firebase.listeners: failed to read llm mode from Firestore", exc_info=True)
    return None


# ── Mode listener ────────────────────────────────────────────────────────────

_mode_listener_unsub = None  # Must be kept alive to prevent GC of the listener
_mode_poll_stop = threading.Event()  # Signal to stop the polling thread


def _apply_mode_if_changed(new_mode: str) -> bool:
    """Apply a mode change if it differs from current. Returns True if changed."""
    if new_mode not in ("local", "cloud", "hybrid", "insane"):
        return False
    from app.llm_mode import get_mode, set_mode
    if new_mode != get_mode():
        set_mode(new_mode)
        logger.info(f"firebase.listeners: mode changed from dashboard -> {new_mode}")
        return True
    return False


def start_mode_listener() -> None:
    """Listen for dashboard mode changes via Firestore on_snapshot + polling fallback.

    The on_snapshot gRPC stream can silently drop in Docker containers,
    so we also poll every 15 seconds as a reliable backup.
    """
    global _mode_listener_unsub

    def _listen():
        global _mode_listener_unsub
        db = _get_db()
        if not db:
            return
        try:
            def on_snapshot(doc_snapshot, changes, read_time):
                for snap in doc_snapshot:
                    data = snap.to_dict()
                    new_mode = data.get("mode")
                    if new_mode:
                        _apply_mode_if_changed(new_mode)

            _mode_listener_unsub = (
                db.collection("config").document("llm").on_snapshot(on_snapshot)
            )
            logger.info("firebase.listeners: mode listener started (on_snapshot)")
        except Exception:
            logger.debug("firebase.listeners: mode listener failed", exc_info=True)
    _fire(_listen)

    # Start polling fallback in a daemon thread
    def _poll_mode():
        """Poll Firestore every 15s for mode changes — backup for flaky gRPC streams."""
        while not _mode_poll_stop.wait(15):
            try:
                mode = read_llm_mode_from_firestore()
                if mode:
                    _apply_mode_if_changed(mode)
            except Exception:
                pass  # never crash the poll loop
        logger.debug("firebase.listeners: mode poll stopped")

    t = threading.Thread(target=_poll_mode, daemon=True, name="firebase-mode-poll")
    t.start()
    logger.info("firebase.listeners: mode poll started (15s interval)")


# ── KB Queue Poller ──────────────────────────────────────────────────────────

_kb_poll_stop = threading.Event()


def start_kb_queue_poller() -> None:
    """Poll Firestore kb_queue for pending uploads and ingest them."""

    def _poll_kb():
        import base64
        import tempfile

        while not _kb_poll_stop.wait(10):
            db = _get_db()
            if not db:
                continue
            try:
                docs = (
                    db.collection("kb_queue")
                    .where("status", "==", "pending")
                    .limit(5)
                    .get()
                )
                for snap in docs:
                    data = snap.to_dict()
                    try:
                        content_b64 = data.get("content_b64", "")
                        fname = data.get("filename", "upload.txt")
                        category = data.get("category", "general")
                        action = data.get("action", "upload")

                        # Handle delete action
                        if action == "delete":
                            source_path = data.get("source_path", "")
                            if source_path:
                                from app.knowledge_base.vectorstore import KnowledgeStore
                                ks = KnowledgeStore()
                                removed = ks.remove_document(source_path)
                                snap.reference.update({
                                    "status": "done",
                                    "removed": removed,
                                    "processed_at": _now_iso(),
                                })
                                logger.info(f"firebase.listeners: KB removed '{source_path}' ({removed} chunks)")
                                from app.firebase.publish import report_knowledge_base
                                report_knowledge_base()
                            else:
                                snap.reference.update({"status": "error", "error": "No source_path"})
                            continue

                        raw = base64.b64decode(content_b64)

                        # Use original filename as prefix so ingestion captures it
                        import re as _re
                        safe_prefix = _re.sub(r'[^\w\-.]', '_', fname.rsplit('.', 1)[0])[:40] + "_"
                        suffix = "." + fname.rsplit(".", 1)[-1] if "." in fname else ".txt"
                        with tempfile.NamedTemporaryFile(
                            suffix=suffix, prefix=f"kb_{safe_prefix}",
                            delete=False,
                        ) as tmp:
                            tmp.write(raw)
                            tmp_path = tmp.name

                        try:
                            from app.knowledge_base.vectorstore import KnowledgeStore
                            ks = KnowledgeStore()
                            result = ks.add_document(tmp_path, category=category)

                            # Store original filename mapping in Firestore for dashboard
                            snap.reference.update({
                                "status": "done",
                                "chunks_created": result.chunks_created,
                                "original_filename": fname,
                                "source_path": tmp_path,
                                "processed_at": _now_iso(),
                            })
                            logger.info(f"firebase.listeners: KB ingested '{fname}' -> {result.chunks_created} chunks")
                            from app.firebase.publish import report_knowledge_base
                            report_knowledge_base()
                        finally:
                            import os as _os
                            _os.unlink(tmp_path)

                    except Exception as e:
                        snap.reference.update({
                            "status": "error",
                            "error": str(e)[:200],
                            "processed_at": _now_iso(),
                        })
                        logger.warning(f"firebase.listeners: KB ingest failed for '{data.get('filename')}': {e}")

            except Exception:
                pass
        logger.debug("firebase.listeners: KB poll stopped")

    t = threading.Thread(target=_poll_kb, daemon=True, name="firebase-kb-poll")
    t.start()
    logger.info("firebase.listeners: KB queue poller started (10s interval)")


# ── Philosophy Queue Poller ──────────────────────────────────────────────────

_phil_poll_stop = threading.Event()


def start_phil_queue_poller() -> None:
    """Poll Firestore phil_queue for pending uploads/actions and process them."""

    def _poll_phil():
        while not _phil_poll_stop.wait(10):
            db = _get_db()
            if not db:
                continue
            try:
                docs = (
                    db.collection("phil_queue")
                    .where("status", "==", "pending")
                    .limit(5)
                    .get()
                )
                if not docs:
                    continue

                for snap in docs:
                    data = snap.to_dict()
                    action = data.get("action", "upload")

                    try:
                        if action == "delete":
                            # Delete a text and its chunks
                            fname = data.get("filename", "")
                            if fname:
                                from app.philosophy.vectorstore import get_store
                                store = get_store()
                                removed = store.remove_by_source(fname)
                                # Remove file from disk
                                from pathlib import Path
                                from app.philosophy import config as phil_config
                                fpath = Path(phil_config.TEXTS_DIR) / fname
                                if fpath.exists():
                                    fpath.unlink()
                                snap.reference.update({
                                    "status": "done",
                                    "chunks_removed": removed,
                                    "processed_at": _now_iso(),
                                })
                                logger.info(f"firebase.listeners: phil deleted '{fname}' ({removed} chunks)")
                            else:
                                snap.reference.update({"status": "error", "error": "No filename"})

                        elif action == "reingest":
                            # Re-ingest all texts
                            from app.philosophy.vectorstore import get_store
                            from app.philosophy.ingestion import ingest_directory
                            from pathlib import Path
                            from app.philosophy import config as phil_config
                            store = get_store()
                            store.reset_collection()
                            summary = ingest_directory(Path(phil_config.TEXTS_DIR), store)
                            snap.reference.update({
                                "status": "done",
                                "files_processed": summary.get("files_processed", 0),
                                "total_chunks": summary.get("total_chunks", 0),
                                "processed_at": _now_iso(),
                            })
                            logger.info(f"firebase.listeners: phil reingest complete: {summary}")

                        else:
                            # Default: upload/ingest a new text
                            content = data.get("content", "")
                            fname = data.get("filename", "upload.md")
                            author = data.get("author", "")
                            tradition = data.get("tradition", "")
                            era = data.get("era", "")
                            title = data.get("title", "")

                            if not content:
                                snap.reference.update({"status": "error", "error": "Empty content"})
                                continue

                            # Sanitize filename
                            import re as _re
                            safe_name = _re.sub(r"[^\w\-.]", "_", fname)
                            if not safe_name.endswith((".md", ".txt")):
                                safe_name += ".md"

                            # If no frontmatter in content and metadata provided, prepend it
                            if not content.lstrip().startswith("---") and any([author, tradition, era, title]):
                                fm_lines = ["---"]
                                if author: fm_lines.append(f"author: {author}")
                                if tradition: fm_lines.append(f"tradition: {tradition}")
                                if era: fm_lines.append(f"era: {era}")
                                if title: fm_lines.append(f"title: {title}")
                                fm_lines.append("---\n")
                                content = "\n".join(fm_lines) + content

                            # Save to texts directory
                            from pathlib import Path
                            from app.philosophy import config as phil_config
                            texts_dir = Path(phil_config.TEXTS_DIR)
                            texts_dir.mkdir(parents=True, exist_ok=True)
                            dest = texts_dir / safe_name
                            dest.write_text(content, encoding="utf-8")

                            # Ingest
                            from app.philosophy.ingestion import ingest_text
                            chunks_added = ingest_text(
                                text=content,
                                filename=safe_name,
                                author=author or "Unknown",
                                tradition=tradition or "Unknown",
                                era=era or "Unknown",
                                title=title or fname,
                            )

                            snap.reference.update({
                                "status": "done",
                                "chunks_created": chunks_added,
                                "processed_at": _now_iso(),
                            })
                            logger.info(f"firebase.listeners: phil ingested '{safe_name}' -> {chunks_added} chunks")

                    except Exception as e:
                        snap.reference.update({
                            "status": "error",
                            "error": str(e)[:200],
                            "processed_at": _now_iso(),
                        })
                        logger.warning(f"firebase.listeners: phil queue error for '{data.get('filename', '?')}': {e}")

                # After processing any items, refresh dashboard stats
                from app.firebase.publish import report_philosophy_kb
                report_philosophy_kb()

            except Exception:
                pass
        logger.debug("firebase.listeners: phil poll stopped")

    t = threading.Thread(target=_poll_phil, daemon=True, name="firebase-phil-poll")
    t.start()
    logger.info("firebase.listeners: Philosophy queue poller started (10s interval)")


# ── Fiction Inspiration Queue Poller ──────────────────────────────────────────

_fiction_poll_stop = threading.Event()


def start_fiction_queue_poller() -> None:
    """Poll Firestore fiction_queue for pending uploads/actions."""

    def _poll_fiction():
        while not _fiction_poll_stop.wait(10):
            db = _get_db()
            if not db:
                continue
            try:
                docs = (
                    db.collection("fiction_queue")
                    .where("status", "==", "pending")
                    .limit(5)
                    .get()
                )
                if not docs:
                    continue

                for snap in docs:
                    data = snap.to_dict()
                    action = data.get("action", "upload")

                    try:
                        if action == "delete":
                            fname = data.get("filename", "")
                            if fname:
                                from app.fiction_inspiration import _get_collection as _fc
                                coll = _fc()
                                # Find and delete chunks by filename
                                results = coll.get(
                                    where={"source_file": fname},
                                    include=["metadatas"],
                                )
                                if results["ids"]:
                                    coll.delete(ids=results["ids"])
                                    removed = len(results["ids"])
                                else:
                                    removed = 0
                                # Remove file from disk
                                from pathlib import Path
                                from app.fiction_inspiration import FICTION_LIBRARY_DIR
                                fpath = FICTION_LIBRARY_DIR / fname
                                if fpath.exists():
                                    fpath.unlink()
                                snap.reference.update({
                                    "status": "done",
                                    "chunks_removed": removed,
                                    "processed_at": _now_iso(),
                                })
                                logger.info(f"firebase.listeners: fiction deleted '{fname}' ({removed} chunks)")
                            else:
                                snap.reference.update({"status": "error", "error": "No filename"})

                        elif action == "reingest":
                            from app.fiction_inspiration import ingest_library
                            result = ingest_library()
                            snap.reference.update({
                                "status": "done",
                                "books_ingested": result.get("books_ingested", 0),
                                "total_chunks": result.get("total_chunks", 0),
                                "processed_at": _now_iso(),
                            })
                            logger.info(f"firebase.listeners: fiction reingest complete: {result}")

                        else:
                            # Default: upload/ingest a new fiction book
                            content = data.get("content", "")
                            fname = data.get("filename", "upload.md")
                            author = data.get("author", "")
                            title_val = data.get("title", "")
                            themes = data.get("themes", [])

                            if not content:
                                snap.reference.update({"status": "error", "error": "Empty content"})
                                continue

                            import re as _re
                            safe_name = _re.sub(r"[^\w\-.]", "_", fname)
                            if not safe_name.endswith((".md", ".txt")):
                                safe_name += ".md"

                            # Prepend frontmatter if metadata provided
                            if not content.lstrip().startswith("---") and any([author, title_val, themes]):
                                import json as _json2
                                fm = ["---"]
                                if title_val: fm.append(f"title: \"{title_val}\"")
                                if author: fm.append(f"author: \"{author}\"")
                                if themes:
                                    fm.append("themes:")
                                    for t in themes:
                                        fm.append(f"  - {t}")
                                fm.append("---\n")
                                content = "\n".join(fm) + content

                            # Save to fiction library directory
                            from pathlib import Path
                            from app.fiction_inspiration import FICTION_LIBRARY_DIR
                            FICTION_LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
                            dest = FICTION_LIBRARY_DIR / safe_name
                            dest.write_text(content, encoding="utf-8")

                            # Ingest into ChromaDB
                            from app.fiction_inspiration import ingest_book
                            result = ingest_book(dest)

                            snap.reference.update({
                                "status": "done",
                                "chunks_created": result.get("ingested", 0),
                                "processed_at": _now_iso(),
                            })
                            logger.info(f"firebase.listeners: fiction ingested '{safe_name}' -> {result.get('ingested', 0)} chunks")

                    except Exception as e:
                        snap.reference.update({
                            "status": "error",
                            "error": str(e)[:200],
                            "processed_at": _now_iso(),
                        })
                        logger.warning(f"firebase.listeners: fiction queue error: {e}")

                from app.firebase.publish import report_fiction_library
                report_fiction_library()

            except Exception:
                pass
        logger.debug("firebase.listeners: fiction poll stopped")

    t = threading.Thread(target=_poll_fiction, daemon=True, name="firebase-fiction-poll")
    t.start()
    logger.info("firebase.listeners: Fiction queue poller started (10s interval)")


# ── Chat inbox poller ────────────────────────────────────────────────────────

def start_chat_inbox_poller(handle_fn) -> None:
    """Poll Firestore chat_inbox for messages sent from the dashboard.

    When a new message is found, calls handle_fn(text) which should
    process it exactly like a Signal message and return the response.
    The response is then written back to chat_messages AND sent via Signal.

    Args:
        handle_fn: async function(text: str) -> str -- processes the message
    """
    import asyncio

    _stop = threading.Event()

    def _poll():
        db = _get_db()
        if not db:
            logger.warning("firebase.listeners: chat inbox poller — no Firestore, skipping")
            return

        logger.info("firebase.listeners: chat inbox poller started (3s interval)")
        while not _stop.is_set():
            try:
                docs = (
                    db.collection("chat_inbox")
                    .where("status", "==", "pending")
                    .limit(5)
                    .stream()
                )
                for snap in docs:
                    data = snap.to_dict()
                    text = (data.get("text") or "").strip()
                    if not text:
                        snap.reference.update({"status": "empty"})
                        continue

                    # Mark as processing immediately
                    snap.reference.update({"status": "processing"})
                    logger.info(f"firebase.listeners: chat inbox message: {text[:80]}")

                    # Write the user message to chat_messages so Signal users see it
                    from app.firebase.publish import report_chat_message
                    report_chat_message("user", text, source="dashboard")

                    # Process via the same handler as Signal (sync call)
                    try:
                        result = handle_fn(text)

                        # Write assistant response to chat_messages
                        report_chat_message("assistant", result, source="dashboard")

                        snap.reference.update({
                            "status": "done",
                            "response": result[:4000],
                            "processed_at": _now_iso(),
                        })
                    except Exception as e:
                        snap.reference.update({
                            "status": "error",
                            "error": str(e)[:200],
                            "processed_at": _now_iso(),
                        })
                        report_chat_message("assistant", f"Error: {str(e)[:200]}", source="dashboard")

            except Exception:
                logger.debug("firebase.listeners: chat inbox poll error", exc_info=True)

            _stop.wait(3)  # Poll every 3 seconds for responsive chat

    t = threading.Thread(target=_poll, daemon=True, name="firebase-chat-poll")
    t.start()
    logger.info("firebase.listeners: chat inbox poller started (3s interval)")
