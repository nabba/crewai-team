#!/usr/bin/env python3
"""
Knowledge Base CLI

Usage:
    python ingest.py add <file_or_url> --category policy --tags "Q1,board"
    python ingest.py add-dir /path/to/docs/ --category product --recursive
    python ingest.py query "What is our refund policy?"
    python ingest.py status
    python ingest.py list
    python ingest.py remove <source_path>
    python ingest.py reset  (WARNING: deletes all data)
"""

import sys
import os
import logging
from pathlib import Path

from app.knowledge_base.vectorstore import KnowledgeStore
from app.knowledge_base import config

logging.basicConfig(level=logging.WARNING)


def get_store() -> KnowledgeStore:
    return KnowledgeStore()


def cmd_add(args):
    """Add a document or URL."""
    if len(args) < 1:
        print("Usage: python ingest.py add <file_or_url> [--category CAT] [--tags TAG1,TAG2]")
        sys.exit(1)

    source = args[0]
    category = "general"
    tags = []

    i = 1
    while i < len(args):
        if args[i] in ("--category", "-c") and i + 1 < len(args):
            category = args[i + 1]
            i += 2
        elif args[i] in ("--tags", "-t") and i + 1 < len(args):
            tags = [t.strip() for t in args[i + 1].split(",") if t.strip()]
            i += 2
        else:
            i += 1

    store = get_store()
    print(f"Ingesting {source}...")
    result = store.add_document(source=source, category=category, tags=tags)

    if result.success:
        print(f"Success: {result.source}")
        print(f"  Format: {result.format}")
        print(f"  Chunks: {result.chunks_created}")
        print(f"  Characters: {result.total_characters:,}")
        print(f"  Category: {category}")
        if tags:
            print(f"  Tags: {tags}")
    else:
        print(f"Failed: {result.error}")
        sys.exit(1)


def cmd_add_dir(args):
    """Add all supported documents from a directory."""
    if len(args) < 1:
        print("Usage: python ingest.py add-dir <directory> [--category CAT] [--recursive]")
        sys.exit(1)

    directory = args[0]
    category = "general"
    tags = []
    recursive = False

    i = 1
    while i < len(args):
        if args[i] in ("--category", "-c") and i + 1 < len(args):
            category = args[i + 1]
            i += 2
        elif args[i] in ("--tags", "-t") and i + 1 < len(args):
            tags = [t.strip() for t in args[i + 1].split(",") if t.strip()]
            i += 2
        elif args[i] in ("--recursive", "-r"):
            recursive = True
            i += 1
        else:
            i += 1

    dir_path = Path(directory)
    if not dir_path.is_dir():
        print(f"Error: '{directory}' is not a directory.")
        sys.exit(1)

    pattern = "**/*" if recursive else "*"
    files = [
        f for f in dir_path.glob(pattern)
        if f.is_file() and f.suffix.lower() in config.SUPPORTED_EXTENSIONS
    ]

    if not files:
        print(f"No supported files found in {directory}")
        return

    print(f"Found {len(files)} supported files.\n")
    store = get_store()
    ok = 0
    fail = 0

    for f in files:
        result = store.add_document(source=str(f), category=category, tags=tags)
        if result.success:
            ok += 1
            print(f"  + {f.name} -> {result.chunks_created} chunks")
        else:
            fail += 1
            print(f"  x {f.name}: {result.error}")

    print(f"\nDone: {ok} ingested, {fail} failed")


def cmd_query(args):
    """Query the knowledge base."""
    if len(args) < 1:
        print("Usage: python ingest.py query <question> [--top-k N] [--category CAT]")
        sys.exit(1)

    question = args[0]
    top_k = config.DEFAULT_TOP_K
    category = None

    i = 1
    while i < len(args):
        if args[i] in ("--top-k", "-k") and i + 1 < len(args):
            top_k = int(args[i + 1])
            i += 2
        elif args[i] in ("--category", "-c") and i + 1 < len(args):
            category = args[i + 1]
            i += 2
        else:
            i += 1

    store = get_store()
    results = store.query(question=question, top_k=top_k, category=category)

    if not results:
        print("No relevant results found.")
        return

    for i, r in enumerate(results, 1):
        print(f"\n--- Result {i} ({r['score']:.0%}) | {r['source']} ({r['category']}) ---")
        text = r["text"][:500]
        if len(r["text"]) > 500:
            text += "..."
        print(text)


def cmd_status(args):
    """Show knowledge base statistics."""
    store = get_store()
    stats = store.stats()

    print("Knowledge Base Stats:")
    print(f"  Documents: {stats['total_documents']}")
    print(f"  Chunks: {stats['total_chunks']}")
    print(f"  Characters: {stats['total_characters']:,}")
    print(f"  Est. Tokens: ~{stats['estimated_tokens']:,}")

    if stats["categories"]:
        print("\nCategories:")
        for cat, count in sorted(stats["categories"].items()):
            print(f"  {cat}: {count} chunks")


def cmd_list(args):
    """List all documents."""
    store = get_store()
    docs = store.list_documents()

    if not docs:
        print("Knowledge base is empty.")
        return

    print(f"Documents ({len(docs)} total):\n")
    for doc in docs:
        print(
            f"  {doc['source']} ({doc['format']}) | "
            f"cat: {doc['category']} | "
            f"chunks: {doc['total_chunks']} | "
            f"added: {doc['ingested_at'][:10] if doc['ingested_at'] else '?'}"
        )


def cmd_remove(args):
    """Remove a document by source path."""
    if len(args) < 1:
        print("Usage: python ingest.py remove <source_path>")
        sys.exit(1)

    store = get_store()
    count = store.remove_document(args[0])
    if count:
        print(f"Removed {count} chunks from '{args[0]}'")
    else:
        print(f"No document found with path '{args[0]}'")


def cmd_reset(args):
    """Delete all data."""
    confirm = input("This will delete ALL knowledge base data. Type 'yes' to confirm: ")
    if confirm.strip().lower() != "yes":
        print("Aborted.")
        return
    store = get_store()
    store.reset()
    print("Knowledge base has been reset.")


COMMANDS = {
    "add": cmd_add,
    "add-dir": cmd_add_dir,
    "query": cmd_query,
    "status": cmd_status,
    "list": cmd_list,
    "remove": cmd_remove,
    "reset": cmd_reset,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print("Knowledge Base CLI")
        print(f"Commands: {', '.join(COMMANDS.keys())}")
        print("\nUsage: python ingest.py <command> [args...]")
        sys.exit(1)

    COMMANDS[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    main()
