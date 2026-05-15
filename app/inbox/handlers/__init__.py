"""Inbox content handlers — one module per dispatch kind.

Each module exposes a ``run(path: Path) -> str`` function. The router
catches any raised exception and records ``status="failed"`` for the
file, leaving it in place for operator inspection. On success, the
file moves to ``workspace/inbox/.archive/<YYYY-MM-DD>/``.

PROGRAM §46.7 (Q9.4) — handlers shipped:

  - ``audio_transcribe``    — voice memo / podcast → transcript
  - ``image_vision``        — whiteboard / screenshot → text + structure
  - ``pdf_extract``         — receipt OR document → expense ledger or note
  - ``spreadsheet_to_csv``  — XLSX/ODS → CSV → notes pipeline
  - ``youtube_link``        — .url/.webloc → ``watch <url>`` skill
"""
