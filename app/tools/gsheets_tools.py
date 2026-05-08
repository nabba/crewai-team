"""
gsheets_tools.py — agent-callable Google Sheets operations.

Four CrewAI tools:

    create_google_sheet        new spreadsheet with optional headers + initial rows
    read_google_sheet_range    read a cell range as a list-of-lists
    append_google_sheet_row    append one row to a sheet (USER_ENTERED so
                                formulas like "=A1+B1" are evaluated)
    write_google_sheet_range   overwrite a range with a list-of-lists

Both URLs and bare spreadsheet ids are accepted; the helper extracts the
canonical id from URLs of the form
``https://docs.google.com/spreadsheets/d/<ID>/edit``.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_SHEET_URL_RE = re.compile(r"docs\.google\.com/spreadsheets/d/([A-Za-z0-9_-]{20,})")


def _service():
    from app.google_workspace import get_service
    return get_service("sheets")


def _sheet_id(value: str) -> str:
    value = (value or "").strip()
    m = _SHEET_URL_RE.search(value)
    return m.group(1) if m else value


def _create(title: str, headers: list[str] | None = None, rows: list[list] | None = None) -> dict[str, Any]:
    svc = _service()
    if svc is None:
        return {"error": "Sheets not configured"}
    spreadsheet = svc.spreadsheets().create(body={
        "properties": {"title": title},
    }).execute()
    sid = spreadsheet.get("spreadsheetId")

    payload_rows: list[list] = []
    if headers:
        payload_rows.append(list(headers))
    if rows:
        payload_rows.extend([list(r) for r in rows])
    if payload_rows:
        svc.spreadsheets().values().update(
            spreadsheetId=sid,
            range="A1",
            valueInputOption="USER_ENTERED",
            body={"values": payload_rows},
        ).execute()

    return {
        "id": sid,
        "url": f"https://docs.google.com/spreadsheets/d/{sid}/edit",
        "title": title,
        "rows_seeded": len(payload_rows),
    }


def _read(sheet_id: str, range_a1: str) -> dict[str, Any]:
    svc = _service()
    if svc is None:
        return {"error": "Sheets not configured"}
    real = _sheet_id(sheet_id)
    resp = svc.spreadsheets().values().get(
        spreadsheetId=real, range=range_a1,
        valueRenderOption="UNFORMATTED_VALUE",
    ).execute()
    return {
        "id": real,
        "range": resp.get("range", range_a1),
        "values": resp.get("values", []),
    }


def _append(sheet_id: str, row: list) -> dict[str, Any]:
    svc = _service()
    if svc is None:
        return {"error": "Sheets not configured"}
    real = _sheet_id(sheet_id)
    resp = svc.spreadsheets().values().append(
        spreadsheetId=real,
        range="A1",
        valueInputOption="USER_ENTERED",
        insertDataOption="INSERT_ROWS",
        body={"values": [list(row)]},
    ).execute()
    updates = resp.get("updates", {})
    return {
        "id": real,
        "updated_range": updates.get("updatedRange"),
        "rows_appended": updates.get("updatedRows", 0),
    }


def _write(sheet_id: str, range_a1: str, values: list[list]) -> dict[str, Any]:
    svc = _service()
    if svc is None:
        return {"error": "Sheets not configured"}
    real = _sheet_id(sheet_id)
    resp = svc.spreadsheets().values().update(
        spreadsheetId=real,
        range=range_a1,
        valueInputOption="USER_ENTERED",
        body={"values": [list(r) for r in values]},
    ).execute()
    return {
        "id": real,
        "updated_range": resp.get("updatedRange"),
        "updated_cells": resp.get("updatedCells"),
    }


# ── CrewAI tool factory ────────────────────────────────────────────────────

def create_gsheets_tools(agent_id: str = "researcher") -> list:
    try:
        from app.google_workspace import is_configured
        if not is_configured():
            return []
    except Exception:
        return []
    if _service() is None:
        return []

    try:
        from crewai.tools import tool
    except ImportError:
        return []

    @tool("create_google_sheet")
    def create_tool(title: str, headers: str = "", rows: str = "") -> str:
        """Create a new Google Sheet.

        Args:
            title: spreadsheet title.
            headers: comma-separated column names (optional).
            rows: JSON list-of-lists (optional, e.g. "[[1,2,3],[4,5,6]]").
        """
        head_list = [h.strip() for h in headers.split(",") if h.strip()] or None
        row_list: list[list] | None = None
        if rows:
            try:
                parsed = json.loads(rows)
                if isinstance(parsed, list):
                    row_list = [list(r) if isinstance(r, list) else [r] for r in parsed]
            except (json.JSONDecodeError, TypeError):
                pass
        return json.dumps(_create(title=title, headers=head_list, rows=row_list))

    @tool("read_google_sheet_range")
    def read_tool(sheet_id: str, range_a1: str = "A1:Z1000") -> str:
        """Read a sheet range as list-of-lists. range_a1 example: "Sheet1!A1:D20"."""
        return json.dumps(_read(sheet_id, range_a1), ensure_ascii=False)

    @tool("append_google_sheet_row")
    def append_tool(sheet_id: str, row: str) -> str:
        """Append one row to a sheet. ``row`` is a JSON array, e.g. ``["A","B",42]``.
        Formulas like "=A1+B1" are evaluated (USER_ENTERED mode)."""
        try:
            parsed = json.loads(row)
            if not isinstance(parsed, list):
                parsed = [parsed]
        except (json.JSONDecodeError, TypeError):
            parsed = [row]
        return json.dumps(_append(sheet_id, parsed))

    @tool("write_google_sheet_range")
    def write_tool(sheet_id: str, range_a1: str, values: str) -> str:
        """Overwrite a range with a JSON list-of-lists.

        Example: ``range_a1="Sheet1!A2:C2"``, ``values="[[\"hello\",1,\"=B2*2\"]]"``.
        """
        try:
            parsed = json.loads(values)
            if not isinstance(parsed, list):
                return json.dumps({"error": "values must be a JSON list-of-lists"})
        except (json.JSONDecodeError, TypeError) as exc:
            return json.dumps({"error": f"invalid JSON: {exc}"})
        return json.dumps(_write(sheet_id, range_a1, parsed))

    return [create_tool, read_tool, append_tool, write_tool]
