# PDF generation + Signal attachment delivery

Native, agent-callable PDF composition and Signal-backed delivery.
Turns "the agent has data and needs to give the user a real file" from
"write a 200-line code block in chat and tell them to run it" into
"call `pdf_compose`, then call `signal_send_attachment`, done."

Shipped 2026-05-03 after a 6-hour failed Signal thread where the coding
crew, asked for an Estonia deforestation report, wrote Python source
as the chat response, fabricated CSV data values 40–75× off from real
Hansen v1.12 numbers, and never produced a deliverable file.

---

## 1. Components

| Path | Role |
|---|---|
| `app/tools/pdf_compose.py` | Sandboxed Python execution with matplotlib (Agg), reportlab Platypus, pandas, numpy pre-loaded. Output paths clamped to `/app/workspace/output/`. |
| `app/tools/signal_attachment.py` | Sends 1–5 files (≤25 MB total) from `/app/workspace/output/` to the configured Signal owner via the host's `signal-cli`. Recipient is hard-pinned. |
| `app/agents/coder.py` | Wires both tools via `optional_tool_group("coder", "pdf")` + `(..., "signal_attachment")`. |
| `app/agents/writer.py` | Same wiring (writer also has `generate_pdf` from `document_generator` for prose-shaped output; `pdf_compose` is for chart-heavy data reports). |
| `app/souls/coder.md` | Tool descriptions + two production-grade rules: "NEVER fabricate data" and "USE THE TOOL THAT BUILDS IT". |
| `tests/test_pdf_and_signal_tools.py` | 21 unit tests across 6 classes (path-traversal, sandbox contents, factory degradation, attachment validation, container→host translation). |

---

## 2. Why this exists

Pre-2026-05-03, matplotlib + reportlab were installed in the gateway
image but no agent-visible tool advertised PDF capability. When the
coding crew was asked for "a forest-loss report PDF":

1. It wrote a 200-line Python script as the **response text**.
2. The user copy-pasted it, hit syntax errors, asked the crew to fix
   them, the crew rewrote the script again as response text, etc.
3. After 6 retries the crew started **fabricating CSV data values**
   (e.g. "Estonia 2014 forest loss = 73 kha") that were 40–75× off
   from real Hansen v1.12 numbers, because no tool was producing
   ground-truth data either.
4. The user ended up doing it all manually.

Two gaps were closed:

* **`pdf_compose`** — a real, agent-callable PDF builder. The script
  the agent supplies *runs here*, in-process; matplotlib + reportlab
  + pandas are pre-loaded so short snippets stay short.
* **`signal_send_attachment`** — picks the file up from
  `/app/workspace/output/`, translates the container path to the
  host path, and hands it to `signal-cli` for delivery.

Together they close the loop: gather data → compose PDF → deliver
to user's phone. No code blocks in chat, no manual copy-paste.

The fabrication failure mode is addressed orthogonally in the
`coder.md` soul:

> **Real data only — NEVER fabricate.** If asked to produce a report
> with numbers (forest loss, sales figures, weather data, etc.) and
> you cannot get those numbers from a real tool call, SAY SO. Do
> not invent values, do not pull "plausible" numbers from training
> data, do not hardcode arrays based on memory.

---

## 3. `pdf_compose` — what the agent passes

Single arg `script` (Python source as a string). The script runs in
a sandbox dict pre-populated with:

| Name | What it is | Notes |
|---|---|---|
| `plt` | `matplotlib.pyplot` | Agg backend; no X11 / display surface. |
| `PdfPages` | `matplotlib.backends.backend_pdf.PdfPages` | Multi-page chart PDFs. |
| `matplotlib` | top-level module | For backend introspection. |
| `np` | `numpy` (or `None`) | Optional. |
| `pd` | `pandas` (or `None`) | Optional. |
| `csv` / `json` | stdlib | For CSV/JSON companion files. |
| `reportlab` | dict of common Platypus + canvas symbols | `SimpleDocTemplate`, `Paragraph`, `Spacer`, `Table`, `TableStyle`, `Image`, `PageBreak`, `getSampleStyleSheet`, `colors`, `letter`, `A4`, `canvas`. `None` if reportlab missing. |
| `safe_output_path(name)` | callable | Returns the absolute, clamped path under `/app/workspace/output/`. **Always use this** — writes anywhere else are silently coerced into the workspace dir. |
| `OUTPUT_DIR` | `"/app/workspace/output"` | Constant. |
| `result` | `None` | Set this to the primary output path so the caller can thread it forward to `signal_send_attachment`. |

Example A — chart-only PDF via `PdfPages`:

```python
out = safe_output_path("estonia_loss.pdf")
with PdfPages(out) as pdf:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(years, loss_kha, color="tab:green")
    ax.set_title("Estonia annual forest loss 2012–2024")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
result = str(out)
```

Example B — structured layout via reportlab Platypus:

```python
out = safe_output_path("report.pdf")
styles = reportlab["getSampleStyleSheet"]()
doc = reportlab["SimpleDocTemplate"](str(out))
doc.build([
    reportlab["Paragraph"]("Estonia forest report", styles["Title"]),
    reportlab["Spacer"](1, 12),
    reportlab["Paragraph"](
        "Source: Hansen Global Forest Change v1.12, 2001–2024.",
        styles["BodyText"],
    ),
    reportlab["Table"]([["Year", "Loss (kha)"]] + list(rows)),
])
result = str(out)
```

---

## 4. The `safe_output_path` clamp (path-traversal guard)

Whatever string the agent passes goes through:

```python
p = Path(user_path).expanduser()
base = p.name or "output.pdf"
safe = re.sub(r"[^A-Za-z0-9._-]", "_", base)
return Path("/app/workspace/output") / safe
```

So:

| Agent passes | Lands at |
|---|---|
| `"estonia.pdf"` | `/app/workspace/output/estonia.pdf` |
| `"../../etc/passwd"` | `/app/workspace/output/passwd` |
| `"/var/tmp/sneaky.pdf"` | `/app/workspace/output/sneaky.pdf` |
| `"re port (final)/data.pdf"` | `/app/workspace/output/data.pdf` |
| `""` | `/app/workspace/output/output.pdf` |

The clamp is lossy by design: any path-component manipulation is
reduced to the basename, then unsafe chars are replaced with `_`.
Agents cannot scribble outside the workspace output dir, period.

---

## 5. `signal_send_attachment` — recipient pin + scope guard

Two-arg tool: `body` (string, ≤2000 chars), `attachments` (list of
absolute paths under `/app/workspace/output/`, ≤5 files, ≤25 MB
total).

* **Recipient is hard-pinned** to `settings.signal_owner_number`.
  There is no `to` parameter exposed to the agent. The tool cannot
  be coerced into messaging arbitrary numbers.
* **File scope is hard-locked** to `/app/workspace/output/`. Each
  path is `Path(p).resolve()`-d, then `.relative_to(_ALLOWED_DIR)`
  is required to succeed. Anything resolving outside is rejected
  with a clear "outside" error. Path-traversal (`/app/workspace/output/../../etc/passwd`)
  resolves outside and is rejected.
* **Caps:** 5 attachments, 25 MB total, 2000-char body. The Signal
  protocol + `signal-cli` reject larger payloads anyway; we surface
  the cap upfront so the agent doesn't waste compute writing 30 PDFs
  that will all silently lose.

### Container → host path translation

`signal-cli` runs **on the host**, not in the container, so it needs
the host-side absolute path. The translation is:

```
/app/workspace/...   →   <settings.workspace_host_path>/...
```

E.g. with `WORKSPACE_HOST_PATH=/Users/andrus/BotArmy/crewai-team/workspace`:

```
/app/workspace/output/estonia.pdf
  → /Users/andrus/BotArmy/crewai-team/workspace/output/estonia.pdf
```

Without `WORKSPACE_HOST_PATH` set, the tool is **not registered** —
the factory returns `[]`. Better not to register than to silently
fail at delivery time. Same for `SIGNAL_OWNER_NUMBER` not set.

### Pattern (agent-side)

```python
# 1. Compose
result = pdf_compose(script="""
    out = safe_output_path('estonia_forest.pdf')
    # ... build the PDF ...
    result = str(out)
""")
# tool returns: "PDF compose completed.\n  /app/workspace/output/estonia_forest.pdf  (11,256 bytes)"

# 2. Deliver
signal_send_attachment(
    body="Estonia forest report — 2001–2024 Hansen v1.12 data.",
    attachments=["/app/workspace/output/estonia_forest.pdf"],
)
# tool returns: "Signal message sent.\n  recipient: +37250***1234\n  attachments (1):\n    - estonia_forest.pdf  (11,256 bytes)"
```

---

## 6. Production bug fixed inline: pydantic + matplotlib

`pdf_compose` initially imported matplotlib lazily inside
`_build_sandbox()` — i.e. on every tool invocation. The first live
call bombed with:

```
TypeError: _suppress_pydantic_deprecation_warnings.<locals>.filtered_warn()
got an unexpected keyword argument 'skip_file_prefixes'
```

Pydantic monkey-patches `warnings.warn` with a filtered wrapper that
suppresses its own deprecation warnings. Matplotlib (Python 3.12+)
calls `warnings.warn(..., skip_file_prefixes=...)` during import —
a kwarg pydantic's wrapper doesn't accept.

Fix: import matplotlib + reportlab **once at module load time**,
inside a `_import_with_warn_shim()` helper that swaps in a no-op
`warnings.warn` shim around the heavy imports, then restores
whatever was there before. References get cached at module level;
`_build_sandbox()` is now a fast dict-construction with no further
imports.

```python
def _import_with_warn_shim():
    def _swallow(*_a, **_kw):  # eats every kwarg, incl. skip_file_prefixes
        return None
    _orig_warn = _warnings.warn
    _warnings.warn = _swallow
    try:
        import matplotlib as _mpl
        _mpl.use("Agg")
        import matplotlib.pyplot as _plt
        from matplotlib.backends.backend_pdf import PdfPages as _PdfPages
        # ... reportlab ...
    finally:
        _warnings.warn = _orig_warn

_MPL_PACK, _RL_PACK = _import_with_warn_shim()
```

After the fix, a smoke test from inside the container produces a
real 11 KB PDF (PDF document, version 1.4, 1 page) on the host
side at `/Users/andrus/BotArmy/crewai-team/workspace/output/`.

---

## 7. Tool description as LLM steering

Both tools' `description` fields embed:

* An **explicit anti-pattern** label — `pdf_compose`'s description
  contains the literal phrase "instead of writing Python source as
  the response text" so the LLM sees the contrast. The
  `test_description_steers_away_from_writing_source_as_text` unit
  test asserts this phrase + `safe_output_path` + `pdfpages` +
  `platypus` are all present. Lock-in against future
  refactors that lose the steering.
* **Worked GOOD examples** for both common backends (PdfPages for
  chart-only quick reports, Platypus for structured layout). The
  GEE tool taught us this: descriptions are the LLM's primary
  context for "how do I use this", and a one-liner doesn't beat a
  6-line worked example.

`signal_send_attachment` similarly hard-codes the recipient-pin
constraint ("Recipient is HARD-PINNED ... there is no `to`
parameter; do not try to specify one") and the file-scope
constraint ("Files MUST live under /app/workspace/output/").

---

## 8. Tests

`tests/test_pdf_and_signal_tools.py` — 21 tests across 6 classes:

| Class | Coverage |
|---|---|
| `TestSafeOutputPath` | basename preservation, path-traversal stripping, absolute-path collapse, unicode/special-char replacement, empty-string fallback. |
| `TestSandboxContents` | matplotlib pre-loaded with Agg backend, `safe_output_path` callable, `result` initial value. |
| `TestPdfFactory` | factory returns 1 tool, description contains anti-pattern + safe_output_path + PdfPages + platypus phrases. |
| `TestAttachmentValidation` | empty list, too many, outside-workspace, missing file, path-traversal — all rejected. |
| `TestPathTranslation` | container→host translation, trailing-slash handling, unmapped-path passthrough (loud failure). |
| `TestSignalFactory` | empty when owner missing, empty when workspace_host_path missing, returns 1 tool when both configured. |

Run inside the container (the workspace path doesn't exist on the
host):

```bash
docker exec crewai-team-gateway-1 python -m pytest tests/test_pdf_and_signal_tools.py -v
```

Expected: `21 passed`.

---

## 9. When the tool is NOT registered

`create_signal_attachment_tools()` returns `[]` when:

* `SIGNAL_OWNER_NUMBER` env var is empty.
* `WORKSPACE_HOST_PATH` env var is empty.

In either case, the agent simply doesn't see the tool — it falls
back to "I produced the PDF at `/app/workspace/output/X.pdf`".
This is intentional: silently failing to deliver is worse than
visibly not having the tool.

`create_pdf_tools()` is unconditional — it always returns
`[PdfComposeTool]`, since matplotlib + reportlab are baked into
the gateway image. If a future deployment strips them, the tool's
invocation will fail with a clear ImportError surfaced to the
agent in the tool's return string; the factory itself doesn't gate.

---

## 10. Operational guardrails recap

| Property | Mechanism |
|---|---|
| Output dir hard-clamped to `/app/workspace/output/`. | `_safe_output_path` regex-strips path components + unsafe chars. |
| No outbound network from the PDF builder. | Matplotlib forced to Agg backend; sandbox dict has no HTTP client. |
| Recipient hard-pinned. | `signal_send_attachment` has no `to` parameter. |
| Attachment scope hard-locked. | `_validate_attachments` does `Path.resolve()` + `.relative_to(_ALLOWED_DIR)`. |
| Per-call caps. | 5 attachments, 25 MB total, 2000-char body. |
| Container/host path mapping is explicit. | `_container_to_host` + `WORKSPACE_HOST_PATH` env var; tool refuses to register without it. |
| Heavy-import compatibility with pydantic. | `_import_with_warn_shim()` wraps matplotlib/reportlab imports at module load time. |

---

## 11. Future work

* **Inline attachment preview in the React Consciousness UI.** Right
  now the user sees the file land in their Signal thread; the React
  app shows the agent's tool-call result string but no preview.
  Wiring a thumbnail (PDF page 1 → PNG) into the consciousness
  attachment pane would make agent-side PDF authoring more visible.
* **Multi-page chart-+-table compositions.** Both the PdfPages and
  Platypus paths work in isolation. The current pattern for "chart
  page 1 + table page 2" is to call `pdf_compose` twice and merge,
  which is awkward. A `merge_pdfs` helper in the sandbox would close
  this.
* **Crew-level fabrication-detection guard.** Orthogonal to delivery,
  but the same incident motivated a parked task to detect when an
  agent writes data values into a CSV/JSON file without a citing
  tool call earlier in the same task. See the corresponding spawned
  task chip.
