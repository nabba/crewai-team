# Auditor proposal — `research:ConnectionError` (attempt 1)

> Auto-mirrored from `workspace/audit_journal.json` by the
> ``app.healing.auditor_bridge`` daemon. The auditor's
> ``run_error_resolution`` cron produced this fix proposal but the
> proposals system surface (`/cp/proposals`) was unattended; this
> file gives the change-request gate (`/cp/changes`) a concrete
> artefact to approve.

- **Pattern:** `research:ConnectionError`
- **Attempt:** 1
- **Proposed at:** 2026-05-03T17:30:39.080579+00:00
- **Files referenced:** —

## Proposed fix (auditor's own description)

```
Pattern research:ConnectionError attempt #1: The error is a network timeout. Increase the timeout setting in the LLM configuration or set the OPENAI_TIMEOUT environment variable. Code change: In the LLM instantiation, add `timeout=60` (or a higher value) to the constructor.
```

## Operator action

The auditor's description is prose, not a runnable diff. Approving
this CR lands this markdown as a record. Apply the actual code change
yourself based on the description above, then the next pass of
``auditor.run_error_resolution`` will mark the pattern resolved if no
new errors of the same shape appear within 24 hours.

If the proposal turns out to be wrong, **reject** the CR. The
auditor's progressive-refinement loop will try a different angle on
attempt #2.
