"""Operator CLI for Andrus AI.

Narrow operational + recovery + scripting surface. NOT a chat surface
(Signal / Discord / ``/cp/chat`` already cover that).

Three things this CLI buys:

1. **Substrate-level escape hatch** — when Signal is broken or the dashboard
   is sick, ``python -m app.cli status --endpoint tailnet`` still works.
2. **Consolidation** of scattered ``python -m app.X`` modules under one
   discoverable umbrella (``aai brainstorm``, ``aai drill run X`` …).
3. **Scriptability** — pipe-friendly recall, briefing fetch, ledger tail
   that compose with ``jq`` / ``grep`` / ``less``.

Invoke via ``python -m app.cli``. Operator-recommended shell alias:

.. code-block:: bash

    alias aai='python -m app.cli'

See ``docs/CLI.md`` for the full subcommand inventory.
"""

__all__ = ["main"]
