"""Allow `python -m app.brainstorm` to invoke the CLI."""
from app.brainstorm.cli import main

raise SystemExit(main())
