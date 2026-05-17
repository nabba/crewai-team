"""``python -m app.cli`` entry point."""
from __future__ import annotations

import sys

from app.cli.main import main

if __name__ == "__main__":
    sys.exit(main())
