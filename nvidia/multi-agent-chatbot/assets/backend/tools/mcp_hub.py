"""Entrypoint for the resilient MCP hub."""

from pathlib import Path
import sys

# Ensure the repository root (which contains the ``tools`` package) is on the
# import path when this file is executed directly as ``python tools/mcp_hub.py``.
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.mcp_hub.hub import run_hub


if __name__ == "__main__":
    run_hub()

