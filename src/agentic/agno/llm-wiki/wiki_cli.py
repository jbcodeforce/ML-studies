#!/usr/bin/env python3
"""Launcher so you can run without setting PYTHONPATH (see README)."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from llm_wiki.cli import main

if __name__ == "__main__":
    main()
