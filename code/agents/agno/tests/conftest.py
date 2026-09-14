"""Shared pytest fixtures for the qa_agent test-suite.

The QA agent talks to a local LLM server for its interactive/agent flows, but
the functions we unit-test are pure and deterministic.  These fixtures give us:

* ``isolated_records`` -> a ``QARecordStore`` backed by a throwaway SQLite file,
  so tests never touch the project's real ``tmp/qa_agent.db``.
* ``tmp_source`` -> a temp directory with a couple of sample files for the
  static-analysis tests.
* ``workspace`` -> a temp directory used by the test-runner tests.

Run with::

    python -m pytest tests/ -q
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import qa_agent
from qa_agent import QARecordStore


@pytest.fixture
def isolated_records(tmp_path: Path) -> QARecordStore:
    """A QARecordStore pointed at a fresh temp database."""
    store = QARecordStore(tmp_path / "qa_agent.db")
    store._init(store._connect())
    return store


@pytest.fixture
def sample_files(tmp_path: Path) -> Path:
    """Create two sample source files for the analyzer."""
    root = tmp_path / "src"
    root.mkdir()

    bad = root / "bad.py"
    bad.write_text(
        "def add(a, b):\n"
        "    password = 'hunter2'\n"
        "    return eval(a + b)\n"
        "\n"
        "def run():\n"
        "    try:\n"
        "        do_work()\n"
        "    except:\n"
        "        pass\n"
        "\n"
        "    conn = sqlite3.connect('db')\n"
        "    conn.execute(f\"SELECT * FROM t WHERE id={x}\")\n"
        "    # TODO: implement caching\n"
        "    return None == value\n",
    )

    good = root / "good.py"
    good.write_text(
        "def greet(name: str) -> str:\n"
        "    return f'Hello, {name}'\n"
        "\n"
        "def main() -> int:\n"
        "    print(greet('world'))\n"
        "    return 0\n",
    )
    return root


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """An empty temp directory that stands in for the workspace root."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace
