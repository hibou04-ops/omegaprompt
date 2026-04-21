"""Test helpers that avoid pytest tmpdir plugin issues in sandboxed runs."""

from __future__ import annotations

import shutil
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4


@contextmanager
def workspace_tmpdir() -> Path:
    root = Path.cwd() / ".sandbox-test-tmp"
    root.mkdir(exist_ok=True)
    path = root / f"case-{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
