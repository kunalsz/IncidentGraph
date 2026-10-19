"""Minimal .env loader (no dependency). KEY=VALUE lines, '#' comments.

Existing environment variables always win (the file only fills gaps), so
tests and CI that set vars explicitly are never overridden. Loaded once per
process; defaults only to the known repository root (never arbitrary CWD).
"""

from __future__ import annotations

import os
from pathlib import Path

_LOADED = False


def default_env_path() -> Path:
    """`<repo root>/.env`. This module lives at backend/app/envfile.py, so the
    repository root is two levels up, not three (a three-level path pointed one
    directory above the checkout, which silently meant `.env` was never read)."""
    return Path(__file__).resolve().parents[2] / ".env"


def load_env_file(path: str | None = None) -> None:
    """Load .env into os.environ (never overriding set variables).

    A default-location load happens once per process; an explicit `path` is
    always re-read (tests use this). CLI scripts, `make` targets and the test
    suite all rely on the default resolving to the repository root.
    """
    global _LOADED
    if _LOADED and path is None:
        return
    if path is None:
        _LOADED = True
    candidates = ([Path(path)] if path else [default_env_path()])
    for candidate in candidates:
        if not candidate.is_file():
            continue
        for line in candidate.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
        return
