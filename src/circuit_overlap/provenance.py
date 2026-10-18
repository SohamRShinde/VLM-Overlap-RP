"""Result provenance. Plan section 7: every result file records model,
checkpoint, n, seed, prompt, note layout and commit hash."""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = ["git_commit", "result_envelope", "write_result"]


def git_commit(repo_dir: str | Path | None = None) -> str:
    """Short commit hash, with ``-dirty`` when the tree has uncommitted changes.

    Returns ``"unknown"`` outside a git checkout (e.g. a Kaggle notebook that
    pasted the code in rather than cloning) -- a result must say so plainly
    rather than carry a fake hash.
    """
    cwd = str(repo_dir) if repo_dir else None
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd, capture_output=True, text=True, timeout=10, check=True,
        )
        commit = out.stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd, capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        return f"{commit}-dirty" if dirty else commit
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return "unknown"


def result_envelope(
    experiment: str,
    payload: dict,
    *,
    model: str,
    checkpoint: str,
    seed: int | None,
    prompt: str,
    n: int | None = None,
    repo_dir: str | Path | None = None,
    **extra: Any,
) -> dict:
    """Wrap a result payload with the metadata RESULTS.md requires."""
    return {
        "experiment": experiment,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": git_commit(repo_dir),
        "model": model,
        "checkpoint": checkpoint,
        "n": n,
        "seed": seed,
        "prompt": prompt,
        "python": platform.python_version(),
        **extra,
        "result": payload,
    }


def write_result(path: str | Path, envelope: dict) -> Path:
    """Write an envelope as indented JSON, creating parent directories."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(envelope, indent=2, sort_keys=False) + "\n")
    return p
