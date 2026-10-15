"""Configuration, provenance and structured logs."""

from __future__ import annotations

import hashlib
import json
import logging
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def digest(value: Any) -> str:
    """Hash a canonical JSON-serializable value."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def setup(seed: int) -> None:
    """Seed CPU generators and enable JSON event logging."""
    random.seed(seed)
    np.random.seed(seed)

    class JSONFormatter(logging.Formatter):
        def format(self, record):
            return json.dumps(
                {"level": record.levelname, "logger": record.name, "event": record.getMessage()}
            )

    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)


def config(path: str) -> dict:
    """Read the complete experiment configuration."""
    return yaml.safe_load(Path(path).read_text())


def provenance(cfg: dict) -> dict:
    """Capture config, source state and UTC timestamp for every artifact."""

    def git(*args: str) -> str:
        result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
        return result.stdout.strip() or "uncommitted"

    return {
        "config": cfg,
        "config_hash": digest(cfg),
        "source_hash": digest(
            {str(p): digest(p.read_text()) for p in sorted(Path("src/signalbench").rglob("*.py"))}
        ),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def artifact(path: str | Path, cfg: dict, content: str) -> None:
    """Write an artifact and adjacent machine-readable provenance."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    path.with_suffix(path.suffix + ".meta.json").write_text(json.dumps(provenance(cfg), indent=2))
