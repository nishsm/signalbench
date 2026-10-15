"""Lazy model factory with frozen upstream revision manifests."""

import json
from pathlib import Path

from huggingface_hub import HfApi

from signalbench.core import artifact


def revision(cfg: dict, repo: str) -> str:
    """Resolve a branch once, then reuse its immutable commit."""
    path = Path(cfg["paths"]["results"]) / "model_manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {}
    if repo not in manifest:
        manifest[repo] = HfApi().model_info(repo).sha
        artifact(path, cfg, json.dumps(manifest, indent=2))
    return manifest[repo]


def model(name: str, cfg: dict, train):
    """Instantiate one model at a time to limit RAM/VRAM use."""
    if name in {"majority", "tfidf"}:
        from signalbench.models.local import Classic

        return Classic(train, cfg["seed"], majority=name == "majority")
    if name == "finbert":
        from signalbench.models.local import FinBERT

        return FinBERT(revision(cfg, "ProsusAI/finbert"))
    if name.startswith("api_"):
        from signalbench.models.api_llm import APIModel
        from signalbench.models.rag_fewshot import Examples

        mode = name.split("_", 1)[1]
        rev = revision(cfg, "sentence-transformers/all-MiniLM-L6-v2") if mode == "rag" else None
        return APIModel(
            cfg, Examples(train, mode, cfg["benchmark"]["shots_per_label"], cfg["seed"], rev)
        )
    if name in {"qwen_base", "qlora"}:
        from signalbench.models.qlora import Qwen

        return Qwen(cfg, revision(cfg, cfg["training"]["base"]), adapter=name == "qlora")
    raise ValueError(f"Unknown model {name}")
