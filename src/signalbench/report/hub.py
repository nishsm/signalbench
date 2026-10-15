"""Adapter-only publication, with provenance and required research-license notices."""

import json
import os
from pathlib import Path

import httpx
from huggingface_hub import HfApi

from signalbench.core import artifact
from signalbench.models.registry import revision


def publish(cfg: dict) -> None:
    """Publish a trained adapter only when measured metrics and license notices exist."""
    repo = cfg["hub"]["repo_id"]
    if not repo:
        raise ValueError("Set hub.repo_id to the intended Hugging Face namespace/repository")
    folder = Path(cfg["paths"]["adapter"])
    if not (folder / "adapter_model.safetensors").exists():
        raise RuntimeError("No trained adapter exists; run make finetune on CUDA first")
    results = Path(cfg["paths"]["results"])
    if not (results / "benchmark.csv").exists() or (results / "REVIEW_REQUIRED.json").exists():
        raise RuntimeError("Measured, reviewed benchmark required before publication")
    rev = revision(cfg, cfg["training"]["base"])
    response = httpx.get(
        f"https://huggingface.co/{cfg['training']['base']}/raw/{rev}/LICENSE", follow_redirects=True
    )
    response.raise_for_status()
    (folder / "LICENSE").write_text(response.text)
    (folder / "NOTICE").write_text(
        "Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, "
        "Copyright (c) Alibaba Cloud. All Rights Reserved.\nModified by Nishanth Madthila: "
        "QLoRA sentiment adapter trained on FinancialPhraseBank. Built with Qwen.\n"
    )
    card = f"""---
base_model: {cfg["training"]["base"]}
library_name: peft
license: other
license_name: qwen-research
license_link: LICENSE
tags: [finance, sentiment, qlora]
---
# Financial sentiment adapter — Built with Qwen

Noncommercial research/evaluation only. Adapter weights modify the frozen base through LoRA.
Base revision: `{rev}`. No base weights are included.

Training data: FinancialPhraseBank sentences_allagree, Malo et al. (2014),
CC BY-NC-SA 3.0. Fixed deduplicated training split; validation early stopping.
Twitter validation is used only as an out-of-domain evaluation.

Intended use: reproducible financial sentiment research. Not investment advice.
No evidence of reliable tradable alpha is claimed. Pretraining contamination, FinBERT's
PhraseBank exposure, survivorship and incomplete free price histories affect interpretation.

## Measured benchmark

{(results / "benchmark.md").read_text()}

## Training configuration

```json
{json.dumps(cfg["training"], indent=2)}
```
"""
    artifact(folder / "README.md", cfg, card)
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo_id=repo, exist_ok=True, private=False)
    api.upload_folder(
        repo_id=repo,
        folder_path=folder,
        allow_patterns=[
            "adapter_*",
            "tokenizer*",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "README.md*",
            "LICENSE",
            "NOTICE",
        ],
    )
