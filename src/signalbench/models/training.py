"""Notebook/CLI training preparation without duplicating the training loop."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

from signalbench.core import config, digest


def supports_native_bf16() -> bool:
    """Exclude emulated bf16 on pre-Ampere CUDA GPUs such as Colab T4."""
    import torch

    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability(0)[0] >= 8
        and torch.cuda.is_bf16_supported()
    )


def precision_name(requested: str, bf16_supported: bool) -> str:
    """Resolve precision explicitly; old GPUs can request conservative FP32 compute."""
    if requested == "auto":
        return "bf16" if bf16_supported else "fp16"
    if requested not in {"bf16", "fp16", "fp32"}:
        raise ValueError(f"Unknown precision: {requested}")
    if requested == "bf16" and not bf16_supported:
        raise ValueError("This GPU does not support bf16; choose auto, fp16 or fp32")
    return requested


def smoke_config(cfg: dict) -> dict:
    """Isolate a two-update memory/backward/checkpoint test from research results."""
    smoke = copy.deepcopy(cfg)
    for key in ["adapter", "results", "checkpoints"]:
        original = smoke["paths"].get(key, "models/checkpoints")
        smoke["paths"][key] = str(Path(original).parent / (Path(original).name + "-smoke"))
    smoke["training"].update(
        max_steps=2, accumulation=2, eval_steps=1, save_steps=1, smoke_test=True, logging_steps=1
    )
    return smoke


def prepare_run(profile: str, root: str | Path) -> Path:
    """Write an isolated run config, carrying forward the committed dataset revisions."""
    allowed = {"hosted_3b", "rtx4060", "gtx1650"}
    if profile not in allowed:
        raise ValueError(f"Choose a profile from {sorted(allowed)}")
    cfg = config(f"configs/{profile}.yaml")
    root = Path(root).expanduser().resolve() / profile
    root.mkdir(parents=True, exist_ok=True)
    cfg["paths"].update(
        db=str(root / "data/signalbench.duckdb"),
        adapter=str(root / "adapter"),
        checkpoints=str(root / "checkpoints"),
        results=str(root / "results"),
    )
    # Large downloaded base weights stay on the runtime disk; outputs/checkpoints persist.
    cfg["paths"]["cache"] = str(Path("data/cache").resolve())
    cfg["benchmark"]["models"] = ["qwen_base", "qlora"]
    output = root / "results"
    output.mkdir(exist_ok=True)
    manifest = output / "data_manifest.json"
    if not manifest.exists():
        shutil.copyfile("results/data_manifest.json", manifest)
    # Keep the exact source model revision once resolved, including interrupted sessions.
    frozen = Path("results/cpu/model_manifest.json")
    target = output / "model_manifest.json"
    if frozen.exists() and not target.exists():
        shutil.copyfile(frozen, target)
    import yaml

    from signalbench.core import artifact

    path = root / "experiment.yaml"
    artifact(path, cfg, yaml.safe_dump(cfg, sort_keys=False))
    return path


def validate_training_frame(frame) -> None:
    """Reject empty partitions, unknown labels and overlapping normalized texts."""
    from signalbench import LABELS
    from signalbench.data.datasets import normalize

    for partition in ["train", "val"]:
        part = frame[frame.split == partition]
        if part.empty or set(part.label) != set(LABELS):
            raise ValueError(f"Training requires all three labels in {partition}")
    if not set(frame.label) <= set(LABELS):
        raise ValueError("Unknown sentiment label")
    if frame.groupby(frame.text.map(normalize)).split.nunique().max() > 1:
        raise ValueError("Text leakage across partitions")


def verify_saved_splits(cfg: dict) -> None:
    """Verify hosted acquisition matches the checked-in split IDs before spending GPU time."""
    import pandas as pd

    saved = pd.read_csv("results/splits/ids.csv")
    generated = pd.read_csv(Path(cfg["paths"]["results"]) / "splits/ids.csv")
    fields = ["source", "id", "split", "text_hash"]
    if digest(saved[fields].sort_values("id").to_dict("records")) != digest(
        generated[fields].sort_values("id").to_dict("records")
    ):
        raise ValueError("Run splits differ from committed benchmark; investigate before training")


def hardware(cfg: dict) -> dict:
    """Inspect the selected GPU and enforce a profile-specific minimum before loading weights."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Select a CUDA GPU runtime. Mac CPU/MPS cannot run this NF4 trainer.")
    properties = torch.cuda.get_device_properties(0)
    info = {
        "name": properties.name,
        "vram_gb": properties.total_memory / 1024**3,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "bf16_supported": supports_native_bf16(),
    }
    minimum = cfg["training"].get("minimum_vram_gb", 0)
    if info["vram_gb"] < minimum:
        raise RuntimeError(
            f"Profile requires >= {minimum} GiB; detected {info['vram_gb']:.2f}. "
            "Use gtx1650 for the separate 0.5B fallback, or a hosted GPU."
        )
    info["precision"] = precision_name(
        cfg["training"].get("precision", "auto"), info["bf16_supported"]
    )
    print(json.dumps(info, indent=2))
    return info


def bundle_run(cfg: dict, destination: str | Path) -> Path:
    """Export the adapter and measured results; exclude raw data, secrets and base weights."""
    import zipfile

    from huggingface_hub import hf_hub_download

    from signalbench.core import artifact

    adapter = Path(cfg["paths"]["adapter"])
    results = Path(cfg["paths"]["results"])
    if not (adapter / "adapter_model.safetensors").exists():
        raise ValueError("No completed adapter to export")
    experiment = json.loads((adapter / "experiment.json").read_text())
    if experiment["training"].get("smoke_test"):
        raise ValueError("Smoke-test adapters are not research exports")
    if not (adapter / "LICENSE").exists():
        license_path = hf_hub_download(
            cfg["training"]["base"], "LICENSE", revision=experiment["base_revision"]
        )
        shutil.copyfile(license_path, adapter / "LICENSE")
    notice = (
        f"Built with Qwen. Base: {cfg['training']['base']} "
        f"at {experiment['base_revision']}.\n"
        "Base license is reproduced in LICENSE.\n"
        "Modified by Nishanth Madthila: FinancialPhraseBank sentiment QLoRA adapter.\n"
        "Training data: FinancialPhraseBank, Malo et al. (2014), CC BY-NC-SA 3.0.\n"
        "Intended for noncommercial research. No commercial permissions are granted.\n"
    )
    (adapter / "NOTICE").write_text(notice)
    benchmark = results / "benchmark.md"
    card = (
        f"# Sentiment adapter — Built with Qwen\n\nBase: `{cfg['training']['base']}`.\n\n"
        "Adapter only. Noncommercial research; not investment advice.\n"
        "FinancialPhraseBank (Malo et al., 2014) is CC BY-NC-SA 3.0. "
        "See LICENSE and NOTICE.\n\n"
        "Training uses the fixed deduplicated training split "
        "and validation-loss early stopping.\n"
        "Base-model pretraining contamination remains possible. No trading claim is made.\n\n"
        + (benchmark.read_text() if benchmark.exists() else "Evaluation has not run.\n")
    )
    artifact(adapter / "README.md", cfg, card)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    allowed = (
        "adapter_*",
        "tokenizer*",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "experiment.json*",
        "README.md*",
        "LICENSE",
        "NOTICE",
    )
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(adapter.iterdir()):
            if path.is_file() and any(path.match(pattern) for pattern in allowed):
                archive.write(path, "adapter/" + path.name)
        for path in sorted(results.rglob("*")):
            if path.is_file() and path.suffix in {".json", ".csv", ".md", ".png", ".txt"}:
                archive.write(path, "results/" + str(path.relative_to(results)))
        import yaml

        archive.writestr("experiment.yaml", yaml.safe_dump(cfg, sort_keys=False))
    return destination
