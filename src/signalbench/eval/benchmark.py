"""Comparable benchmark runs and validation-only API selection."""

import gc
import json
import platform
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from signalbench import LABELS
from signalbench.core import artifact
from signalbench.eval.metrics import bootstrap, calibration, mcnemar, metrics
from signalbench.models.registry import model


def run(cfg: dict, con) -> None:
    """Evaluate every configured approach against identical frozen partitions."""
    data = con.execute("SELECT * FROM labeled_text ORDER BY id").df()
    train = data[(data.source == "sentences_allagree") & (data.split == "train")]
    if train.empty:
        raise RuntimeError("Run make data first")
    out = Path(cfg["paths"]["results"])
    summaries, predictions = [], {}
    for name in cfg["benchmark"]["models"]:
        estimator = model(name, cfg, train)
        for partition in ["val", "test", "ood", "secondary_only"]:
            frame = data[data.split == partition]
            if frame.empty:
                continue
            texts = frame.text.tolist()
            if name.startswith("api_"):
                estimate = estimator.estimate(texts)
                print(
                    json.dumps({"model": name, "partition": partition, "estimated_usd": estimate})
                )
                if estimate > cfg["api"]["budget"]:
                    raise RuntimeError("Estimate exceeds $20; check in with Nish before spending")
            start = time.perf_counter()
            rows = estimator.predict(texts)
            elapsed = time.perf_counter() - start
            pred = [r["label"] for r in rows]
            m = metrics(frame.label, pred)
            summary = {
                "model": name,
                "partition": partition,
                "n": len(frame),
                "accuracy": m["accuracy"],
                "macro_f1": m["macro_f1"],
                **bootstrap(frame.label, pred, cfg["seed"], cfg["benchmark"]["bootstrap"]),
                "usd_per_1000": sum(r.get("cost", 0) for r in rows) / len(rows) * 1000,
                "seconds_per_document": sum(r.get("seconds", 0) for r in rows) / len(rows)
                if name.startswith("api_")
                else elapsed / len(rows),
                "documents_per_second": len(rows) / elapsed,
                "hardware": platform.platform(),
                "peak_vram_gb": None,
                "ece": None,
            }
            try:
                import torch

                if name in {"qwen_base", "qlora"}:
                    summary["hardware"] = torch.cuda.get_device_name()
                    summary["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 1024**3
            except ImportError:
                pass
            if "probabilities" in rows[0]:
                ece, reliability = calibration(frame.label, [r["probabilities"] for r in rows])
                summary["ece"] = ece
                fig, ax = plt.subplots()
                ax.plot([0, 1], [0, 1], "--")
                ax.plot(
                    [r["confidence"] for r in reliability],
                    [r["accuracy"] for r in reliability],
                    "o-",
                )
                ax.set(xlabel="Confidence", ylabel="Accuracy", title=f"{name}: {partition}")
                savefig(fig, out / f"figures/{name}_{partition}_reliability.png", cfg)
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_predictions(frame.label, pred, labels=LABELS, ax=ax)
            savefig(fig, out / f"figures/{name}_{partition}_confusion.png", cfg)
            payload = [
                {"id": identifier, "truth": truth, **row}
                for identifier, truth, row in zip(frame.id, frame.label, rows, strict=True)
            ]
            artifact(
                out / f"predictions/{name}_{partition}.json", cfg, json.dumps(payload, indent=2)
            )
            artifact(
                out / f"per_class/{name}_{partition}.json", cfg, json.dumps(m["classes"], indent=2)
            )
            predictions[name, partition] = payload
            summaries.append(summary)
            # Persist progress, but quarantine any suspicious result before headline reporting.
            if partition == "ood" and m["macro_f1"] > 0.98:
                artifact(out / "REVIEW_REQUIRED.json", cfg, json.dumps(summary, indent=2))
                raise RuntimeError("OOD F1 > 0.98: investigate contamination before reporting")
        del estimator
        gc.collect()
    table = pd.DataFrame(summaries)
    validation = table[(table.partition == "val") & table.model.str.startswith("api_")]
    best = (
        validation.sort_values(["macro_f1", "model"], ascending=[False, True]).iloc[0].model
        if len(validation)
        else None
    )
    artifact(
        out / "selection.json",
        cfg,
        json.dumps(
            {"best_api": best, "criterion": "validation macro-F1 only; lexical tie-break"}, indent=2
        ),
    )
    tests = {}
    for partition in ["test", "ood"]:
        for other in [best, "finbert"]:
            if ("qlora", partition) in predictions and (other, partition) in predictions:
                a, b = predictions["qlora", partition], predictions[other, partition]
                if [r["id"] for r in a] != [r["id"] for r in b]:
                    raise ValueError("Unpaired model results")
                tests[f"qlora_vs_{other}_{partition}"] = mcnemar(
                    [r["truth"] for r in a], [r["label"] for r in a], [r["label"] for r in b]
                )
    artifact(out / "significance.json", cfg, json.dumps(tests, indent=2))
    artifact(out / "benchmark.csv", cfg, table.to_csv(index=False))
    artifact(out / "benchmark.md", cfg, "# Measured benchmark\n\n" + table.to_markdown(index=False))
    print("Benchmark checkpoint: review with Nish before setting edgar.scaleup_approved=true")


def savefig(fig, path: Path, cfg: dict) -> None:
    """Save figure and provenance sidecar."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    artifact(str(path) + ".provenance.json", cfg, "{}")
