"""Offline notebook/profile checks; no pretrained weights or cloud services."""

import ast
import json
import zipfile
from pathlib import Path

import nbformat
import pandas as pd
import pytest

from signalbench.core import config
from signalbench.models.training import (
    bundle_run,
    precision_name,
    prepare_run,
    smoke_config,
    validate_training_frame,
    verify_saved_splits,
)


def test_notebook_is_valid_python_and_has_no_saved_outputs():
    nb = json.loads(Path("notebooks/finetune_signalbench.ipynb").read_text())
    assert nb["nbformat"] == 4
    nbformat.validate(nb)
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert cell["outputs"] == []
            assert cell["execution_count"] is None
    source = "\n".join("".join(c["source"]) for c in nb["cells"])
    assert source.index('stage("finetune", "--smoke")') < source.index('stage("finetune")')


@pytest.mark.parametrize(
    "requested,supported,expected",
    [
        ("auto", True, "bf16"),
        ("auto", False, "fp16"),
        ("fp32", False, "fp32"),
        ("fp16", True, "fp16"),
    ],
)
def test_precision_policy(requested, supported, expected):
    assert precision_name(requested, supported) == expected


def test_unsupported_bf16_rejected():
    with pytest.raises(ValueError, match="does not support"):
        precision_name("bf16", False)


def test_profiles_keep_small_model_separate(tmp_path):
    primary = config("configs/hosted_3b.yaml")
    fallback = config("configs/gtx1650.yaml")
    assert primary["training"]["base"] == "Qwen/Qwen2.5-3B-Instruct"
    assert fallback["training"]["base"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert fallback["training"]["precision"] == "fp32"
    path = prepare_run("gtx1650", tmp_path)
    cfg = config(str(path))
    assert Path(cfg["paths"]["checkpoints"]).parent == path.parent
    assert json.loads((Path(cfg["paths"]["results"]) / "data_manifest.json").read_text())
    smoke = smoke_config(cfg)
    assert cfg["training"]["max_steps"] == -1
    assert smoke["training"]["max_steps"] == 2
    for key in ["checkpoints", "adapter", "results"]:
        assert smoke["paths"][key] != cfg["paths"][key]
    assert smoke["paths"]["db"] == cfg["paths"]["db"]


def test_training_frame_requires_complete_nonleaking_partitions():
    frame = pd.DataFrame(
        [
            {"split": split, "label": label, "text": f"{split} {label}"}
            for split in ["train", "val"]
            for label in ["negative", "neutral", "positive"]
        ]
    )
    validate_training_frame(frame)
    frame.loc[3, "text"] = frame.loc[0, "text"]
    with pytest.raises(ValueError, match="leakage"):
        validate_training_frame(frame)


def test_split_check_rejects_changed_ids(tmp_path):
    cfg = config("configs/default.yaml")
    cfg["paths"]["results"] = str(tmp_path)
    (tmp_path / "splits").mkdir()
    original = pd.read_csv("results/splits/ids.csv")
    original.to_csv(tmp_path / "splits/ids.csv", index=False)
    verify_saved_splits(cfg)
    original.loc[0, "id"] = "changed"
    original.to_csv(tmp_path / "splits/ids.csv", index=False)
    with pytest.raises(ValueError, match="differ"):
        verify_saved_splits(cfg)


def test_export_excludes_raw_data_secrets_and_base_weights(tmp_path):
    cfg = config("configs/gtx1650.yaml")
    adapter, results = tmp_path / "adapter", tmp_path / "results"
    adapter.mkdir()
    results.mkdir()
    cfg["paths"].update(adapter=str(adapter), results=str(results))
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter fixture")
    (adapter / "LICENSE").write_text("base license fixture")
    (adapter / "experiment.json").write_text(
        json.dumps({"base_revision": "frozen", "training": {}})
    )
    (adapter / ".env").write_text("fixture only")
    (adapter / "model.safetensors").write_bytes(b"base fixture")
    (results / "benchmark.md").write_text("Fixture metrics only")
    destination = bundle_run(cfg, tmp_path / "export.zip")
    with zipfile.ZipFile(destination) as archive:
        names = archive.namelist()
    assert "adapter/adapter_model.safetensors" in names
    assert "adapter/README.md" in names
    assert "adapter/LICENSE" in names
    assert "adapter/.env" not in names
    assert "adapter/model.safetensors" not in names
    assert "experiment.yaml" in names


@pytest.mark.parametrize("major,expected", [(7, False), (8, True)])
def test_t4_emulated_bf16_does_not_enable_amp_bf16(monkeypatch, major, expected):
    import sys
    from types import SimpleNamespace

    from signalbench.models.training import supports_native_bf16

    # The framework may report bf16 allocation/emulation even on older hardware.
    cuda = SimpleNamespace(
        is_available=lambda: True,
        get_device_capability=lambda _: (major, 5),
        is_bf16_supported=lambda: True,
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    assert supports_native_bf16() is expected
