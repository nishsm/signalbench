"""Conservative CUDA QLoRA and deterministic full-label likelihood scoring."""

from pathlib import Path

import numpy as np

from signalbench import LABELS
from signalbench.models.common import output, prompt


def bounded_prompt(tokenizer, text: str, maximum: int) -> str:
    """Truncate only user content, preserving instruction and assistant prefix."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    while True:
        content = tokenizer.decode(tokens, skip_special_tokens=True)
        rendered = tokenizer.apply_chat_template(
            prompt(content), tokenize=False, add_generation_prompt=True
        )
        length = len(tokenizer.encode(rendered, add_special_tokens=False))
        if length <= maximum - 8:
            return rendered
        if not tokens:
            raise ValueError("Sequence budget cannot fit instruction template")
        tokens = tokens[: max(0, len(tokens) - (length - maximum + 8))]


def load(cfg: dict, revision: str, adapter: bool = False):
    """Load NF4 on CUDA only; fail clearly rather than attempting CPU training."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from signalbench.models.training import hardware

    info = hardware(cfg)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
        info["precision"]
    ]
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg["training"]["base"], revision=revision)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["training"]["base"],
        revision=revision,
        quantization_config=quant,
        device_map={"": 0},
        torch_dtype=dtype,
        attn_implementation="eager",
    )
    if adapter:
        import json

        from peft import PeftModel

        saved = json.loads((Path(cfg["paths"]["adapter"]) / "experiment.json").read_text())
        if saved["base_revision"] != revision:
            raise ValueError("Adapter base revision differs from the requested frozen base")
        model = PeftModel.from_pretrained(model, cfg["paths"]["adapter"])
    return model, tokenizer


class Qwen:
    """Conditional label probabilities normalized over all three complete label strings."""

    def __init__(self, cfg: dict, revision: str, adapter: bool = False):
        self.model, self.tokenizer = load(cfg, revision, adapter)
        self.model.eval()
        self.max_length = cfg["training"]["max_length"]

    def predict(self, texts: list[str]) -> list[dict]:
        """Sum token log-probabilities; never assume a label is a single token."""
        import torch

        probabilities = []
        for text in texts:
            rendered = bounded_prompt(self.tokenizer, text, self.max_length)
            prefix = self.tokenizer.encode(rendered, add_special_tokens=False)
            likelihoods = []
            for label in LABELS:
                target = self.tokenizer.encode(label, add_special_tokens=False)
                ids = torch.tensor([prefix + target], device=self.model.device)
                with torch.inference_mode():
                    logits = self.model(ids).logits.float().log_softmax(-1)
                value = sum(
                    logits[0, len(prefix) - 1 + i, token].item() for i, token in enumerate(target)
                )
                likelihoods.append(value)
            z = np.array(likelihoods)
            p = np.exp(z - z.max())
            probabilities.append(p / p.sum())
        return output(probabilities)


def train(cfg: dict, frame, revision: str) -> None:
    """Train completion-only labels, early stop on validation loss, save adapter only."""
    import json

    import torch
    from datasets import Dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import EarlyStoppingCallback, set_seed
    from trl import SFTConfig, SFTTrainer

    from signalbench.core import artifact, digest
    from signalbench.models.training import validate_training_frame

    validate_training_frame(frame)
    if cfg["training"].get("smoke_test"):
        frame = (
            frame[frame.split.isin(["train", "val"])]
            .groupby(["split", "label"], group_keys=False)
            .head(8)
            .copy()
        )
    manifest = {
        "base_revision": revision,
        "training": cfg["training"],
        "seed": cfg["seed"],
        "data_ids": sorted(frame.loc[frame.split.isin(["train", "val"]), "id"].tolist()),
    }
    manifest["hash"] = digest(manifest)
    checkpoint_dir = Path(cfg["paths"].get("checkpoints", "models/checkpoints"))
    manifest_path = checkpoint_dir / "experiment.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text())["hash"] != manifest["hash"]:
        raise ValueError("Checkpoint experiment differs; choose a new checkpoint directory")
    set_seed(cfg["seed"])
    model, tokenizer = load(cfg, revision)
    artifact(manifest_path, cfg, json.dumps(manifest, indent=2))
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False
    tr = cfg["training"]

    def dataset(partition: str):
        rows = []
        for row in frame[frame.split == partition].itertuples():
            prefix = bounded_prompt(tokenizer, row.text, tr["max_length"])
            rows.append({"prompt": prefix, "completion": row.label + tokenizer.eos_token})
        return Dataset.from_list(rows)

    from signalbench.models.training import precision_name, supports_native_bf16

    compute_precision = precision_name(tr.get("precision", "auto"), supports_native_bf16())
    args = SFTConfig(
        output_dir=str(checkpoint_dir),
        max_length=tr["max_length"],
        completion_only_loss=True,
        per_device_train_batch_size=tr["batch_size"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=tr["accumulation"],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=compute_precision == "bf16",
        fp16=compute_precision == "fp16",
        max_steps=tr.get("max_steps", -1),
        num_train_epochs=tr["epochs"],
        learning_rate=tr["learning_rate"],
        eval_strategy="steps",
        eval_steps=tr["eval_steps"],
        save_steps=tr["save_steps"],
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=tr.get("logging_steps", 10),
        report_to="none",
        seed=cfg["seed"],
        packing=False,
        optim=tr.get("optimizer", "paged_adamw_8bit"),
    )
    lora = LoraConfig(
        r=tr["rank"],
        lora_alpha=tr["alpha"],
        lora_dropout=tr["dropout"],
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type="CAUSAL_LM",
    )
    callbacks = [EarlyStoppingCallback(early_stopping_patience=tr["patience"])]
    if tr.get("tracking") == "trackio":
        import trackio
        from transformers import TrainerCallback

        trackio.init(
            project="signalbench", name=Path(cfg["paths"]["results"]).parent.name, config=tr
        )

        class TrackioCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                trackio.log(logs or {}, step=state.global_step)

        callbacks.append(TrackioCallback())
    trainer = SFTTrainer(
        model=model,
        args=args,
        processing_class=tokenizer,
        train_dataset=dataset("train"),
        eval_dataset=dataset("val"),
        peft_config=lora,
        callbacks=callbacks,
    )
    checkpoint = Path(cfg["paths"].get("checkpoints", "models/checkpoints"))
    from transformers.trainer_utils import get_last_checkpoint

    resume = get_last_checkpoint(str(checkpoint)) if checkpoint.exists() else None
    import time
    from datetime import datetime, timezone

    from signalbench.core import provenance

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    trainer.train(resume_from_checkpoint=resume)
    history = trainer.state.log_history
    for entry in history:
        for key in ["loss", "eval_loss", "grad_norm"]:
            if key in entry and not np.isfinite(entry[key]):
                raise RuntimeError(f"Nonfinite {key}; investigate before exporting an adapter")
    measured = {
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_vram_gb": torch.cuda.max_memory_reserved() / 1024**3,
        "gpu": torch.cuda.get_device_name(0),
        "smoke_test": tr.get("smoke_test", False),
        "run_completed_at": datetime.now(timezone.utc).isoformat(),
        "provenance": provenance(cfg),
    }
    artifact(
        Path(cfg["paths"]["results"]) / "training/runtime.json", cfg, json.dumps(measured, indent=2)
    )
    if tr.get("tracking") == "trackio":
        trackio.finish()
    trainer.save_model(cfg["paths"]["adapter"])
    tokenizer.save_pretrained(cfg["paths"]["adapter"])
    artifact(Path(cfg["paths"]["adapter"]) / "experiment.json", cfg, json.dumps(manifest, indent=2))
    artifact(
        Path(cfg["paths"]["results"]) / "training/log.json",
        cfg,
        json.dumps(trainer.state.log_history, indent=2),
    )

    import matplotlib.pyplot as plt

    from signalbench.eval.benchmark import savefig

    history = trainer.state.log_history
    fig, ax = plt.subplots()
    for key in ["loss", "eval_loss"]:
        records = [entry for entry in history if key in entry]
        ax.plot([entry["step"] for entry in records], [entry[key] for entry in records], label=key)
    ax.set(xlabel="Training step", ylabel="Loss")
    ax.legend()
    savefig(fig, Path(cfg["paths"]["results"]) / "training/curves.png", cfg)
