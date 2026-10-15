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

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen QLoRA requires the Linux CUDA machine; this host has no CUDA")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
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

    manifest = {
        "base_revision": revision,
        "training": cfg["training"],
        "seed": cfg["seed"],
        "data_ids": sorted(frame.loc[frame.split.isin(["train", "val"]), "id"].tolist()),
    }
    manifest["hash"] = digest(manifest)
    checkpoint_dir = Path("models/checkpoints")
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

    args = SFTConfig(
        output_dir="models/checkpoints",
        max_length=tr["max_length"],
        completion_only_loss=True,
        per_device_train_batch_size=tr["batch_size"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=tr["accumulation"],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        num_train_epochs=tr["epochs"],
        learning_rate=tr["learning_rate"],
        eval_strategy="steps",
        eval_steps=tr["eval_steps"],
        save_steps=tr["save_steps"],
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=10,
        report_to="none",
        seed=cfg["seed"],
        packing=False,
        optim="paged_adamw_8bit",
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
    trainer = SFTTrainer(
        model=model,
        args=args,
        processing_class=tokenizer,
        train_dataset=dataset("train"),
        eval_dataset=dataset("val"),
        peft_config=lora,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=tr["patience"])],
    )
    checkpoint = Path("models/checkpoints")
    from transformers.trainer_utils import get_last_checkpoint

    resume = get_last_checkpoint(str(checkpoint)) if checkpoint.exists() else None
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(cfg["paths"]["adapter"])
    tokenizer.save_pretrained(cfg["paths"]["adapter"])
    artifact(Path(cfg["paths"]["adapter"]) / "experiment.json", cfg, json.dumps(manifest, indent=2))
    import json

    from signalbench.core import artifact

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
