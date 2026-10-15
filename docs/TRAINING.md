# Train the adapter: notebook or local overnight

[Open the notebook in Colab](https://colab.research.google.com/github/nishsm/signalbench/blob/main/notebooks/finetune_signalbench.ipynb).
Select a GPU runtime, then execute the cells in order. All training/evaluation logic stays in
`src/signalbench`; the notebook orchestrates the same CLI used locally. It has no saved outputs,
embedded credentials or claimed training results.

## Which machine

| Profile | Base | Intended hardware | Interpretation |
| --- | --- | --- | --- |
| `hosted_3b` | Qwen2.5-3B-Instruct | Hosted CUDA GPU such as a Colab T4/L4 | Primary experiment |
| `rtx4060` | Qwen2.5-3B-Instruct | RTX 4060 laptop, sufficient free VRAM | Same primary experiment |
| `gtx1650` | Qwen2.5-0.5B-Instruct | GTX 1650-class laptop, if the smoke test fits | Separate smaller-model experiment |

The GTX 1650 profile assumes you meant the GTX 1650, commonly a 4 GB card; the notebook measures
actual VRAM. It does not attempt to squeeze the primary 3B run onto that card. It uses a shorter
sequence length, smaller adapter rank and FP32 compute with NF4 weights. FP32 avoids assuming
bf16 or fast mixed-precision support on older hardware, but may be slow. No fit, completion-time
or performance guarantee is made. Overnight changes available time, not available memory.
Do not call the fallback's scores “3B results.” Preserve its config and base-model identity.

The profiles use the repository's conservative TRL/PEFT path. Unsloth could be a separately
validated memory optimization later; this notebook does not mix another training backend into
the benchmark. Gradient checkpointing, accumulation and small batches are already enabled.

## Colab

Use a fresh Python kernel and select Runtime → Change runtime type → GPU. The notebook installs
pinned dependencies, mounts Drive for persistent checkpoints/results, resolves precision and
checks the GPU before loading weights. Public model downloads do not require a token. Optional
HF credentials use hidden input and environment variables, never notebook cells with literal keys.
Optional Trackio monitoring stores local metrics; training curves and JSON logs are saved regardless.

The data stage reuses the checked-in immutable dataset revisions, rebuilds deduplication/splits,
and checks the resulting split manifest against the committed benchmark. The smoke test performs
backward/optimizer, validation and checkpoint writes in separate folders. Full training starts
only after the smoke cell succeeds. A completed smoke test is not a benchmark result.

Colab GPU access and session length vary, and free GPUs are not guaranteed. Sessions can interrupt
an overnight run. Leave Drive persistence enabled, then rerun the same profile/run root to resume
the last completed checkpoint. Read [Colab's resource limits](https://research.google.com/colaboratory/faq.html).
Base weights cache on runtime disk; after a reset they may need downloading again.

After training, run base-versus-adapter evaluation, inspect learning curves and download the ZIP.
The ZIP contains the adapter, tokenizer, license notices, config, environment and measured results.
Raw data, secrets, base weights and optimizer checkpoints are excluded. Checkpoints remain in the
persistent run folder for resumption. Neither the notebook nor local commands publish anything.

## Local Linux / WSL2

Use a CUDA-capable driver, sufficient disk space and Python 3.11–3.13. For Windows, use WSL2 with
GPU support rather than expecting this Linux-pinned bitsandbytes setup to work natively.
The Mac can prepare data but cannot run this CUDA/NF4 training path.

```bash
git clone https://github.com/nishsm/signalbench.git
cd signalbench
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[ml,train,tracking]'

# Choose rtx4060 or gtx1650 explicitly. Outputs are isolated by profile.
python -c 'from signalbench.models.training import prepare_run; print(prepare_run("rtx4060", "runs"))'
RUN_CONFIG="$PWD/runs/rtx4060/experiment.yaml"
python -m signalbench.cli --config "$RUN_CONFIG" data
python -c 'from signalbench.core import config; from signalbench.models.training import verify_saved_splits; verify_saved_splits(config("runs/rtx4060/experiment.yaml"))'
python -m signalbench.cli --config "$RUN_CONFIG" finetune --smoke

# Start the real run inside a persistent terminal session.
tmux new -s signalbench
source .venv/bin/activate
RUN_CONFIG="$PWD/runs/rtx4060/experiment.yaml"
python -u -m signalbench.cli --config "$RUN_CONFIG" finetune 2>&1 | tee runs/rtx4060/train.log
# Detach with Ctrl-B, D. Reattach with: tmux attach -t signalbench
# After a successful run:
python -m signalbench.cli --config "$RUN_CONFIG" bench
```

For GTX 1650, replace `rtx4060` with `gtx1650` in preparation and paths. Plug in the laptop, prevent
sleep and keep it cooled. Re-running `finetune` resumes only a compatible checkpoint. Use a new
run root after changing hyperparameters; incompatible experiments must never share checkpoints.
If a smoke/full run reports OOM or nonfinite losses, investigate it rather than reporting output.
The recorded runtime is elapsed training time for that invocation, including resume; it is not
an inferred full-run wall-clock duration or an overnight completion promise.

## Hugging Face or Tinker

[Hugging Face Jobs](https://huggingface.co/docs/hub/jobs-overview) can run the same Python project on
managed GPUs. Jobs are batch jobs rather than a notebook server. Review current eligibility,
hardware pricing and billing, set an explicit timeout, and persist checkpoints during training
to the Hub or durable storage: a final ZIP on an ephemeral container alone is insufficient.
The notebook's Drive persistence is for Colab, not an HF Jobs persistence implementation.
GPU charges are separate from this project's API inference budget. No paid job is submitted here.

[Tinker](https://tinker-docs.thinkingmachines.ai/tinker/models/) supports managed LoRA training,
but the checked catalog does not list Qwen2.5-3B-Instruct. A supported substitute would be a new
base-model/backend experiment, with its own scoring/export integration; Tinker LoRA is not the
same implementation as this bitsandbytes QLoRA run. Keep the existing experiment for the first
complete result, then consider that comparison if useful.

Source checks: official provider documentation linked above; GPU memory preflight and smoke
results must come from the actual runtime. The notebook has not been GPU-executed on this Mac.

## Tinker credits and a changed base model

Changing the base model for a Tinker run is allowed for this project; keep it named separately
from the existing local/hosted QLoRA experiment and compare its own untouched base as well.
Use the provider's current model catalog at execution time. No Tinker integration or paid run
is implied by the Colab notebook; it currently executes the repository's TRL/PEFT trainer.

Thinking Machines' [published grant announcement](https://thinkingmachines.ai/news/tinker-research-and-teaching-grants/)
(announced October 29, 2025; checked October 14, 2026) lists research grants starting at $5,000
and teaching grants of $250 per student for academic classes, with applications by instructors.
These are application-based programs, not automatic student entitlements. The page includes
both application links. This check did not establish a current official automatic $150 signup
credit offer. Account-specific credits, eligibility and expiry must be checked in Tinker's console.
See [a short application draft](TINKER_CREDITS.md); no application has been submitted.
