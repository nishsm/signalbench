# Training notebook

[Open finetune_signalbench in Colab](https://colab.research.google.com/github/nishsm/signalbench/blob/main/notebooks/finetune_signalbench.ipynb)

Run on a hosted CUDA GPU or a local Linux/WSL2 NVIDIA laptop. It includes fixed-split checks,
GPU/precision preflight, isolated smoke training, compatible checkpoint resume, actual runtime
and VRAM logs, base/adapter evaluation and an adapter/results export. No outputs or keys are saved.

Choose `hosted_3b` / `rtx4060` for the primary model, or `gtx1650` for the separate 0.5B fallback.
See [the training guide](../docs/TRAINING.md). Final numbers still come from source modules/CLI;
there is no notebook-only research logic. No paid job or public model upload is triggered.
