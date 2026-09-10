# Preserving Long-Tailed Expert Information in Mixture-of-Experts Tuning

**Official implementation of ExpertCondenser — auxiliary-loss-free supervised fine-tuning (SFT) for sparse Mixture-of-Experts (MoE) large language models.**

[![arXiv](https://img.shields.io/badge/arXiv-2604.23036-b31b1b.svg)](https://arxiv.org/abs/2604.23036)
[![Venue](https://img.shields.io/badge/COLM%202026-Published-4c1.svg)](https://arxiv.org/abs/2604.23036)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

> **Preserving Long-Tailed Expert Information in Mixture-of-Experts Tuning**
> Haoze He, Xingyuan Ding, Xuan Jiang, Xinkai Zou, Alex Cheng, Yibo Zhao, Juncheng Billy Li, Heather Miller
> **COLM 2026** · [arXiv:2604.23036](https://arxiv.org/abs/2604.23036)

## Abstract

Despite MoE models leading many benchmarks, supervised fine-tuning (SFT) for the
MoE architectures remains difficult because its router layers are fragile.
Methods such as DenseMixer and ESFT mitigate router collapse with dense mixing
or auxiliary load-balancing losses, but these introduce noisy gradients that
often degrade performance. In preliminary experiments, we systematically pruned
experts and observed that while certain super experts are activated far more
frequently, discarding less used experts still leads to notable performance
degradation. This suggests that even rarely activated experts encode non-trivial
knowledge useful for downstream tasks. Motivated by this, we propose an
auxiliary-loss-free MoE SFT framework that combines bias-driven sparsification
with always-active gated condenser experts. Rather than enforcing balanced
activation across all experts, our method encourages task-relevant experts to
remain active while pushing long-tailed experts toward inactivity. The condenser
experts provide a persistent, learnable pathway that alleviates gradient
starvation and facilitates consolidation of information that would otherwise
remain fragmented across sparsely activated experts. Analysis further suggest
that this design better preserves long-tailed expert information under sparse
routing. Experiments on large-scale MoE models demonstrate that our approach
outperforms state-of-the-art SFT baselines such as DenseMixer and ESFT,
achieving average gain of 2.5%+ on both mathematical reasoning and commonsenseQA
benchmarks.

## TL;DR

- **Problem.** **Supervised fine-tuning of sparse MoE LLMs** is unstable because
  the **router** is fragile. **DenseMixer** and **ESFT** mitigate **router
  collapse** with dense mixing or **auxiliary load-balancing loss**, but both
  inject noisy gradients that often make things worse.
- **Finding.** We systematically **pruned experts** and found that although a
  small number of **super experts** dominate activation, discarding the rarely
  activated **long-tailed experts** still causes notable degradation — those
  experts encode non-trivial knowledge that downstream tasks depend on.
- **Method.** **ExpertCondenser** is an **auxiliary-loss-free** MoE SFT
  framework: **bias-driven sparsification** plus **always-active gated condenser
  experts**. Instead of forcing balanced activation, it keeps task-relevant
  experts active, pushes long-tailed experts toward inactivity, and gives a
  persistent learnable pathway that relieves **gradient starvation** and
  consolidates knowledge that would otherwise stay fragmented across sparsely
  activated experts.
- **Result.** Beats DenseMixer and ESFT by **+2.5% on average** across
  mathematical reasoning and commonsenseQA.

Where this sits in the literature: most work on adapting **Mixture-of-Experts
language models** either fine-tunes a subset of experts (**ESFT**, expert-choice
routing), densifies the router during training (**DenseMixer**), or adds an
**auxiliary load-balancing loss** to keep expert utilization even. This work
argues the opposite of load balancing — **balanced activation is not the goal**,
preserving the information held by the long tail is — and achieves it without any
auxiliary loss. The same long-tail result is the premise of our follow-up work on
**MoE compression and expert pruning**, [Less is MoE](https://github.com/HectorHHZ/Less-is-MoE)
(EMNLP 2026 Oral).

**Keywords:** mixture-of-experts, MoE, MoE fine-tuning, supervised fine-tuning,
SFT, router collapse, load balancing, auxiliary-loss-free, expert routing,
long-tailed experts, super experts, expert pruning, sparse activation, gradient
starvation, DenseMixer, ESFT, large language models, DeepSeek-V2, OLMoE,
Qwen2-MoE.

## Supported models

Aux-free routing patches are provided for three MoE families:

| Model family | Patch |
| --- | --- |
| DeepSeek-V2 / DeepSeek-V2-Lite | `CondenserExpert/utils/deepseek_patch.py` |
| OLMoE | `CondenserExpert/utils/olmoe_patch.py` |
| Qwen2-MoE | `CondenserExpert/utils/qwen_patch.py` |

Each patch has a corresponding unit test under `CondenserExpert/tests/`.
Logic shared by the OLMoE and Qwen patches (bias buffer, forced-expert
selection, bias update rule) lives in `CondenserExpert/utils/aux_free.py`.

## Project Structure

```
CondenserExpert/
├── sft_unified.py          # Unified SFT training entrypoint
├── configs.py              # Extended SFTConfig and GRPOConfig
├── utils/
│   ├── __init__.py         # Package exports (patches, bias utilities)
│   ├── aux_free.py         # Shared aux-free mixin (bias, forced experts)
│   ├── moe_utils.py        # MoE bias state save/load helpers
│   ├── model_utils.py      # Tokenizer and model loading utilities
│   ├── callbacks.py        # Training callbacks (hub push, bias updates)
│   ├── evaluation.py       # LightEval benchmark integration
│   ├── hub.py              # Hugging Face Hub upload utilities
│   ├── import_utils.py     # Optional dependency checks
│   ├── wandb_logging.py    # Weights & Biases setup
│   ├── deepseek_patch.py   # Aux-free routing patch for DeepSeek V2
│   ├── olmoe_patch.py      # Aux-free routing patch for OLMoE
│   └── qwen_patch.py       # Aux-free routing patch for Qwen2-MoE
└── tests/                  # Unit tests for each patch
```


## Installation

```bash
pip install -e .        # installs torch, transformers, datasets, trl, accelerate, huggingface_hub
```

Or with Docker:

```bash
docker build -t expertcondenser .
docker run --rm expertcondenser pytest CondenserExpert/tests/
```

## Usage

```bash
accelerate launch \
  --num_processes 8 \
  --config_file recipes/accelerate_configs/zero3_offload.yaml \
  CondenserExpert/sft_unified.py \
  --model_name_or_path deepseek-ai/DeepSeek-V2-Lite \
  --dataset_name HuggingFaceH4/Bespoke-Stratos-17k \
  --learning_rate 2.0e-5 \
  --num_train_epochs 1 \
  --packing \
  --max_seq_length 1024 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --gradient_checkpointing \
  --bf16 \
  --output_dir output/deepseek-v2-lite-sft
```

The Accelerate configuration passed to `--config_file` is not vendored in this
repository; point it at your own ZeRO-3 offload config, or any Accelerate config
that matches your hardware.

## Tests

```bash
pip install pytest
pytest CondenserExpert/tests/
```

## Related work from the authors

- **Less is MoE: Trimming Experts in Domain-Specialist Language Models**
  (EMNLP 2026 Main, Oral) — Fisher-MoE, intra-expert structured pruning and
  compression for MoE LLMs, built on the long-tailed expert result below.
  [arXiv:2606.05538](https://arxiv.org/abs/2606.05538) ·
  [code](https://github.com/HectorHHZ/Less-is-MoE)
- **SMT: Fine-Tuning Large Language Models with Sparse Matrices** (ICLR 2025).
  [code](https://github.com/HectorHHZ/Sparse_Matrix_Tuning)

## Citation

Published at **COLM 2026**. Would appreciate your citation :). This paper establishes the result that *Less is
MoE* (EMNLP 2026 Main, Oral) is built on: although a small number of **super
experts** dominate activation, discarding the rarely activated **long-tailed
experts** still causes notable degradation, so knowledge in a sparse MoE is
*distributed across the long tail of experts*. That means **whole-expert pruning
cannot be lossless no matter how the experts are ranked** — the failure is
inherent to the granularity, not to the scoring criterion. Less is MoE acts on
that finding by moving compression *inside* the expert and pruning FFN
intermediate dimensions, the granularity at which task capability actually
concentrates.

```bibtex
@article{he2026expertcondenser,
  title   = {Preserving Long-Tailed Expert Information in Mixture-of-Experts Tuning},
  author  = {He, Haoze and Ding, Xingyuan and Jiang, Xuan and Zou, Xinkai and
             Cheng, Alex and Zhao, Yibo and Li, Juncheng Billy and Miller, Heather},
  journal = {arXiv preprint arXiv:2604.23036},
  year    = {2026}
}

@article{he2026lessismoe,
  title   = {Less is {MoE}: Trimming Experts in Domain-Specialist Language Models},
  author  = {He, Haoze and Zou, Xinkai and Jiang, Xuan and Ding, Xingyuan and
             Qu, Ao and Li, Juncheng Billy and Miller, Heather},
  journal = {arXiv preprint arXiv:2606.05538},
  year    = {2026}
}
```

## License

ExpertCondenser is released under Apache-2.0. See [LICENSE](LICENSE).
