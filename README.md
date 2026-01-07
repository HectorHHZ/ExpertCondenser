# Finetuning-MOE

## Directory Overview
- `qwen-patch/patch.py`: Overrides Qwen2 sparse MoE blocks to add aux-free forward passes, bias updates, optional forced experts, and a Sinkhorn routing variant.
- `deepseek-patch/patch.py`: Monkey-patches DeepSeek V2 MoE modules to remove aux losses, inject learnable bias terms, and enable forced expert activation across routing modes.
- `olmoe-patch/patch.py`: Extends the Olmoe sparse block with aux-free routing, bias tracking hooks, and a manual bias update helper for post-optimizer steps.
- `utils/moe_utils.py`: Helpers for persisting and restoring MoE bias tensors, including optional downloads from the Hugging Face Hub and distributed-safe saves.

## Tests
- `tests/test_qwen_patch.py`: Confirms the Qwen aux-free block picks the lowest-bias experts and records metadata when forced experts are enabled.
- `tests/test_deepseek_patch.py`: Checks that the DeepSeek gate patch selects and logs forced experts consistent with the aux-free routing logic.
- `tests/test_olmoe_patch.py`: Exercises the Olmoe aux-free block’s forced expert selection and record keeping.

### Running
1. Ensure `pytest` is installed (e.g. `pip install pytest`).
2. From the repository root, run `pytest tests` to execute the suite.
