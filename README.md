## Project Structure

```
CondenserExpert/
├── sft_unified.py          # Unified SFT training entrypoint
├── configs.py              # Extended SFTConfig and GRPOConfig
├── utils/
│   ├── __init__.py         # Package exports (patches, bias utilities)
│   ├── moe_utils.py        # MoE bias state save/load helpers
│   ├── model_utils.py      # Tokenizer and model loading utilities
│   ├── callbacks.py        # Training callbacks (hub push, benchmarks)
│   ├── evaluation.py       # LightEval benchmark integration
│   ├── hub.py              # Hugging Face Hub upload utilities
│   ├── import_utils.py     # Optional dependency checks
│   ├── wandb_logging.py    # Weights & Biases setup
│   ├── deepseek-patch/     # Aux-free routing patch for DeepSeek V2
│   ├── olmoe-patch/        # Aux-free routing patch for OLMoE
│   └── qwen-patch/         # Aux-free routing patch for Qwen2-MoE
└── tests/                  # Unit tests for each patch
```


## Installation

```bash
pip install torch transformers datasets trl huggingface_hub accelerate
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

## Tests

```bash
pip install pytest
pytest CondenserExpert/tests/
```
