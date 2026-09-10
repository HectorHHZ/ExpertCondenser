# Reproducible training environment for ExpertCondenser.
#
# Build:
#   docker build -t expertcondenser .
#
# Run the unit tests:
#   docker run --rm expertcondenser pytest CondenserExpert/tests/
#
# Run training (mount caches and pass GPUs through):
#   docker run --rm --gpus all \
#     -v $HOME/.cache/huggingface:/root/.cache/huggingface \
#     expertcondenser \
#     accelerate launch --num_processes 8 CondenserExpert/sft_unified.py \
#       --model_name_or_path deepseek-ai/DeepSeek-V2-Lite \
#       --dataset_name HuggingFaceH4/Bespoke-Stratos-17k \
#       --learning_rate 2.0e-5 --num_train_epochs 1 --packing \
#       --max_seq_length 1024 --per_device_train_batch_size 1 \
#       --gradient_accumulation_steps 8 --gradient_checkpointing --bf16 \
#       --output_dir output/deepseek-v2-lite-sft

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /workspace/ExpertCondenser

# Install dependencies first so code changes don't invalidate this layer.
COPY pyproject.toml LICENSE README.md ./
RUN pip install --no-cache-dir \
    "transformers>=4.45,<5" datasets "trl>=0.18,<0.19" accelerate huggingface_hub pytest \
    deepspeed wandb

COPY CondenserExpert ./CondenserExpert
RUN pip install --no-cache-dir -e .

CMD ["python", "-c", "import CondenserExpert.utils as u; print('ExpertCondenser image OK:', u.__name__)"]
