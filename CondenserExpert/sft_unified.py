"""Unified supervised fine-tuning entrypoint for aux-free MoE models.

# One 1 node of 8 x H100s

accelerate launch \
  --num_processes 2 \
  --main_process_port $master_port \
  --config_file=recipes/accelerate_configs/zero3_offload.yaml \
  src/open_r1/sft_unified.py \
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
  --logging_steps 5 \
  --eval_strategy steps \
  --eval_steps 100 \
  --remove_unused_columns False \
  --output_dir data/deepseek-v2-lite-sft 
"""

from __future__ import annotations

import functools
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, Tuple

import datasets
import torch
import torch.distributed as dist
import transformers
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, set_seed
from transformers.trainer_utils import get_last_checkpoint

from CondenserExpert.configs import SFTConfig
from CondenserExpert.utils import (
    DEEPSEEK_FORCED_EXPERTS_RECORDS,
    OLMOE_FORCED_EXPERTS_RECORDS,
    QWEN_FORCED_EXPERTS_RECORDS,
    AuxFreeOlmoeSparseMoeBlock,
    AuxFreeQwen2MoeSparseMoeBlock,
    get_tokenizer,
    load_moe_bias_states,
    patch_deepseek_model,
    save_moe_bias_states,
)
from CondenserExpert.utils.callbacks import get_callbacks
from CondenserExpert.utils.wandb_logging import init_wandb_training
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.utils import DataCollatorForChatML

logger = logging.getLogger(__name__)


class DatasetProcessor:
    """Load datasets and mirror legacy preprocessing logic."""

    def __init__(self, script_args: ScriptArguments):
        self.script_args = script_args

    def load(self) -> Tuple[Dataset, Optional[Dataset]]:
        dataset = load_dataset(
            self.script_args.dataset_name,
            name=self.script_args.dataset_config,
        )

        if isinstance(dataset, Dataset):
            train_dataset = dataset
            eval_dataset = None
        else:
            train_split = self._resolve_split(
                dataset,
                self.script_args.dataset_train_split,
                default_candidates=("train", "training", "default"),
            )
            eval_split = self._resolve_split(
                dataset,
                self.script_args.dataset_test_split,
                default_candidates=("test", "eval", "validation", "dev", "valid"),
            )
            train_dataset = dataset[train_split] if train_split else None
            eval_dataset = dataset[eval_split] if eval_split else None

        if train_dataset is None:
            raise ValueError("Training dataset could not be resolved from the provided splits")

        train_dataset, eval_dataset = self._apply_dataset_specific_processing(
            self.script_args.dataset_name,
            train_dataset,
            eval_dataset,
        )

        self._validate_dataset(train_dataset)
        if eval_dataset is not None:
            self._validate_dataset(eval_dataset)

        return train_dataset, eval_dataset

    @staticmethod
    def _resolve_split(
        dataset: DatasetDict,
        requested: Optional[str],
        default_candidates: Tuple[str, ...],
    ) -> Optional[str]:
        available = list(dataset.keys())
        if requested and requested in available:
            return requested
        for candidate in default_candidates:
            if candidate in available:
                return candidate
        return available[0] if available else None

    def _apply_dataset_specific_processing(
        self,
        dataset_name: str,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset],
    ) -> Tuple[Dataset, Optional[Dataset]]:
        lowered_name = dataset_name.lower()

        if any(tag in lowered_name for tag in ["math10k", "math7k", "math14k"]):
            def convert_math(example):
                messages = [
                    {"role": "user", "content": example.get("instruction", example.get("question", ""))},
                    {
                        "role": "assistant",
                        "content": f"<think>{example.get('output', '')}</think>\\boxed{{{example.get('answer', '')}}}",
                    },
                ]
                return {"messages": messages}

            train_dataset = train_dataset.map(convert_math)
            if eval_dataset is not None:
                eval_dataset = eval_dataset.map(convert_math)

        elif "codeforces-cots" in lowered_name:
            def remove_prompt(example):
                example.pop("prompt", None)
                return example

            train_dataset = train_dataset.map(remove_prompt)
            if eval_dataset is not None:
                eval_dataset = eval_dataset.map(remove_prompt)

            if "messages" in train_dataset.column_names:
                train_dataset = train_dataset.map(lambda x: {"message_length": len(str(x["messages"]))})
                train_dataset = train_dataset.sort("message_length")
                train_dataset = train_dataset.select(range(min(38000, len(train_dataset)))).shuffle(seed=42)
                if eval_dataset is not None:
                    eval_dataset = eval_dataset.map(lambda x: {"message_length": len(str(x["messages"]))})

        elif "openr1-math-220k" in lowered_name:
            train_dataset = train_dataset.map(lambda x: {"message_length": len(str(x["messages"]))})
            train_dataset = train_dataset.sort("message_length")
            train_dataset = train_dataset.select(range(min(38000, len(train_dataset)))).shuffle(seed=42)

        elif "evol-codealpaca" in lowered_name:
            def convert_alpaca(example):
                prompt = example.get("instruction", "")
                if example.get("input"):
                    prompt = f"{prompt}\nInput: {example['input']}"
                return {"prompt": prompt, "completion": example.get("output", "")}

            train_dataset = train_dataset.map(convert_alpaca, remove_columns=train_dataset.column_names)
            if eval_dataset is not None:
                eval_dataset = eval_dataset.map(convert_alpaca, remove_columns=eval_dataset.column_names)

        elif "commonsense" in lowered_name:
            def convert_commonsense(example):
                return {
                    "messages": [
                        {"role": "user", "content": example.get("instruction", "")},
                        {"role": "assistant", "content": example.get("output", "")},
                    ]
                }

            train_dataset = train_dataset.map(convert_commonsense)
            if eval_dataset is not None:
                eval_dataset = eval_dataset.map(convert_commonsense)

        elif not self._has_messages(train_dataset):
            train_dataset = self._convert_generic(train_dataset)
            if eval_dataset is not None:
                eval_dataset = self._convert_generic(eval_dataset)

        return train_dataset, eval_dataset

    @staticmethod
    def _has_messages(dataset: Dataset) -> bool:
        if len(dataset) == 0:
            return False
        return "messages" in dataset.column_names and isinstance(dataset[0]["messages"], list)

    @staticmethod
    def _convert_generic(dataset: Dataset) -> Dataset:
        if len(dataset) == 0:
            return dataset
        sample = dataset[0]

        if "conversations" in sample:
            def convert_conversations(example):
                messages = []
                for turn in example["conversations"]:
                    role = "user" if turn.get("from") == "human" else "assistant"
                    messages.append({"role": role, "content": turn.get("value", "")})
                return {"messages": messages}

            return dataset.map(convert_conversations)

        if "instruction" in sample and "output" in sample:
            def convert_alpaca(example):
                user_text = example.get("instruction", "")
                if example.get("input"):
                    user_text = f"{user_text}\n\n{example['input']}"
                return {
                    "messages": [
                        {"role": "user", "content": user_text},
                        {"role": "assistant", "content": example.get("output", "")},
                    ]
                }

            return dataset.map(convert_alpaca)

        if "prompt" in sample and "response" in sample:
            return dataset.map(
                lambda example: {
                    "messages": [
                        {"role": "user", "content": example.get("prompt", "")},
                        {"role": "assistant", "content": example.get("response", "")},
                    ]
                }
            )

        logger.warning("Unknown dataset format encountered; skip conversion")
        return dataset

    @staticmethod
    def _validate_dataset(dataset: Dataset) -> None:
        if len(dataset) == 0:
            raise ValueError("Dataset is empty")
        for idx in range(min(5, len(dataset))):
            sample = dataset[idx]
            if "messages" not in sample and "prompt" not in sample:
                raise ValueError(
                    f"Sample {idx} missing required keys; available keys: {list(sample.keys())}"
                )


@dataclass
class AuxFreeModelConfig(ModelConfig):
    """
    Configuration class for [`ModelConfig`].

    Args:
        bias_update_speed (`float`, *optional*, defaults to `1e-4`):
            gamma to update the bias.
        remove_aux_loss (`bool`, *optional*, defaults to `True`):
            Whether to remove the aux loss.
        add_aux_free_loss (`bool`, *optional*, defaults to `True`):
            Whether to add the aux free loss.
    """
    bias_update_speed: float = field(
        default=1e-4,
        metadata={"help": "Gamma used when updating aux-free routing bias."},
    )
    remove_aux_loss: bool = field(
        default=True,
        metadata={"help": "Remove legacy auxiliary loss from the MoE gate."},
    )
    add_aux_free_loss: bool = field(
        default=True,
        metadata={"help": "Enable aux-free bias update logic."},
    )
    enable_forced_experts: bool = field(
        default=False,
        metadata={"help": "Force-activate experts with lowest bias values."},
    )
    num_forced_experts: int = field(
        default=2,
        metadata={"help": "Number of experts to force activate."},
    )
    bias_file_path: str = field(
        default="",
        metadata={"help": "Path or repo id containing moe_bias_states.json."},
    )


def detect_model_family(model_name: str) -> str:
    """
    detect model family from model name
    """
    lowered = model_name.lower()
    if "deepseek" in lowered:
        return "deepseek"
    if "qwen" in lowered:
        return "qwen"
    if "olmoe" in lowered:
        return "olmoe"
    return "generic"


FORCED_EXPERTS_RECORDS_MAP = {
    "deepseek": DEEPSEEK_FORCED_EXPERTS_RECORDS,
    "olmoe": OLMOE_FORCED_EXPERTS_RECORDS,
    "qwen": QWEN_FORCED_EXPERTS_RECORDS,
}


def save_forced_experts_records(
    output_dir: str,
    records: Dict[str, Dict[str, object]],
    hub_repo_id: Optional[str] = None,
) -> None:
    """Save forced experts records to JSON and optionally upload to hub"""
    if not records:
        return
    if dist.is_initialized() and dist.get_rank() != 0:
        return

    payload = {
        "metadata": {"total_layers": len(records), "timestamp": datetime.now().isoformat()},
        "forced_experts_records": records,
    }

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "forced_experts_records.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    logger.info("Saved forced expert records to %s", path)

    if hub_repo_id:
        try:
            from huggingface_hub import upload_file

            upload_file(path_or_fileobj=path, path_in_repo="forced_experts_records.json", repo_id=hub_repo_id)
            logger.info("Uploaded forced expert records to Hub repo %s", hub_repo_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to upload forced expert records: %s", exc)



class CustomSFTTrainer(SFTTrainer):
    """Custom trainer that preserves the messages column."""
    def __init__(self, *args, update_bias_after_step: bool = False, bias_module_types=tuple(), **kwargs):
        self.update_bias_after_step = update_bias_after_step
        self.bias_module_types = bias_module_types
        super().__init__(*args, **kwargs)

    def _prepare_dataset(self, dataset, *args, **kwargs):
        if "messages" in dataset.column_names:
            dataset = dataset.add_column("_messages", dataset["messages"])
            dataset = super()._prepare_dataset(dataset, *args, **kwargs)
            if "_messages" in dataset.column_names:
                dataset = dataset.rename_column("_messages", "messages")
            return dataset
        return super()._prepare_dataset(dataset, *args, **kwargs)

    def training_step(self, model, inputs, num_items_in_batch: Optional[int] = None):  
        if num_items_in_batch is not None:
            loss = super().training_step(model, inputs, num_items_in_batch)
        else:
            loss = super().training_step(model, inputs)

        if (
            self.update_bias_after_step
            and self.bias_module_types
            and (self.state.global_step + 1) % self.args.gradient_accumulation_steps == 0
        ):
            self._update_moe_biases(model)
        return loss

    def _update_moe_biases(self, model: torch.nn.Module) -> None:
        for module in model.modules():
            if isinstance(module, self.bias_module_types) and hasattr(module, "update_bias_after_step"):
                module.update_bias_after_step()


def build_model(
    model_family: str,
    model_args: AuxFreeModelConfig,
    training_args: SFTConfig,
) -> torch.nn.Module:
    """
    Construct and return a fully configured model for the requested family:
    - Sets dtype/use_cache and, when allowed, quantization `device_map`/`quantization_config`.
    - DeepSeek: load, patch MoE/Gate, and optionally load bias states.
    - OLMOE: replace official MoE blocks with aux-free variants (optionally Sinkhorn),
      set forced expert options, and optionally load bias.
    - Qwen2: monkey-patch Qwen2-MoE class BEFORE loading weights to aux-free (or Sinkhorn) variants,
      then optionally load bias for forced experts.
    """
    import logging
    logger = logging.getLogger(__name__)

    model_id = model_args.model_name_or_path

    # dtype / quantization
    dtype = model_args.torch_dtype if model_args.torch_dtype not in (None, "auto") else model_args.torch_dtype
    torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype

    quantization_config = None
    if model_id not in {"openai/gpt-oss-20b", "Qwen/Qwen3-30B-A3B"}:
        quantization_config = get_quantization_config(model_args)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
    )
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = get_kbit_device_map()

    training_args.model_init_kwargs = model_kwargs

    # --- DeepSeek ---
    if model_family == "deepseek":
        training_args.model_init_kwargs["trust_remote_code"] = True
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            **training_args.model_init_kwargs,
        )
        model.config.bias_update_speed = model_args.bias_update_speed
        model = patch_deepseek_model(
            model,
            model_args,
            num_forced_experts=model_args.num_forced_experts,
            bias_update_speed=model_args.bias_update_speed,
        )
        if model_args.enable_forced_experts:
            bias_source = model_args.bias_file_path or model_id
            load_moe_bias_states(model, bias_source)
        return model

    # --- OLMOE ---
    if model_family == "olmoe":
        import transformers.models.olmoe.modeling_olmoe as olmoe_module

        
        olmoe_module.OlmoeSparseMoeBlock = AuxFreeOlmoeSparseMoeBlock

        config = AutoConfig.from_pretrained(model_id)
        config.bias_update_speed = model_args.bias_update_speed
        config.enable_forced_experts = model_args.enable_forced_experts
        config.num_forced_experts = model_args.num_forced_experts
        config.use_cache = not training_args.gradient_checkpointing

        architecture = getattr(transformers, config.architectures[0])
        model = architecture.from_pretrained(model_id, config=config, **training_args.model_init_kwargs)
        model.config.bias_update_speed = model_args.bias_update_speed
        if model_args.enable_forced_experts:
            bias_source = model_args.bias_file_path or model_id
            load_moe_bias_states(model, bias_source)
        return model

    # --- Qwen (Qwen2/Qwen1.5 MoE) ---
    if "qwen1.5" in model_args.model_name_or_path.lower():
        import transformers.models.qwen2_moe.modeling_qwen2_moe as qwen2_moe_module

        
        qwen2_moe_module.Qwen2MoeSparseMoeBlock = AuxFreeQwen2MoeSparseMoeBlock
        logger.info("✅ Using standard aux-free routing for Qwen MoE blocks.")

        config = AutoConfig.from_pretrained(model_id)
        architecture = getattr(transformers, config.architectures[0])
        # config comparison
        model = architecture.from_pretrained(model_id, **training_args.model_init_kwargs)
        model.config.bias_update_speed = model_args.bias_update_speed

        if model_args.enable_forced_experts:
            bias_source = model_args.bias_file_path or model_id
            logger.info("🎯 Loading bias for forced expert selection (Qwen).")
            bias_loaded = load_moe_bias_states(model, bias_source)
            if bias_loaded:
                logger.info("✅ Bias loaded successfully for forced experts (Qwen).")
            else:
                logger.warning("⚠️ Failed to load bias; forced experts will use default initialization (Qwen).")

        return model

    # --- Generic fallback ---
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    architecture = getattr(transformers, config.architectures[0])
    model = architecture.from_pretrained(model_id, **training_args.model_init_kwargs)
    model.config.bias_update_speed = model_args.bias_update_speed
    
    return model


def main(script_args, training_args, model_args) -> None:
    set_seed(training_args.seed)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)

    os.makedirs(training_args.output_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(training_args.output_dir, "training.log"))
    file_handler.setLevel(log_level)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s", "%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(file_handler)

    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info("Model parameters: %s", model_args)
    logger.info("Script parameters: %s", script_args)
    logger.info("Training parameters: %s", training_args)

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint and training_args.resume_from_checkpoint is None:
            logger.info("Resuming from checkpoint: %s", last_checkpoint)

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    processor = DatasetProcessor(script_args)
    train_dataset, eval_dataset = processor.load()

    tokenizer = get_tokenizer(model_args, training_args)
    tokenizer.pad_token = tokenizer.eos_token

    original_torch_load = torch.load

    @functools.wraps(original_torch_load)
    def patched_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        kwargs.setdefault("map_location", "cpu")
        return original_torch_load(*args, **kwargs)

    torch.load = patched_torch_load

    model_family = detect_model_family(model_args.model_name_or_path)
    forced_experts_records = FORCED_EXPERTS_RECORDS_MAP.get(model_family, {})
    if model_family in {"qwen", "olmoe"}:
        tokenizer.padding_side = "left"
        data_collator = DataCollatorForChatML(tokenizer=tokenizer, max_length=training_args.max_length)
    else:
        data_collator = None

    model = build_model(model_family, model_args, training_args)
    
    if (not dist.is_initialized()) or dist.get_rank() == 0:
        try:
            print("=== Model.config (dict) ===")
            print(json.dumps(model.config.to_dict(), indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"Failed to dump model.config: {e}")
    ############################
    # Initialize the SFT Trainer
    ############################
    bias_module_types = tuple()
    if model_family == "olmoe":
        bias_module_types = AuxFreeOlmoeSparseMoeBlock

    # trainer_class = CustomSFTTrainer if data_collator is not None else SFTTrainer
    trainer_class = CustomSFTTrainer
    trainer = trainer_class(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
        update_bias_after_step=(model_family == "olmoe"),
        bias_module_types=bias_module_types,
    )
    
    hub_repo_id = getattr(training_args, "hub_model_id", None)
    if model_args.enable_forced_experts:
        save_forced_experts_records(training_args.output_dir, forced_experts_records, hub_repo_id)
        
    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = training_args.resume_from_checkpoint or last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    ##################################
    # Save model and create model card
    ##################################
    if model_args.add_aux_free_loss:
        save_moe_bias_states(trainer.model, training_args.output_dir)

    if model_args.enable_forced_experts:
        save_forced_experts_records(training_args.output_dir, forced_experts_records, hub_repo_id)

    trainer.save_model(training_args.output_dir)
    logger.info("Model saved to %s", training_args.output_dir)

    card_kwargs = {"dataset_name": script_args.dataset_name, "tags": ["open-r1"]}
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**card_kwargs)
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    if training_args.do_eval and eval_dataset is not None:
        logger.info("*** Evaluate ***")
        eval_metrics = trainer.evaluate()
        eval_metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
        
    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing artifacts to Hub")
        trainer.push_to_hub(**card_kwargs)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, AuxFreeModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
