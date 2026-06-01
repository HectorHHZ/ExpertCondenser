import os
import sys
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src", "open_r1"))

# Rebuttal experiment: PER_LAYER_R_GROUPS env var enables per-layer-r evaluation.
# Format: "r_early,r_mid,r_late" (e.g. "4,2,0"). When unset, original uniform behavior.
PER_LAYER_R_GROUPS = os.environ.get("PER_LAYER_R_GROUPS", "").strip()


def _apply_per_layer_forced_experts_eval(model, groups_str):
    """Resize forced_expert_indices buffers per layer-group, AFTER the
    uniform patch installs them. Triggers re-selection on first forward."""
    import re as _re
    parts = [int(x) for x in groups_str.split(",")]

    moe_modules = []
    for name, module in model.named_modules():
        if hasattr(module, "forced_expert_indices"):
            m = _re.search(r"layers\.(\d+)\.", name)
            if m:
                moe_modules.append((int(m.group(1)), name, module))
    if not moe_modules:
        print("[per-layer-r] no forced_expert_indices buffers found", flush=True)
        return
    moe_modules.sort(key=lambda t: t[0])
    num_layers = moe_modules[-1][0] + 1
    if len(parts) == 3:
        g = num_layers // 3
        sizes = [g, g, num_layers - 2 * g]
        layer_r = []
        for r, size in zip(parts, sizes):
            layer_r.extend([r] * size)
    elif len(parts) == num_layers:
        layer_r = list(parts)
    else:
        raise ValueError(f"PER_LAYER_R_GROUPS must be 3 or {num_layers} ints, got {len(parts)}")
    print(f"[per-layer-r eval] groups={parts}  layer_r={layer_r}", flush=True)

    for layer_idx, name, module in moe_modules:
        r = layer_r[layer_idx]
        if r == 0:
            module.enable_forced_experts = False
            print(f"[per-layer-r eval] {name} L{layer_idx}: DISABLED", flush=True)
            continue
        old_device = module.forced_expert_indices.device
        old_dtype = module.forced_expert_indices.dtype
        del module.forced_expert_indices
        module.register_buffer(
            "forced_expert_indices",
            torch.full((r,), -1, device=old_device, dtype=old_dtype),
            persistent=True,
        )
        module.num_forced_experts = r
        module.forced_experts_initialized = False
        print(f"[per-layer-r eval] {name} L{layer_idx}: r={r}", flush=True)

import argparse
import re
import json
import torch
import functools
import torch.nn.functional as F
import torch.distributed as dist
from datetime import datetime
import numpy as np
from typing import Optional, List, Tuple, Union
from transformers import (
    AutoModelForCausalLM,
    GenerationConfig,
)
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
)

from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import set_seed
# from utils.utils import print_rank_0, set_random_seed
import tqdm
# from utils.model_utils import load_hf_tokenizer, create_hf_model
from transformers import StoppingCriteria
from transformers import AutoConfig, AutoTokenizer
i_prompt = '''<s> Below is an instruction that describes a task. Write a response that appropriately completes the request. 

### Instruction:
{instruction}

### Response:
'''  

def load_moe_bias_states(model: torch.nn.Module, model_path_or_repo: str) -> bool:
    """Load bias states from moe_bias_states.json"""
    try:
        import json
        import os
        from huggingface_hub import hf_hub_download
        
        if "/" in model_path_or_repo and not os.path.exists(model_path_or_repo):
            try:
                print(f"🔄 Downloading moe_bias_states.json from: {model_path_or_repo}")
                bias_file_path = hf_hub_download(repo_id=model_path_or_repo, filename="moe_bias_states.json", cache_dir=None)
            except Exception as e:
                print(f"❌ Failed to download: {e}")
                return False
        else:
            bias_file_path = os.path.join(model_path_or_repo, "moe_bias_states.json")
        
        if not os.path.exists(bias_file_path):
            print(f"moe_bias_states.json not found at: {bias_file_path}")
            return False
            
        with open(bias_file_path, 'r') as f:
            bias_data = json.load(f)
        
        bias_states = bias_data.get("moe_bias_states", {})
        loaded_count = 0
        
        for module_name, module in model.named_modules():
            if hasattr(module, 'bias') and module_name in bias_states:
                try:
                    bias_info = bias_states[module_name]
                    bias_values = torch.tensor(bias_info["bias_values"], dtype=module.bias.dtype)
                    module.bias.data.copy_(bias_values.to(module.bias.device))
                    loaded_count += 1
                    print(f"✅ Loaded bias for {module_name}")
                except Exception as e:
                    print(f"⚠️ Failed to load bias for {module_name}: {e}")
        
        print(f"🎉 Successfully loaded bias for {loaded_count} MoE layers")
        return loaded_count > 0
        
    except Exception as e:
        print(f"❌ Error loading MoE bias states: {e}")
        return False

def load_forced_experts_records(model_path_or_repo: str) -> dict:
    """
    Universal function to load forced experts records from JSON files.
    
    Supports:
    - Multiple file names: forced_experts_records.json, forced_experts_mapping.json
    - Multiple formats: Qwen2 (direct list) and DeepSeek (nested dict) formats
    - Multiple data keys: forced_experts_records, forced_experts, experts
    - Both local files and HuggingFace Hub downloads
    
    Args:
        model_path_or_repo: Path to model directory or HuggingFace repo_id
        
    Returns:
        dict: Normalized forced experts records in format {layer_name: [expert_indices]}
    """
    try:
        from huggingface_hub import hf_hub_download
        
        # List of possible filenames to try
        possible_filenames = [
            "forced_experts_records.json",
            "forced_experts_mapping.json", 
            "forced_experts.json"
        ]
        
        file_path = None
        filename_used = None
        
        if "/" in model_path_or_repo and not os.path.exists(model_path_or_repo):
            # Try downloading from HuggingFace Hub
            print(f"🔄 Attempting to download forced experts files from HuggingFace Hub: {model_path_or_repo}")
            
            for filename in possible_filenames:
                try:
                    file_path = hf_hub_download(repo_id=model_path_or_repo, filename=filename, cache_dir=None)
                    filename_used = filename
                    print(f"✅ Downloaded {filename} to: {file_path}")
                    break
                except Exception as e:
                    print(f"   - {filename} not found: {e}")
                    continue
            
            if file_path is None:
                print(f"❌ No forced experts files found in {model_path_or_repo}")
                return {}
        else:
            # Try local files
            print(f"🔍 Looking for forced experts files in: {model_path_or_repo}")
            
            for filename in possible_filenames:
                test_path = os.path.join(model_path_or_repo, filename)
                if os.path.exists(test_path):
                    file_path = test_path
                    filename_used = filename
                    print(f"✅ Found {filename}")
                    break
                else:
                    print(f"   - {filename} not found")
            
            if file_path is None:
                print(f"❌ No forced experts files found in: {model_path_or_repo}")
                # List available JSON files for debugging
                if os.path.exists(model_path_or_repo):
                    print(f"📁 Available JSON files:")
                    for file in os.listdir(model_path_or_repo):
                        if file.endswith('.json'):
                            print(f"   - {file}")
                return {}
        
        # Load and parse the JSON file
        print(f"📖 Loading {filename_used}...")
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        print(f"🔍 JSON file structure - top level keys: {list(data.keys())}")
        
        # Try to find the data under different possible keys
        raw_records = None
        data_key_used = None
        
        for key in ["forced_experts_records", "forced_experts", "experts"]:
            if key in data:
                raw_records = data[key]
                data_key_used = key
                print(f"✅ Found data under key: '{key}'")
                break
        
        if raw_records is None:
            print(f"❌ No forced experts data found. Available keys: {list(data.keys())}")
            return {}
        
        # Normalize the data format
        normalized_records = normalize_forced_experts_format(raw_records, filename_used)
        
        print(f"📊 Successfully loaded {len(normalized_records)} layer records")
        
        # Print sample layer names for debugging
        if normalized_records:
            sample_layers = list(normalized_records.keys())[:3]
            sample_data = {k: normalized_records[k] for k in sample_layers}
            print(f"📝 Sample normalized data: {sample_data}")
            
        return normalized_records
        
    except Exception as e:
        print(f"❌ Error loading forced experts records: {e}")
        import traceback
        traceback.print_exc()
        return {}


def normalize_forced_experts_format(raw_records: dict, filename: str = "") -> dict:
    """
    Normalize different forced experts formats into a standard format.
    
    Input formats:
    1. Qwen2 format: {"model.layers.0.mlp": [38, 36]}
    2. DeepSeek format: {"model.layers.0.mlp.gate": {"forced_expert_indices": [1, 0]}}
    
    Output format: {"layer_name": [expert_indices]}
    
    Args:
        raw_records: Raw records from JSON file
        filename: Original filename for format detection hints
        
    Returns:
        dict: Normalized records in format {layer_name: [expert_indices]}
    """
    try:
        print(f"🔄 Normalizing forced experts format...")
        normalized = {}
        
        for layer_name, record_data in raw_records.items():
            try:
                expert_indices = None
                normalized_layer_name = layer_name
                
                if isinstance(record_data, list):
                    # Format 1: Direct list - Qwen2 style
                    # "model.layers.0.mlp": [38, 36]
                    expert_indices = record_data
                    print(f"   ✅ {layer_name}: Direct list format -> {expert_indices}")
                    
                elif isinstance(record_data, dict):
                    # Format 2: Dictionary with metadata - DeepSeek style  
                    # "model.layers.0.mlp.gate": {"forced_expert_indices": [1, 0], ...}
                    if "forced_expert_indices" in record_data:
                        expert_indices = record_data["forced_expert_indices"]
                        
                        # Normalize layer name: remove .gate suffix for consistency
                        if layer_name.endswith(".gate"):
                            normalized_layer_name = layer_name[:-5]  # Remove ".gate"
                        
                        print(f"   ✅ {layer_name} -> {normalized_layer_name}: Dict format -> {expert_indices}")
                    else:
                        print(f"   ⚠️ {layer_name}: Dict format but no 'forced_expert_indices' key found")
                        continue
                else:
                    print(f"   ❌ {layer_name}: Unsupported format: {type(record_data)}")
                    continue
                
                if expert_indices is not None:
                    normalized[normalized_layer_name] = expert_indices
                    
            except Exception as e:
                print(f"   ❌ Error processing {layer_name}: {e}")
                continue
        
        print(f"✅ Normalized {len(normalized)} records")
        return normalized
        
    except Exception as e:
        print(f"❌ Error in format normalization: {e}")
        return {}

def select_forced_experts(module, layer_name=None, forced_experts_records=None, module_type="general"):
    """
    Universal function to select forced experts using pre-recorded indices or bias-based selection.
    
    Args:
        module: The MoE module (can be DeepSeek MoEGate or Qwen2 MoE module)
        layer_name: Name of the layer for logging and record lookup
        forced_experts_records: Dictionary containing pre-recorded expert indices
        module_type: Type of module for better logging ("deepseek", "qwen2", or "general")
    
    Returns:
        torch.Tensor: Selected forced expert indices
    """
    if not module.forced_experts_initialized:
        print(f"🔍 Selecting forced experts for {layer_name} ({module_type})")
        
        if forced_experts_records:
            print(f"   - Total records available: {len(forced_experts_records)}")
            print(f"   - Layer '{layer_name}' in records: {layer_name in forced_experts_records}")
            if layer_name not in forced_experts_records:
                # Show available layer names for debugging
                available_layers = list(forced_experts_records.keys())[:3]  # Show first 3
                print(f"   - Available layer names (first 3): {available_layers}")
        else:
            print(f"   - No forced experts records available")
        
        # Priority 1: Use pre-recorded forced experts if available
        if forced_experts_records and layer_name in forced_experts_records:
            try:
                expert_indices = forced_experts_records[layer_name]
                print(f"   - Pre-recorded experts for {layer_name}: {expert_indices}")
                
                # Data is already normalized to list format by load_forced_experts_records
                if not isinstance(expert_indices, list):
                    raise ValueError(f"Expected list format but got {type(expert_indices)}: {expert_indices}")
                
                forced_indices = torch.tensor(
                    expert_indices, 
                    device=module.forced_expert_indices.device,
                    dtype=module.forced_expert_indices.dtype
                )
                module.forced_expert_indices.copy_(forced_indices)
                module.forced_experts_initialized = True
                print(f"✅ Used pre-recorded forced experts for {layer_name} ({module_type}): {forced_indices.tolist()}")
                return module.forced_expert_indices
            except Exception as e:
                print(f"⚠️ Failed to load pre-recorded experts for {layer_name}: {e}")
                import traceback
                traceback.print_exc()
        
        # Priority 2: Fall back to bias-based selection if bias is available
        print(f"   - Falling back to bias-based selection")
        print(f"   - Module has bias: {hasattr(module, 'bias')}")
        if hasattr(module, 'bias'):
            print(f"   - Bias is not all zeros: {not torch.all(module.bias == 0)}")
            print(f"   - Bias shape: {module.bias.shape}")
            print(f"   - Bias min/max: [{module.bias.min().item():.6f}, {module.bias.max().item():.6f}]")
            
        if hasattr(module, 'bias') and not torch.all(module.bias == 0):
            _, lowest_indices = torch.topk(module.bias, module.num_forced_experts, largest=False)
            module.forced_expert_indices.copy_(lowest_indices.to(module.forced_expert_indices.device))
            module.forced_experts_initialized = True
            print(f"✅ Selected forced experts from bias for {layer_name} ({module_type}): {lowest_indices.tolist()}")
        else:
            print(f"⚠️ No pre-recorded experts or valid bias found for {layer_name} ({module_type})")
    
    return module.forced_expert_indices

# Backward compatibility functions
def select_forced_experts_deepseek(gate_module, layer_name=None, forced_experts_records=None):
    """Backward compatibility wrapper for DeepSeek models"""
    return select_forced_experts(gate_module, layer_name, forced_experts_records, "deepseek")

def select_forced_experts_qwen2(moe_module, layer_name=None, forced_experts_records=None):
    """Backward compatibility wrapper for Qwen2 models"""
    return select_forced_experts(moe_module, layer_name, forced_experts_records, "qwen2")

@torch.no_grad()
def patched_moe_infer_with_forced_experts(self, x, topk_ids, topk_weight):
    """Modified moe_infer to handle forced experts with variable topk shapes"""
    
    # Use a simpler approach that matches the original moe_infer logic
    cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
    cnts.scatter_(1, topk_ids, 1)
    tokens_per_expert = cnts.sum(dim=0)
    
    # Create index mapping similar to original moe_infer
    idxs = topk_ids.view(-1).argsort()
    # Calculate token indices properly for forced experts
    token_idx_per_expert = idxs // topk_ids.shape[1]
    sorted_tokens = x[token_idx_per_expert]
    sorted_tokens_shape = sorted_tokens.shape
    
    # Handle distributed inference (if ep_size > 1)
    if self.ep_size > 1:
        tokens_per_ep_rank = tokens_per_expert.view(self.ep_size, -1).sum(dim=1)
        tokens_per_expert_group = tokens_per_expert.new_empty(tokens_per_expert.shape[0])
        dist.all_to_all_single(tokens_per_expert_group, tokens_per_expert)
        output_splits = (
            tokens_per_expert_group.view(self.ep_size, -1)
            .sum(1)
            .cpu()
            .numpy()
            .tolist()
        )
        gathered_tokens = sorted_tokens.new_empty(
            tokens_per_expert_group.sum(dim=0).cpu().item(), sorted_tokens.shape[1]
        )
        input_split_sizes = tokens_per_ep_rank.cpu().numpy().tolist()
        dist.all_to_all(
            list(gathered_tokens.split(output_splits)),
            list(sorted_tokens.split(input_split_sizes)),
        )
        tokens_per_expert_post_gather = tokens_per_expert_group.view(
            self.ep_size, self.experts_per_rank
        ).sum(dim=0)
        gatherd_idxs = np.zeros(shape=(gathered_tokens.shape[0],), dtype=np.int32)
        s = 0
        for i, k in enumerate(tokens_per_expert_group.cpu().numpy()):
            gatherd_idxs[s : s + k] = i % self.experts_per_rank
            s += k
        gatherd_idxs = gatherd_idxs.argsort()
        sorted_tokens = gathered_tokens[gatherd_idxs]
        tokens_per_expert = tokens_per_expert_post_gather
    
    tokens_per_expert = tokens_per_expert.cpu().numpy()

    # Process tokens through experts
    outputs = []
    start_idx = 0
    for i, num_tokens in enumerate(tokens_per_expert):
        end_idx = start_idx + num_tokens
        if num_tokens == 0:
            continue
        expert = self.experts[i + self.ep_rank * self.experts_per_rank]
        tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
        expert_out = expert(tokens_for_this_expert)
        outputs.append(expert_out)
        start_idx = end_idx

    outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0, sorted_tokens.shape[-1])
    
    # Handle distributed case cleanup
    if self.ep_size > 1:
        new_x = torch.empty_like(outs)
        new_x[gatherd_idxs] = outs
        gathered_tokens = new_x.new_empty(*sorted_tokens_shape)
        dist.all_to_all(
            list(gathered_tokens.split(input_split_sizes)),
            list(new_x.split(output_splits)),
        )
        outs = gathered_tokens

    # Restore original order using a safer method
    # Create a mapping that only assigns values to indices that have corresponding outputs
    new_x = torch.zeros(len(idxs), sorted_tokens.shape[-1], device=outs.device, dtype=outs.dtype)
    if len(outs) > 0:
        # Only assign outputs where we actually have them
        actual_indices = idxs[:len(outs)]
        new_x[actual_indices] = outs
    
    # Apply weights with the correct shape handling for forced experts
    final_out = (
        new_x.view(*topk_ids.shape, -1)
        .type(topk_weight.dtype)
        .mul_(topk_weight.unsqueeze(dim=-1))
        .sum(dim=1)
        .type(new_x.dtype)
    )
    return final_out

def patch_qwen2_model_for_evaluation(model, model_path_or_repo, enable_forced_experts=True,
                                     num_forced_experts=2, bias_update_speed=1e-4):
    """Patch Qwen2 model for evaluation with forced experts"""
    print("🔧 Applying Qwen2 MoE patches for evaluation...")
    
    # Load forced experts records if available
    forced_experts_records = {}
    if enable_forced_experts:
        forced_experts_records = load_forced_experts_records(model_path_or_repo)
    
    patched_modules = 0
    
    for name, module in model.named_modules():
        # Find Qwen2MoeSparseMoeBlock modules
        if module.__class__.__name__ == "Qwen2MoeSparseMoeBlock":
            print(f"🔧 Patching Qwen2MoeSparseMoeBlock in {name}")
            
            # Add bias if not present
            if not hasattr(module, 'bias'):
                module.bias_update_speed = bias_update_speed
                module.register_buffer("bias", torch.zeros(module.num_experts), persistent=True)
                print(f"✅ Added bias to {name}: {module.num_experts} experts")
            
            if enable_forced_experts:
                module.num_forced_experts = num_forced_experts
                module.register_buffer("forced_expert_indices", torch.full((num_forced_experts,), -1), persistent=True)
                module.forced_experts_initialized = False
                module._layer_name = name
                
                # Store original forward method
                original_forward = module.__class__.forward
                
                def patched_qwen2_moe_forward(self, hidden_states):
                    """Patched forward for Qwen2 MoE with forced experts"""
                    batch_size, sequence_length, hidden_dim = hidden_states.shape
                    hidden_states_flat = hidden_states.view(-1, hidden_dim)
                    
                    # Router computation with bias
                    router_logits = self.gate(hidden_states_flat)
                    
                    # Add aux-free bias
                    if hasattr(self, 'bias'):
                        if self.bias.device != router_logits.device:
                            self.bias = self.bias.to(router_logits.device)
                        router_logits = router_logits + self.bias

                    # Calculate routing weights
                    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
                    
                    # Standard top-k selection
                    routing_weights_topk, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
                    
                    # Add forced experts if enabled
                    if enable_forced_experts and hasattr(self, 'forced_expert_indices'):
                        if not self.forced_experts_initialized:
                            layer_name = getattr(self, '_layer_name', None)
                            forced_indices = select_forced_experts_qwen2(self, layer_name, forced_experts_records)
                        else:
                            forced_indices = self.forced_expert_indices
                        
                        if self.forced_experts_initialized and torch.any(forced_indices >= 0):
                            # Add forced experts for each token
                            batch_size_flat = routing_weights.shape[0]
                            
                            # Get routing weights for forced experts
                            forced_expert_weights = routing_weights[:, forced_indices]
                            forced_experts_expanded = forced_indices.unsqueeze(0).expand(batch_size_flat, -1)
                            
                            # Combine original top-k with forced experts
                            final_routing_weights = torch.cat([routing_weights_topk, forced_expert_weights], dim=-1)
                            final_selected_experts = torch.cat([selected_experts, forced_experts_expanded], dim=-1)
                            
                            routing_weights_topk = final_routing_weights
                            selected_experts = final_selected_experts

                    # Normalize routing weights
                    if self.norm_topk_prob:
                        routing_weights_topk /= routing_weights_topk.sum(dim=-1, keepdim=True)
                    routing_weights_topk = routing_weights_topk.to(hidden_states.dtype)

                    # Expert processing (same as original Qwen2 logic)
                    final_hidden_states = torch.zeros(
                        (batch_size * sequence_length, hidden_dim), 
                        dtype=hidden_states.dtype, device=hidden_states.device
                    )

                    # One hot encode the selected experts
                    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

                    # Process through experts
                    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
                    for expert_idx in expert_hit:
                        expert_layer = self.experts[expert_idx]
                        idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

                        current_state = hidden_states_flat[None, top_x].reshape(-1, hidden_dim)
                        current_hidden_states = expert_layer(current_state) * routing_weights_topk[top_x, idx, None]
                        final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

                    # Shared expert
                    shared_expert_output = self.shared_expert(hidden_states_flat)
                    shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states_flat)) * shared_expert_output
                    final_hidden_states = final_hidden_states + shared_expert_output

                    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
                    return final_hidden_states, router_logits
                
                # Apply the patch
                module.__class__.forward = patched_qwen2_moe_forward
                patched_modules += 1
    
    print(f"✅ Successfully patched {patched_modules} Qwen2 MoE modules")
    
    # Load bias states if available
    bias_loaded = load_moe_bias_states(model, model_path_or_repo)
    if bias_loaded:
        print("✅ Bias loaded successfully for Qwen2")
    
    return model

def patch_deepseek_model_for_evaluation(model, model_path_or_repo, enable_forced_experts=True, 
                                       num_forced_experts=2, bias_update_speed=1e-4):
    """Patch DeepSeek model for evaluation with forced experts"""
    print("🔧 Applying DeepSeek V2 patches for evaluation...")
    
    # Load forced experts records if available
    forced_experts_records = {}
    if enable_forced_experts:
        forced_experts_records = load_forced_experts_records(model_path_or_repo)
    
    patched_modules = 0
    
    for name, module in model.named_modules():
        if module.__class__.__name__ == "MoEGate":
            print(f"🔧 Patching MoEGate in {name}")
            
            # Add bias to this specific gate instance
            if not hasattr(module, 'bias'):
                module.bias_update_speed = bias_update_speed
                module.register_buffer("bias", torch.zeros(module.n_routed_experts), persistent=True)
                print(f"✅ Added bias to {name}: {module.n_routed_experts} experts")
            
            if enable_forced_experts:
                module.num_forced_experts = num_forced_experts
                module.register_buffer("forced_expert_indices", torch.full((num_forced_experts,), -1), persistent=True)
                module.forced_experts_initialized = False
                module._layer_name = name
                
                # Patch the forward method for the class
                original_forward = module.__class__.forward
                
                def patched_gate_forward(self, hidden_states):
                    bsz, seq_len, h = hidden_states.shape
                    hidden_states = hidden_states.view(-1, h)
                    logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32), None)
                    
                    # Add aux-free bias
                    if hasattr(self, 'bias'):
                        if self.bias.device != logits.device:
                            self.bias = self.bias.to(logits.device)
                        logits = logits + self.bias

                    # Select top-k experts
                    if self.topk_method == "greedy":
                        topk_weight, topk_idx = torch.topk(
                            logits.softmax(dim=-1, dtype=torch.float32), k=self.top_k, dim=-1, sorted=False
                        )

                    # Add forced experts
                    if enable_forced_experts and hasattr(self, 'forced_expert_indices'):
                        if not self.forced_experts_initialized:
                            layer_name = getattr(self, '_layer_name', None)
                            forced_indices = select_forced_experts_deepseek(self, layer_name, forced_experts_records)
                        else:
                            forced_indices = self.forced_expert_indices
                        
                        if self.forced_experts_initialized and torch.any(forced_indices >= 0):
                            batch_size_flat = topk_weight.shape[0]
                            forced_indices = forced_indices.to(logits.device)
                            
                            routing_weights = F.softmax(logits, dim=-1, dtype=torch.float32)
                            forced_expert_weights = routing_weights[:, forced_indices]
                            forced_experts_expanded = forced_indices.to(topk_idx.device).unsqueeze(0).expand(batch_size_flat, -1)
                            
                            final_topk_weight = torch.cat([topk_weight, forced_expert_weights], dim=-1)
                            final_topk_idx = torch.cat([topk_idx, forced_experts_expanded], dim=-1)
                            
                            topk_weight = final_topk_weight
                            topk_idx = final_topk_idx

                    # Norm gate to sum 1
                    if self.top_k > 1 and self.norm_topk_prob:
                        denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
                        topk_weight = topk_weight / denominator
                    else:
                        topk_weight = topk_weight * self.routed_scaling_factor

                    aux_loss = None
                    return topk_idx, topk_weight, aux_loss
                
                module.__class__.forward = patched_gate_forward
                patched_modules += 1
    
    # Apply patches to DeepseekV2MoE modules
    for name, module in model.named_modules():
        if module.__class__.__name__ == "DeepseekV2MoE":
            print(f"🔧 Patching DeepseekV2MoE.forward in {name}")
            original_forward = module.__class__.forward
            
            def patched_moe_forward(self, hidden_states):
                identity = hidden_states
                orig_shape = hidden_states.shape
                topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
                hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
                flat_topk_idx = topk_idx.view(-1)
                # if self.training:
                actual_experts_per_tok = topk_idx.shape[1]
                hidden_states = hidden_states.repeat_interleave(actual_experts_per_tok, dim=0)
                y = torch.empty_like(hidden_states)
                for i, expert in enumerate(self.experts):
                    y[flat_topk_idx == i] = expert(hidden_states[flat_topk_idx == i])
                y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
                y = y.to(hidden_states.dtype).view(*orig_shape)
                # else:
                #     # Use our custom moe_infer that handles forced experts
                #     if enable_forced_experts:
                #         y = patched_moe_infer_with_forced_experts(self, hidden_states, topk_idx, topk_weight).view(*orig_shape)
                #     else:
                #         y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(*orig_shape)
                if self.config.n_shared_experts is not None:
                    y = y + self.shared_experts(identity)
                return y
            
            module.__class__.forward = patched_moe_forward
            patched_modules += 1
    
    print(f"✅ Successfully patched {patched_modules} DeepSeek V2 modules")
    
    # Load bias states if available
    bias_loaded = load_moe_bias_states(model, model_path_or_repo)
    if bias_loaded:
        print("✅ Bias loaded successfully")
    
    return model

# from transformers.cache_utils import DynamicCache
# # 添加get_max_length方法到DynamicCache类
# if not hasattr(DynamicCache, 'get_max_length'):
#     DynamicCache.get_max_length = DynamicCache.get_seq_length



def replace_model_float_forward(model):
    """
    Delete the float forward method in the model to save memory. logits = logits.float() leads to OOM.
    
    Args:
        model: The model to modify
        
    Returns:
        The model with DeepseekV2MoE modules replaced with dense versions
    """
    import types
    if hasattr(model, "forward"):
        # model.forward = types.MethodType(no_float_forward, model)
        model.prepare_inputs_for_generation = types.MethodType(prepare_inputs_for_generation_updated, model)
    else:
        print(f"Warning: Could not find forward method in model")
        # exit()
        return model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model



def prepare_inputs_for_generation_updated(
    self,
    input_ids,
    past_key_values=None,
    attention_mask=None,
    inputs_embeds=None,
    **kwargs,
):
    from transformers.cache_utils import Cache, DynamicCache
    if past_key_values is not None:
        if isinstance(past_key_values, Cache):
            cache_length = past_key_values.get_seq_length()
            past_length = past_key_values.seen_tokens
            # change this for the new version of the cache, original: max_cache_length = past_key_values.get_max_length() will lead to bug
            max_cache_length = past_key_values.get_max_cache_shape()
        else:
            cache_length = past_length = past_key_values[0][0].shape[2]
            max_cache_length = None

        # Keep only the unprocessed tokens:
        # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
        # some of the inputs are exclusivelly passed as part of the cache (e.g. when passing input_embeds as
        # input)
        if (
            attention_mask is not None
            and attention_mask.shape[1] > input_ids.shape[1]
        ):
            input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
        # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
        # input_ids based on the past_length.
        elif past_length < input_ids.shape[1]:
            input_ids = input_ids[:, past_length:]
        # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

        # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
        if (
            max_cache_length is not None
            and attention_mask is not None
            and cache_length + input_ids.shape[1] > max_cache_length
        ):
            attention_mask = attention_mask[:, -max_cache_length:]

    position_ids = kwargs.get("position_ids", None)
    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1] :]

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}

    model_inputs.update(
        {
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }
    )
    return model_inputs





def no_float_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, transformers.,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, transformers., config.vocab_size]`.

        Returns:

        Example:

        python
        >>> from transformers import AutoTokenizer, DeepseekV2ForCausalLM

        >>> model = DeepseekV2ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        >>>  Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you.
        """


        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        # logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



def extract_answer_number(args, sentence: str) -> float:
    dataset = args.dataset.lower()
    if dataset in ["multiarith", "addsub", "singleeq", "gsm8k", "svamp", "mawps"]:
        sentence = sentence.replace(',', '')
        pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
        if not pred:
            return float('inf')
        pred_answer = float(pred[-1])
    else:
        raise NotImplementedError(' not support dataset: {}'.format(dataset))
    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError as e:
            pred_answer = float('inf')
    return pred_answer


def extract_answer_letter(args, sentence: str) -> str:
    sentence_ = sentence.strip()
    pred_answers = re.findall(r'A|B|C|D|E', sentence_)
    if pred_answers:
        if not pred_answers:
            return ''
        return pred_answers[0]
    else:
        return ''

@torch.no_grad()
def main(args):
    accelerator = Accelerator()
    set_random_seed(args.seed)
    t_test_data = json.load(open(args.data_path, 'r'))

    prompts = []
    for example in t_test_data:
        prompt = i_prompt.format_map(example)
        prompts.append(prompt)
    print_rank_0(prompts[0])

    print_rank_0("Loading model and tokenizer...")
    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
    tokenizer.padding_side = "left"
    print_rank_0(f"tokenizer pad side: {tokenizer.padding_side}")
        

    model = create_hf_model(AutoModelForCausalLM,
                        args.model_name_or_path,
                        tokenizer)

    if "deepseek" in args.model_name_or_path.lower():
        model = replace_model_float_forward(model)
        
        # Apply forced experts patches if enabled
        if getattr(args, 'enable_forced_experts', False):
            model = patch_deepseek_model_for_evaluation(
                model, 
                args.model_name_or_path,
                enable_forced_experts=args.enable_forced_experts,
                num_forced_experts=getattr(args, 'num_forced_experts', 2),
                bias_update_speed=getattr(args, 'bias_update_speed', 1e-4)
            )
    elif "qwen" in args.model_name_or_path.lower():
        # Apply Qwen2 patches for forced experts if enabled
        if getattr(args, 'enable_forced_experts', False):
            model = patch_qwen2_model_for_evaluation(
                model,
                args.model_name_or_path,
                enable_forced_experts=args.enable_forced_experts,
                num_forced_experts=getattr(args, 'num_forced_experts', 2),
                bias_update_speed=getattr(args, 'bias_update_speed', 1e-4)
            )

    # Rebuttal experiment: per-layer-r override (runs AFTER uniform patch).
    if PER_LAYER_R_GROUPS and getattr(args, 'enable_forced_experts', False):
        _apply_per_layer_forced_experts_eval(model, PER_LAYER_R_GROUPS)

    model = model.to(accelerator.device)
    args.dtype = torch.float16 if args.dtype == 'fp16' else torch.float32 if args.dtype == 'fp32' else torch.bfloat16
    model = model.to(args.dtype)
    model.eval()
    print_rank_0('model is dtype: {}'.format(model.dtype))


    generation_config = GenerationConfig(
        temperature=args.temperature,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        pad_token_id=model.config.pad_token_id,
        eos_token_id=model.config.eos_token_id,
        bos_token_id=model.config.bos_token_id,
    )
    accelerator.wait_for_everyone()
    device = accelerator.device
    with accelerator.split_between_processes(prompts) as prompt:
        model_outputs = []
        outputs = generate_completions(
            # seq_len
            model=model,
            device=device,
            tokenizer=tokenizer,
            prompts=prompt,
            max_new_tokens=600,                          
            batch_size=args.per_device_eval_batch_size,
            stop_id_sequences=[[tokenizer.eos_token]],
            verbose=False,
            generation_config = generation_config
        )
        model_outputs.extend(outputs)
    outputs = gather_object(model_outputs)

    save_outputs = []
    correct = 0
    miss = 0.001
    for example, output in zip(t_test_data, outputs):
        example['raw_output'] = output
        target = example["answer"]
        if args.dataset.lower() in ['aqua']:
            predict = extract_answer_letter(args, output)
            if target == predict:
                correct += 1
        else:
            predict = extract_answer_number(args, output)
            if abs(float(target) - predict) <= miss:
                correct += 1

        example['prediction'] = predict
        save_outputs.append(example)

    print_rank_0(f"Saving outputs to {args.output_dir}")

    weighted_acc = correct/len(t_test_data)
    print_rank_0("Result {:.1f}, total: {}".format(weighted_acc * 100, len(t_test_data)))


    with open(os.path.join(args.output_dir, f"model_predictions.jsonl"), "w") as fout:
        for example in save_outputs:
            fout.write(json.dumps(example) + "\n")




@torch.no_grad()
def generate_completions(model, device, tokenizer, prompts, batch_size=1, stop_id_sequences=None, disable_tqdm=False, verbose=False, **generation_kwargs):
    generations = []
    if hasattr(model, "module"):
        print_rank_0(f'-----{model.module.generation_config}-----')
    else:
        print_rank_0(f'-----{model.generation_config}-----')

    if generation_kwargs:
        print_rank_0(f'-----{generation_kwargs}-----')
    
    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(prompts), desc="Generating Completions")

    num_return_sequences = generation_kwargs.get("num_return_sequences", 1)
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        tokenized_prompts = tokenizer(batch_prompts, padding = 'longest', return_tensors="pt")
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask
        batch_input_ids = batch_input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # try:
        batch_outputs = model.generate(
            input_ids=batch_input_ids,
            attention_mask=attention_mask,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=[KeyWordsCriteria(stop_id_sequences)] if stop_id_sequences else None,
            **generation_kwargs
        )
        batch_outputs = batch_outputs.detach().cpu()

        # the stopping criteria is applied at batch level, so if other examples are not stopped, the entire batch will continue to generate.
        # so some outputs still have the stop sequence, which we need to remove.
        if stop_id_sequences:
            for output_idx in range(batch_outputs.shape[0]):
                for token_idx in range(batch_input_ids.shape[1], batch_outputs.shape[1]):
                    if any(batch_outputs[output_idx, token_idx: token_idx+len(stop_sequence)].tolist() == stop_sequence for stop_sequence in stop_id_sequences):
                        batch_outputs[output_idx, token_idx:] = tokenizer.pad_token_id
                        break

        # in case piece id out of range
        #batch_outputs[batch_outputs >= tokenizer.vocab_size] = tokenizer.unk_token_id
        #batch_outputs[batch_outputs == -1] = tokenizer.unk_token_id
        
        # remove the prompt from the output
        # we need to re-encode the prompt because we need to make sure the special tokens are treated the same way as in the outputs.
        # we changed our previous way of truncating the output token ids dicrectly because some tokenizer (e.g., llama) won't add space token before the first token.
        # space is important for some tasks (e.g., code completion).
        batch_outputs = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)
        batch_prompts = tokenizer.batch_decode(batch_input_ids, skip_special_tokens=True)
        # duplicate the prompts to match the number of return sequences
        batch_prompts = [prompt for prompt in batch_prompts for _ in range(num_return_sequences)]
        batch_generations = [
            output[len(prompt):] for prompt, output in zip(batch_prompts, batch_outputs)
        ]
        # except Exception as e:
        #     print("Error when generating completions for batch:")
        #     print("Error message:")
        #     print(e)
        #     print("Use empty string as the completion.")
        #     batch_generations = [""] * len(batch_prompts) * num_return_sequences

        generations += batch_generations

        if verbose:
            print("--------")
            print(batch_generations[0])
            
        if not disable_tqdm:
            progress.update(len(batch_prompts)//num_return_sequences)

    assert len(generations) == len(prompts) * num_return_sequences, "number of generations should be equal to number of prompts * num_return_sequences"
    return generations

class KeyWordsCriteria(StoppingCriteria):
    def __init__(self, stop_id_sequences):
        assert isinstance(stop_id_sequences[0], list), "stop_id_sequences should be a list of list of ids"
        self.stop_sequences = stop_id_sequences

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        sequences_should_be_stopped = []
        for i in range(input_ids.shape[0]):
            for stop_sequence in self.stop_sequences:
                if input_ids[i][-len(stop_sequence):].tolist() == stop_sequence:
                    sequences_should_be_stopped.append(True)
                    break
            sequences_should_be_stopped.append(False)
        return all(sequences_should_be_stopped)
    
def load_hf_tokenizer(model_name_or_path,
                      fast_tokenizer=True,
                      add_special_tokens=None):
    if os.path.exists(model_name_or_path):
        # Locally tokenizer loading has some issue, so we need to force download
        model_json = os.path.join(model_name_or_path, "config.json")
        if os.path.exists(model_json):
            model_json_file = json.load(open(model_json))
            model_name = model_json_file.get("_name_or_path",
                                             model_name_or_path)
            tokenizer = get_tokenizer(model_name,
                                      fast_tokenizer=fast_tokenizer)
    else:
        tokenizer = get_tokenizer(model_name_or_path,
                                  fast_tokenizer=fast_tokenizer)

    if add_special_tokens is not None:
        add_special_tokens = [add_special_tokens] if isinstance(add_special_tokens, str) \
            else add_special_tokens
        tokenizer.add_special_tokens(
            {'additional_special_tokens': add_special_tokens})
    return tokenizer

def get_tokenizer(model_name_or_path, fast_tokenizer=True):
    if "llama" in model_name_or_path.lower():
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, fast_tokenizer=fast_tokenizer, add_bos_token = False)       # not adding start token 
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            tokenizer.padding_side = 'right'
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, fast_tokenizer=fast_tokenizer, add_bos_token = False)      # not adding start token 
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'
    return tokenizer

def print_rank_0(msg, rank=None):
    if rank is not None and rank <= 0:
        print(msg)
    elif is_rank_0():
        print(msg)

def is_rank_0():
    """Check whether it is rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            return True
        else:
            return False
    else:
        return True

def set_random_seed(seed):
    import random
    import numpy as np
    import torch
    # from accelerate import get_accelerator
    from transformers import set_seed
    if seed is not None:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # get_accelerator().manual_seed_all(seed)

def create_hf_model(model_class,
                    model_name_or_path,
                    tokenizer,
                    trained=False,
                    dropout=None):
    model_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    configure_dropout(model_config, dropout)
    print_rank_0(f"Creating model {model_class} from {model_name_or_path}")
    if trained:
        # the weight loading is handled by create critic model
        model = model_class.from_config(model_config, trust_remote_code=True)
    else:
       model = model_class.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=model_config, trust_remote_code=True
            )

    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    import math
    model.resize_token_embeddings(int(
        8 *
        math.ceil(len(tokenizer) / 8.0)))  # make the vocab size multiple of 8

    return model


def configure_dropout(model_config, dropout):
    if dropout is not None:
        for key in ('dropout', 'attention_dropout', 'hidden_dropout',
                    'activation_dropout'):
            if hasattr(model_config, key):
                print(f"Setting model_config.{key} to {dropout}")
                setattr(model_config, key, dropout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="", required=True)
    parser.add_argument("--dataset", type=str, default="", required=True)
    parser.add_argument("--output_dir", type=str, default="", required=True)
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="A seed for reproducible training.")
    parser.add_argument("--temperature",
                        type=float,
                        default=0.1,
                        help="temperature during generation.")
    parser.add_argument('--dtype',
                        type=str,
                        default='fp16',
                        choices=['fp16', 'bf16', 'fp32'],
                        help='Inference data type')
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16, help="batch size for evaluation.")
    
    # Forced experts parameters
    parser.add_argument("--enable_forced_experts", action="store_true", help="Enable forced expert activation")
    parser.add_argument("--num_forced_experts", type=int, default=2, help="Number of experts to force activate")
    parser.add_argument("--bias_update_speed", type=float, default=1e-4, help="Bias update speed (for loading compatibility)")
    
    args = parser.parse_args()

    main(args) 
        
    
