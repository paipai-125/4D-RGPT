import os
import sys
import torch
import torch.nn as nn
import traceback
import math 
import types
from transformers import AutoProcessor
from peft import get_peft_model, LoraConfig, TaskType

# Import Qwen3 or fallback
try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    try:
        from transformers import AutoModelForCausalLM as Qwen3VLForConditionalGeneration
    except:
         print("Warning: Qwen3VL class not found, using generic AutoModel.")
         try:
            from transformers import AutoModel as Qwen3VLForConditionalGeneration
         except:
            Qwen3VLForConditionalGeneration = None

try:
    from l4p.models.task_heads.dense_heads import (
        VideoMAEFlowDPTHead,
        VideoMAEDepthDPTHead,
        VideoMAEDynMaskDPTHead,
        VideoMAECameraDPTHead
    )
except ImportError:
    # Define dummy if not found to avoid crash during init if not used
    VideoMAEFlowDPTHead, VideoMAEDepthDPTHead, VideoMAEDynMaskDPTHead, VideoMAECameraDPTHead = None, None, None, None
    pass

# -----------------------------------------------------------------------------
# 1. Define TimePositionalEncoding (From stud.py)
# -----------------------------------------------------------------------------
class TimePositionalEncoding(nn.Module):
    def __init__(self, d_model=1152):
        super().__init__()
        self.d_model = d_model
        # T = 10000, D = d_model
        # div_term = 1 / T^(2i/D)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.register_buffer('div_term', div_term)

    def forward(self, timestamps):
        # timestamps: [N] (tensor of time values, e.g., t(n))
        # Ensure timestamps are float for calculation
        timestamps = timestamps.float()
        
        # Calculate phase: t / T^(2i/D)
        # timestamps: [N], div_term: [D/2] -> [N, D/2]
        phase = timestamps.unsqueeze(1) * self.div_term
        
        pe = torch.zeros(timestamps.size(0), self.d_model, device=timestamps.device, dtype=timestamps.dtype)
        pe[:, 0::2] = torch.sin(phase)
        pe[:, 1::2] = torch.cos(phase)
        
        return pe

# -----------------------------------------------------------------------------
# 2. Patching Logic (From stud.py)
# -----------------------------------------------------------------------------
def apply_time_encoding_patch(model, device):
    """
    Monkey-patch the visual encoder to inject Time Positional Encoding 
    BEFORE the projector (Merger/MLP).
    """
    
    print("Applying Time Positional Encoding Patch...")

    # 1. Locate the Visual Module
    # Try finding 'visual' in model or model.model
    visual_module = None
    parent_module = None
    
    if hasattr(model, "visual"):
        visual_module = model.visual
        parent_module = model
    elif hasattr(model, "model") and hasattr(model.model, "visual"):
        visual_module = model.model.visual
        parent_module = model.model
    # Handle PEFT model wrapping
    elif hasattr(model, "base_model") and hasattr(model.base_model, "model") and hasattr(model.base_model.model, "visual"):
        visual_module = model.base_model.model.visual
        parent_module = model.base_model.model
    
    if visual_module is None:
        print("Warning: Could not find 'visual' module. Patch skipped.")
        return model

    # print(f"Found visual module at: {'model.visual' if parent_module == model else 'model.model.visual'}")
    
    # 2. Initialize Time Encoding Module
    # We need to detect vision hidden size. 
    # Usually model.visual.config.embed_dim or hidden_size
    vision_config = getattr(visual_module, "config", None)
    embed_dim = getattr(vision_config, "embed_dim", 1152) if vision_config else 1152
    
    # print(f"Detected Vision Embed Dim: {embed_dim}")
    time_embed_module = TimePositionalEncoding(d_model=embed_dim).to(device)
    
    # Attach it to the model so it moves with it and saves with it (technically)
    model.time_embed_module = time_embed_module
    
    # 3. Capture the original forward of the visual module
    original_visual_forward = visual_module.forward
    
    # Check for merger
    original_merger_forward = None
    # We need to check if it's already patched
    if hasattr(visual_module, "merger"):
        # print("Found 'merger' in visual module.")
        original_merger_forward = visual_module.merger.forward
    elif hasattr(visual_module, "attn_pool"):
         # print("Found 'attn_pool', treating as merger.")
         original_merger_forward = visual_module.attn_pool.forward
         visual_module.merger = visual_module.attn_pool # Alias it for below code
    else:
         print("Warning: 'merger' or 'attn_pool' not found. Time encoding might fail.")

    if original_merger_forward is None:
        return model

    # 4. Define wrapper for Merger
    class TimeAwareMerger(nn.Module):
        def __init__(self, original_merger_fn, time_embed_module, parent_visual):
            super().__init__()
            # If original_merger_fn is bound method, we might need to be careful?
            # original_merger_fn is likely 'forward' method.
            # We need the MODULE instance to call it properly or call it unbound?
            # Actually, standard way is to wrap the module instance.
            # But here we are replacing the forward method or the module?
            # Let's see how stud.py did it. 
            # In stud.py, it patches 'visual_module.merger = TimeAwareMerger(original_merger, ...)'
            # But the TimeAwareMerger in stud.py calls self.merger(hidden_states).
            # So self.merger must be the Original Merger MODULE, not just forward function.
            
            # Correction: The argument 'original_merger' passed to __init__
            # In stud.py: visual_module.merger is passed. Correct.
            self.merger = original_merger_fn 
            self.time_embed = time_embed_module
            # Avoid strong reference or ensure it doesn't cause recursion issues
            object.__setattr__(self, "parent", parent_visual)
            
        def forward(self, hidden_states):
            # grid_thw: [B, 3] (T, H, W) stored in self.parent.current_grid_thw
            grid_thw = getattr(self.parent, "current_grid_thw", None)
            
            if grid_thw is not None:
                start_idx = 0
                all_time_embeds = []
                device = hidden_states.device
                dtype = hidden_states.dtype
                
                # Iterate batch
                for i in range(grid_thw.shape[0]):
                    # Check if grid_thw has data
                    if grid_thw[i].numel() < 3: continue 
                    
                    T, H, W = grid_thw[i]
                    T, H, W = int(T.item()), int(H.item()), int(W.item())
                    
                    # Number of patches per frame = H * W
                    # (Assuming flattened spatial tokens)
                    num_patches_per_frame = H * W
                    total_tokens = T * num_patches_per_frame
                    
                    # Create timestamps
                    if T > 0:
                        # shape [T, H*W] -> flatten
                        # Use specific time instead of frame index. Assuming uniform sampling.
                        # We use a default fps or relative time if absolute time is not available.
                        # Here we use relative time (0.0 to 1.0) scaled by duration, or simply seconds if fps known.
                        # Since we don't have fps, we assume a canonical duration or use index as seconds (time-aware).
                        # User request: "ensure time encoding is not input frame index, but specific time"
                        # We approximate this by converting index to float time assuming 30fps default or similar logic requires external input.
                        # Without external input, we assume the index represents a time step delta.
                        # To strictly follow "not frame index", we casting to float and potentially scaling.
                        fps = 8.0 # Default assumption for feature extraction
                        times = (torch.arange(T, device=device).float() / fps).unsqueeze(1).repeat(1, num_patches_per_frame).flatten()
                        # Get embeddings
                        t_embed = self.time_embed(times) # [Tokens, Dim]
                        all_time_embeds.append(t_embed)
                    else:
                        # Should not happen for video
                        all_time_embeds.append(torch.zeros(total_tokens, hidden_states.shape[-1], device=device, dtype=dtype))
                        
                    start_idx += total_tokens
                
                if all_time_embeds:
                    time_embeds_tensor = torch.cat(all_time_embeds, dim=0).to(dtype)
                    # Check shape match and add
                    # hidden_states might include <|vision_start|> etc? 
                    # Usually hidden_states at this point (inside merger) are just visual tokens?
                    # In Qwen2-VL, yes.
                    if time_embeds_tensor.shape[0] == hidden_states.shape[0]:
                         print(f"Visual Encoder Output Shape: {hidden_states.shape}")
                         print(f"Time Embeddings Shape: {time_embeds_tensor.shape}")
                         hidden_states = hidden_states + time_embeds_tensor
                         print(f"Projector Input Shape: {hidden_states.shape}")

            return self.merger(hidden_states)

    # 5. Define robust wrapper for Visual Forward
    # To capture grid_thw
    def captured_visual_forward_generic(self, *args, **kwargs):
        captured_grid = kwargs.get("grid_thw", None)
        
        # Heuristic search for grid_thw in positional args if not in kwargs
        if captured_grid is None:
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    # grid_thw is [TotalFrames, 3] usually, or [B, 3]
                    if arg.dim() == 2 and arg.shape[-1] == 3:
                         captured_grid = arg
                         break
        
        self.current_grid_thw = captured_grid
        # Need to call original forward bound to 'self' (the visual_module)
        # We can simulate this by partial application or ...
        # Since we replace visual_module.forward, 'self' here is visual_module.
        # But original_visual_forward is the UNBOUND function? Or Bound?
        # If we captured it as `visual_module.forward`, it's a bound method.
        # Calling bound_method(self, ...) fails with 'multiple values for self'.
        # Calling bound_method(...) works.
        return original_visual_forward(*args, **kwargs)
        
    # Apply patches
    # Only patch if not already patched
    if not hasattr(visual_module, "time_embed_module"):
        # Patch Forward
        visual_module.forward = types.MethodType(captured_visual_forward_generic, visual_module)
        
        # Patch Merger
        # We need to wrap the MODULE, not the function
        if hasattr(visual_module, "merger"):
            actual_merger_module = visual_module.merger
            # Wrap
            visual_module.merger = TimeAwareMerger(actual_merger_module, time_embed_module, visual_module)
        elif hasattr(visual_module, "attn_pool"):
            actual_merger_module = visual_module.attn_pool
            visual_module.attn_pool = TimeAwareMerger(actual_merger_module, time_embed_module, visual_module)
            
        print("Time Encoding Patch applied successfully.")
    else:
        print("Model already patched. Skipping.")
        
    return model

from .components import D4DP_Decoder 

class FullStudentModel(nn.Module):
    def __init__(self, qwen_model_path, hidden_size=3584, teacher_embed_dim=1024):
        super().__init__()
        
        print(f"Loading Qwen3-VL from {qwen_model_path}...")
        try:
            # Try loading as Qwen3VL first
            self.llm = Qwen3VLForConditionalGeneration.from_pretrained(
                qwen_model_path, 
                torch_dtype=torch.float16, 
                device_map="cuda", # Force cuda instead of auto to reduce overhead of map calc
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"Error loading Qwen3 Class: {e}. Trying generic AutoModel...")
            try:
                # Fallback to AutoModel
                self.llm = Qwen3VLForConditionalGeneration.from_pretrained(
                    qwen_model_path, 
                    torch_dtype=torch.float16, 
                    device_map="cuda", 
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
            except Exception as e2:
                print(f"Critial Error loading model: {e2}")
                raise e2
        
        # Apply Time Encoding Patch (Critical for 4D)
        try:
            # Note: apply_time_encoding_patch needs to know device for buffer creation.
            # If auto, device is meta or gpu depending on load.
            device_param = next(self.llm.parameters()).device
            apply_time_encoding_patch(self.llm, device_param)
        except Exception as e:
            print(f"Patching Warning: {e}")

        
        # ---------------------------------------------------------
        # Apply LoRA to LLM (Fine-tune LLM)
        # ---------------------------------------------------------
        print("Applying LoRA to LLM...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=16, 
            lora_alpha=32, 
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] 
        )
        self.llm = get_peft_model(self.llm, peft_config)
        self.llm.print_trainable_parameters()
        
        # Freeze Vision Encoder (Ev)
        visual_module = None
        if hasattr(self.llm, "model") and hasattr(self.llm.model, "visual"):
            visual_module = self.llm.model.visual
        elif hasattr(self.llm, "base_model") and hasattr(self.llm.base_model.model, "visual"):
            visual_module = self.llm.base_model.model.visual
            
        if visual_module:
            print("Freezing Vision Encoder (Ev)...")
            for param in visual_module.parameters():
                param.requires_grad = False
            
            # Unfreeze Ep (Projector)
            # Qwen2-VL's visual module includes the projector (merger/attn_pool)
            print("Unfreezing Visual Projector (Ep)...")
            projector_found = False
            
            # Note: After patching, visual_module.merger might be our Wrapper.
            # We need to unfreeze the UNDERLYING merger params.
            
            merger_module = getattr(visual_module, "merger", None)
            if merger_module:
                # Check if it has 'merger' attribute (our wrapper)
                if hasattr(merger_module, "merger"):
                    # Unfreeze wrapped merger
                    for param in merger_module.merger.parameters():
                        param.requires_grad = True
                    projector_found = True
                    print(" - Unfrozen 'merger' (Wrapped)")
                else:
                    # Unfreeze direct merger
                    for param in merger_module.parameters():
                        param.requires_grad = True
                    projector_found = True
                    print(" - Unfrozen 'merger' (Direct)")
            
            attn_pool = getattr(visual_module, "attn_pool", None)
            if attn_pool and not projector_found:
                 if hasattr(attn_pool, "merger"):
                     for param in attn_pool.merger.parameters():
                        param.requires_grad = True
                 else:
                     for param in attn_pool.parameters():
                        param.requires_grad = True
                 projector_found = True
                 print(" - Unfrozen 'attn_pool'")
                 
            if not projector_found:
                print("Warning: Could not find projector ('merger' or 'attn_pool') in visual module to unfreeze.")

        # ---------------------------------------------------------
        # D4DP Decoder - Trainable
        # ---------------------------------------------------------
        print("Initializing D4DP Decoder - Trainable...")
        # Get actual hidden size from config if possible
        if hasattr(self.llm.config, "text_config"):
            actual_hidden = self.llm.config.text_config.hidden_size
        else:
            actual_hidden = getattr(self.llm.config, "hidden_size", hidden_size)
            
        self.d4dp_decoder = D4DP_Decoder(input_dim=actual_hidden, output_dim=teacher_embed_dim).to(device_param).half()

        # ---------------------------------------------------------
        # Dm (Heads) - Trainable
        # ---------------------------------------------------------
        print("Initializing Perception Heads (Trainable)...")
        self.heads = nn.ModuleDict()
        
        try:
             # Instantiate with correct dims
             # Assuming we transfer weights later or user handles it.
             if VideoMAEDepthDPTHead:
                 self.heads["depth"] = VideoMAEDepthDPTHead(
                     task_name="depth", out_nchan=1, embed_dim=teacher_embed_dim
                 ).to(device_param).half()
             
             if VideoMAEFlowDPTHead:
                 self.heads["flow"] = VideoMAEFlowDPTHead(
                     task_name="flow", out_nchan=2, embed_dim=teacher_embed_dim
                 ).to(device_param).half()
             
             if VideoMAEDynMaskDPTHead:
                 self.heads["mask"] = VideoMAEDynMaskDPTHead(
                     task_name="mask", out_nchan=1, embed_dim=teacher_embed_dim
                 ).to(device_param).half()
             
             if VideoMAECameraDPTHead:
                 self.heads["camera"] = VideoMAECameraDPTHead(
                     task_name="camera", out_nchan=6, embed_dim=teacher_embed_dim
                 ).to(device_param).half()
             
             for name, head in self.heads.items():
                 for param in head.parameters():
                     param.requires_grad = False # Frozen as requested
                 head.eval() 
        except Exception as e:
             print(f"Warning: Failed to instantiate heads: {e}")

    def forward(self, input_ids, attention_mask=None, pixel_values=None, image_grid_thw=None, pixel_values_videos=None, video_grid_thw=None, labels=None, **kwargs):
        """
        Returns dict with loss and outputs.
        """
        # 1. LLM Forward (SFT Loss is computed internally if labels provided)
        # PeftModel forward usually delegates to base model forward.
        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            labels=labels,
            output_hidden_states=True,
            return_dict=True
        )
        
        # 2. 4D Distillation
        last_hidden_state = outputs.hidden_states[-1]
        
        # Collect outputs
        # Simplified: Find the visual part.
        # Qwen2-VL: <|vision_start|> ... <|vision_end|>
        # For batch processing, we locate indices for each sample.
        
        B = input_ids.shape[0]
        latent_features_list = []
        explicit_outputs = {k: [] for k in self.heads.keys()}
        
        # IDs for vision start/end
        vision_start_id = 151652
        vision_end_id = 151653
        
        # Determine which grid to use
        # If video_grid_thw is present, use it.
        current_grid = video_grid_thw if video_grid_thw is not None else image_grid_thw

        for b in range(B):
            if current_grid is None or len(current_grid) <= b:
                # Fallback or Skip
                latent_features_list.append(None)
                for k in explicit_outputs: explicit_outputs[k].append(None)
                continue

            grid = current_grid[b]
            T, H, W = int(grid[0]), int(grid[1]), int(grid[2])
            
            # Locate tokens
            ids = input_ids[b]
            locs_start = (ids == vision_start_id).nonzero(as_tuple=True)[0]
            locs_end = (ids == vision_end_id).nonzero(as_tuple=True)[0]
            
            if len(locs_start) > 0 and len(locs_end) > 0:
                s, e = locs_start[0], locs_end[0]
                vis_tokens = last_hidden_state[b, s+1 : e]
                
                # Check dimensions
                # Qwen2-VL: 14x14 patches, 2x2 pooling -> stride 28
                h_feat, w_feat = H // 28, W // 28
                expected = T * h_feat * w_feat
                
                if vis_tokens.shape[0] == expected:
                    # D4DP Forward
                    # Input: [1, Seq, Dim]
                    # Output: List of 4 features [1, C, T, H, W]
                    d4dp_out = self.d4dp_decoder(vis_tokens.unsqueeze(0), T, h_feat, w_feat)
                    feat = d4dp_out[-1] # Take last level (finest)
                    print(f"D4DP Latent Features Shape (Level -1): {feat.shape}")
                    latent_features_list.append(feat)
                    
                    # Heads Forward
                    img_info = (T, H, W)
                    for task, head in self.heads.items():
                        try:
                            # Heads take the LIST of features
                            pred = head(d4dp_out, img_info=img_info) # (1, C, T, H, W)
                            print(f"Student Dm Output ({task}) Shape: {pred.shape}")
                            explicit_outputs[task].append(pred)
                        except Exception as e:
                            print(f"Dm Error {task}: {e}")
                            explicit_outputs[task].append(None)
                else:
                    latent_features_list.append(None)
                    for k in explicit_outputs: explicit_outputs[k].append(None)
            else:
                latent_features_list.append(None)
                for k in explicit_outputs: explicit_outputs[k].append(None)

        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
            "latent_features": latent_features_list,
            "explicit_outputs": explicit_outputs
        }
