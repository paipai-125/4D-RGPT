import os
# 在导入任何可能使用OpenMP的库之前设置环境变量
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['VECLIB_MAXIMUM_THREADS'] = '4'

import json
import cv2
import torch
import torch.nn as nn
import math
import numpy as np
from PIL import Image
from transformers import AutoProcessor
import types

# User requested specifically for Qwen3 model class
try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    from transformers import AutoModelForCausalLM as Qwen3VLForConditionalGeneration

import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# 1. Define TimePositionalEncoding (from student_model.py)
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
# 2. Patching Logic
# -----------------------------------------------------------------------------
def apply_time_encoding_patch(model, device):
    """
    Monkey-patch the visual encoder to inject Time Positional Encoding 
    BEFORE the projector (Merger/MLP).
    """
    
    print("Applying Time Positional Encoding Patch...")

    # 1. Locate the Visual Module
    # Try finding 'visual' in model or model.model
    if hasattr(model, "visual"):
        visual_module = model.visual
        parent_module = model
    elif hasattr(model, "model") and hasattr(model.model, "visual"):
        visual_module = model.model.visual
        parent_module = model.model
    else:
        raise AttributeError("Could not find 'visual' module in model. Checked model.visual and model.model.visual")

    print(f"Found visual module at: {'model.visual' if parent_module == model else 'model.model.visual'}")
    
    # 2. Initialize Time Encoding Module
    # We need to detect vision hidden size. 
    # Usually model.visual.config.embed_dim or hidden_size
    vision_config = getattr(visual_module, "config", None)
    embed_dim = getattr(vision_config, "embed_dim", 1152) if vision_config else 1152
    
    print(f"Detected Vision Embed Dim: {embed_dim}")
    time_embed_module = TimePositionalEncoding(d_model=embed_dim).to(device)
    
    # Attach it to the model so it moves with it and saves with it (technically)
    model.time_embed_module = time_embed_module
    
    # 3. Capture the original forward of the visual module
    original_visual_forward = visual_module.forward
    
    # Check for merger
    if hasattr(visual_module, "merger"):
        print("Found 'merger' in visual module. Patching patch_merger.forward...")
        original_merger_forward = visual_module.merger.forward
    else:
        # Fallback if merger is named differently or doesn't exist (e.g. old Qwen-VL)
        # But Qwen2-VL usually has it.
        print("Warning: 'merger' not found in visual module. Time encoding might not be applied correctly if not using merger.")
        # Proceed assuming standard Qwen2-VL structure or fail?
        # Let's try to proceed, maybe it's named 'attn_pool' or similar in other versions.
        # Check for 'attn_pool'
        if hasattr(visual_module, "attn_pool"):
             print("Found 'attn_pool', treating as merger.")
             original_merger_forward = visual_module.attn_pool.forward
             visual_module.merger = visual_module.attn_pool # Alias it for below code
        else:
             raise AttributeError("Could not find 'merger' or 'attn_pool' in visual module.")

    # 4. Define wrapper for Merger
    class TimeAwareMerger(nn.Module):
        def __init__(self, original_merger, time_embed_module, parent_visual):
            super().__init__()
            self.merger = original_merger
            self.time_embed = time_embed_module
            self.parent = parent_visual
            
        def forward(self, hidden_states):
            # hidden_states: [TotalTokens, Dim]
            # grid_thw: [B, 3] (T, H, W) stored in self.parent.current_grid_thw
            
            grid_thw = getattr(self.parent, "current_grid_thw", None)
            
            if grid_thw is not None:
                # Add Time Embeddings
                # We need to construct a timestamp tensor matching hidden_states
                
                # grid_thw contains (T, H, W) for each sample in batch.
                # Pixel values are flattened.
                # We need to iterate over grid_thw to create time indices.
                
                start_idx = 0
                all_time_embeds = []
                
                device = hidden_states.device
                dtype = hidden_states.dtype
                
                for i in range(grid_thw.shape[0]):
                    T, H, W = grid_thw[i]
                    T, H, W = int(T.item()), int(H.item()), int(W.item())
                    
                    # Number of patches per frame = H * W
                    num_patches_per_frame = H * W
                    total_tokens = T * num_patches_per_frame
                    
                    # Create timestamps
                    if T > 0:
                        # shape [T, H*W] -> flatten
                        times = torch.arange(T, device=device).unsqueeze(1).repeat(1, num_patches_per_frame).flatten()
                        # Get embeddings
                        t_embed = self.time_embed(times) # [Tokens, Dim]
                        all_time_embeds.append(t_embed)
                    else:
                        # Should not happen for video
                        all_time_embeds.append(torch.zeros(total_tokens, hidden_states.shape[-1], device=device, dtype=dtype))
                        
                    start_idx += total_tokens
                
                if all_time_embeds:
                    time_embeds_tensor = torch.cat(all_time_embeds, dim=0).to(dtype)
                    # Check shape match
                    if time_embeds_tensor.shape[0] == hidden_states.shape[0]:
                         print(f"Vision Features Shape: {hidden_states.shape}")
                         print(f"Time Positional Encoding Shape: {time_embeds_tensor.shape}")
                         hidden_states = hidden_states + time_embeds_tensor
                         print(f"Shape after Adding Time Encoding (Input to Projector): {hidden_states.shape}")
            
            return self.merger(hidden_states)

    # 5. Define robust wrapper for Visual Forward
    def captured_visual_forward_generic(self, *args, **kwargs):
        # self is passed implicitly to bound method if calling visual_module.forward
        # But here, we bind it manually later or just use the closure over original_visual_forward
        
        captured_grid = kwargs.get("grid_thw", None)
        
        # Heuristic search for grid_thw in positional args if not in kwargs
        if captured_grid is None:
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    # grid_thw is [TotalFrames, 3] usually, or [B, 3] if batching videos?
                    # Qwen2-VL: grid_thw is 2D tensor of shape [batch_size * time_length, 3]
                    # Values are (t, h, w). Usually integer types.
                    if arg.dim() == 2 and arg.shape[-1] == 3:
                         # Likely grid_thw
                         captured_grid = arg
                         break
        
        self.current_grid_thw = captured_grid
        return original_visual_forward(*args, **kwargs)
        
    # Apply patches
    visual_module.forward = types.MethodType(captured_visual_forward_generic, visual_module)
    if hasattr(visual_module, "merger"):
        visual_module.merger = TimeAwareMerger(original_merger_forward, time_embed_module, visual_module)
    elif hasattr(visual_module, "attn_pool"):
        visual_module.attn_pool = TimeAwareMerger(original_merger_forward, time_embed_module, visual_module)
    
    print("Patch applied successfully.")
    return model

# -----------------------------------------------------------------------------
# 3. Main Functions (Same as qwen3.py but with processing logic)
# -----------------------------------------------------------------------------

def load_video(video_path, num_frames=16, target_resolution=336, save_dir=None):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
        
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        raise ValueError("Video has 0 frames.")
        
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (target_resolution, target_resolution), interpolation=cv2.INTER_CUBIC)
        
        if save_dir:
            frame_save_path = os.path.join(save_dir, f"frame_{i:04d}.jpg")
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(frame_save_path, frame_bgr)
             
        frame_pil = Image.fromarray(frame)
        frames.append(frame_pil)
        
    cap.release()
    
    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else Image.new('RGB', (target_resolution, target_resolution)))
    
    return frames

def process_video_with_qwen(video_frames, processor, model, device, question_text):
    refined_question = question_text + " Answer concisely in a single sentence."
    
    messages = [
        {
            "role": "user",
            "content": [
                *[{"type": "image"} for _ in video_frames],
                {"type": "text", "text": refined_question}
            ]
        }
    ]
    
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    
    inputs = processor(
        text=[text],
        images=video_frames,
        padding=True,
        return_tensors="pt"
    )
    
    inputs = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id
        )
    
    generated_ids_trimmed = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs["input_ids"], generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )[0]
    
    return output_text

def process_video_with_single_frame(video_frames, processor, model, device, question_text):
    refined_question = question_text + " Answer concisely in a single sentence."

    key_frame = video_frames[0] if video_frames else Image.new('RGB', (336, 336))
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": refined_question}
            ]
        }
    ]
    
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    
    inputs = processor(
        text=[text],
        images=[key_frame],
        padding=True,
        return_tensors="pt"
    )
    
    inputs = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id
        )
    
    generated_ids_trimmed = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs["input_ids"], generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )[0]
    
    return output_text

def main():
    local_model_path = "./src/models/Qwen3-VL-8B-Instruct" 
    data_json_path = "./RoboFAC/training_qa.json"
    data_root = "./RoboFAC/simulation_data"
    num_frames = 16
    target_resolution = 448
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if not os.path.exists(local_model_path):
        print(f"Error: Local model not found at {local_model_path}")
        return
    
    try:
        print("Loading processor and model from local directory...")
        processor = AutoProcessor.from_pretrained(local_model_path, trust_remote_code=True, use_fast=False)
        
        if device == "cuda":
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                local_model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            ).eval()
        else:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                local_model_path,
                torch_dtype=torch.float32,
                device_map="auto",
                trust_remote_code=True
            ).eval()
            
        print("Model loaded successfully!")
        
        # APPLY PATCH HERE
        apply_time_encoding_patch(model, device)
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    try:
        with open(data_json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    valid_samples = []
    for item in data:
        vid_path = os.path.join(data_root, item.get('video', ''))
        if os.path.exists(vid_path):
            valid_samples.append(item)
            if len(valid_samples) >= 5:
                break
    
    if not valid_samples:
        print("No valid video files found.")
        return
    
    print(f"Found {len(valid_samples)} samples to test.")
    
    for i, sample in enumerate(valid_samples):
        print(f"\n[{i+1}/{len(valid_samples)}] Testing sample ID: {sample.get('id', 'N/A')}")
        
        video_abs_path = os.path.join(data_root, sample.get('video', ''))
        question = sample.get('conversations', [{}])[0].get('value', '')
        question_text = question.replace("<video>\n", "").replace("<video>", "")
        
        try:
            # We don't save frames here to avoid clutter during this quick test, 
            # unless needed for debugging.
            video_frames = load_video(
                video_abs_path, 
                num_frames=num_frames, 
                target_resolution=target_resolution
            )
        except Exception as e:
            print(f"Error loading video: {e}")
            continue
        
        try:
            response = process_video_with_qwen(
                video_frames, processor, model, device, question_text
            )
            
            if len(response) < 10 or "sorry" in response.lower() or "I cannot" in response:
                print("Multi-frame response not satisfactory, trying single frame...")
                response = process_video_with_single_frame(
                    video_frames, processor, model, device, question_text
                )
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error during inference: {e}")
            response = "Error: Unable to process video"
        
        print("=" * 60)
        print("MODEL RESPONSE:")
        print(response)
        print("GROUND TRUTH:")
        ground_truth = sample.get('conversations', [{}])[1].get('value', 'No ground truth')
        print(ground_truth)
        print("=" * 60)
        
        result_dir = "./qwen_time_results"
        os.makedirs(result_dir, exist_ok=True)
        result_path = os.path.join(result_dir, f"result_{sample.get('id', f'sample_{i}')}.txt")
        
        with open(result_path, 'w', encoding='utf-8') as f:
            f.write(f"Sample ID: {sample.get('id', 'N/A')}\n")
            f.write(f"Video: {sample.get('video', 'N/A')}\n")
            f.write(f"Question: {question_text}\n")
            f.write(f"\nModel Response:\n{response}\n")
            f.write(f"\nGround Truth:\n{ground_truth}\n")
            f.write(f"\nProcessing Info:\n")
            f.write(f"  Model: Local - {local_model_path}\n")
            f.write(f"  Time Encoding: Enabled\n")
        
        print(f"Result saved to {result_path}")

if __name__ == "__main__":
    main()
