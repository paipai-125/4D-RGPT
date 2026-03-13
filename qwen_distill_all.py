import os
# 在导入任何可能使用OpenMP的库之前设置环境变量
os.environ['OMP_NUM_THREADS'] = '4'

import sys
import json
import cv2
import torch
torch.set_float32_matmul_precision('high') 

import torch.nn as nn
import numpy as np
from PIL import Image
from transformers import AutoProcessor, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

# 添加 L4P 路径以供后续导入
sys.path.append(os.path.abspath("./L4P-main"))
sys.path.append(os.path.abspath(".")) # 当前目录

# 尝试导入自定义的蒸馏工具
try:
    from distill_utils import TeacherWrapper, D4DPerception, compute_distillation_loss
except ImportError as e:
    print(f"Error importing distill_utils: {e}")
    sys.exit(1)

# Logger to capture all output
class Logger(object):
    def __init__(self, filename="output_dis.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# Model Import Logic
try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    try:
        from transformers import Qwen2VLForConditionalGeneration as Qwen3VLForConditionalGeneration
    except ImportError:
        from transformers import AutoModel as Qwen3VLForConditionalGeneration

import warnings
warnings.filterwarnings("ignore")

# Load Video Function 
def load_video_pil(video_path, num_frames=16, target_resolution=448):
    """
    为了适配 TeacherWrapper，这里直接返回 PIL Image 列表
    """
    if not os.path.exists(video_path):
        print(f"Warning: Video not found at {video_path}")
        return None
        
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        return None
        
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    
    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 保持原始分辨率或这里 resize？ TeacherWrapper 会处理 resize 到 224
        # Student 需要 448
        # 所以我们这里 resize 到 448 给 Student, TeacherWrapper 内部会再 resize 到 224
        frame = cv2.resize(frame, (target_resolution, target_resolution), interpolation=cv2.INTER_CUBIC)
        frame_pil = Image.fromarray(frame)
        frames.append(frame_pil)
        
    cap.release()
    
    while len(frames) < num_frames:
        if frames:
            frames.append(frames[-1])
        else:
            frames.append(Image.new('RGB', (target_resolution, target_resolution)))
    
    return frames

def main():
    # Redirect stdout
    sys.stdout = Logger("output_dis.txt")
    
    # Configuration
    local_model_path = "./src/models/Qwen3-VL-8B-Instruct"
    data_json_path = "./RoboFAC/training_qa.json"
    data_root = "./RoboFAC/simulation_data"
    num_frames = 16
    target_resolution = 448
    limit_samples = 10
    num_epochs = 10
    learning_rate = 2e-5
    
    # Distillation Params
    alpha = 0.5 # Latent Loss
    beta = 0.1  # Explicit Loss
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ==========================
    # 1. Load Student Model & Processor
    # ==========================
    print("Loading processor and student model...")
    try:
        processor = AutoProcessor.from_pretrained(local_model_path, trust_remote_code=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            local_model_path,
            torch_dtype=torch.float32 if device == "cuda" else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Error loading student model: {e}")
        return
    print("Student Model loaded.")

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    
    # Freeze Visual Encoder
    if hasattr(model, "visual"):
        model.visual.requires_grad_(False)
    elif hasattr(model, "model") and hasattr(model.model, "visual"):
        model.model.visual.requires_grad_(False)

    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["merger", "attn_pool"] # Detect dynamically if needed, kept simple here
    )
    
    # Capture hidden size before PEFT wrapping just in case
    # Qwen3VLConfig has sub-configs. The LLM hidden size is usually in model.config.text_config.hidden_size
    if hasattr(model.config, "text_config") and hasattr(model.config.text_config, "hidden_size"):
        student_hidden_dim = model.config.text_config.hidden_size
    elif hasattr(model.config, "hidden_size"):
        student_hidden_dim = model.config.hidden_size
    else:
        # Fallback for Qwen2-VL/Qwen3-VL specific structure if accessed differently
        # Based on config.json, it's inside text_config
        student_hidden_dim = 3136
        print(f"Warning: Could not auto-detect hidden size, using default {student_hidden_dim}")
    
    model = get_peft_model(model, peft_config)
    
    # ==========================
    # 2. Initialize Teacher Model
    # ==========================
    print("Initializing Teacher Model (L4P)...")
    teacher = TeacherWrapper(device=device) # Loads weights internally
    # Teacher is already frozen in wrapper
    print("Teacher Model initialized.")

    # ==========================
    # 3. Initialize D4DP (4D Perception Decoder)
    # ==========================
    print("Initializing D4DP Decoder...")
    # student_hidden_dim defined above
    d4dp = D4DPerception(input_dim=student_hidden_dim, output_dim=1408)
    d4dp.to(device)
    # Ensure D4DP is in same dtype as model (Float16) if model is Float16
    # Force float16 on cuda
    if device == "cuda":
        d4dp = d4dp.to(dtype=torch.float32, device=device)
    else:
        d4dp = d4dp.to(dtype=torch.float32, device=device)
        
    # D4DP weights need training
    d4dp.train()
    # print(f"D4DP initialized with Input Dim: {student_hidden_dim}")

    # ==========================
    # 4. Optimizer & Data
    # ==========================
    # Optimize both LoRA parameters and D4DP parameters
    params_to_optimize = list(model.parameters()) + list(d4dp.parameters())
    optimizer = torch.optim.AdamW(params_to_optimize, lr=learning_rate)

    print("Loading dataset...")
    # ... (Data loading code omitted for brevity but conceptually same as original)
    with open(data_json_path, 'r') as f:
        full_data = json.load(f)

    all_valid_samples = []
    for item in full_data:
        vid_rel_path = item.get('video')
        if not vid_rel_path: continue
        vid_path = os.path.join(data_root, vid_rel_path)
        if os.path.exists(vid_path):
            conversations = item.get('conversations', [])
            if len(conversations) >= 2:
                question = conversations[0]['value'].replace("<video>\n", "").replace("<video>", "")
                answer = conversations[1]['value']
                all_valid_samples.append({
                    "video_path": vid_path,
                    "question": question,
                    "answer": answer
                })
    
    # Use All Valid Samples (Shuffle for randomness)
    import random
    random.seed(42)
    random.shuffle(all_valid_samples)
    training_samples = all_valid_samples
    
    print(f"Total JSON entries: {len(full_data)}")
    print(f"Valid samples found (videos exist): {len(training_samples)}")
    print(f"Skipped samples: {len(full_data) - len(training_samples)}")
    print(f"Prepared {len(training_samples)} samples for training.")

    # Training Setup
    num_training_steps = num_epochs * len(training_samples)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps
    )
    
    model.train()
    
    # ==========================
    # 5. Training Loop
    # ==========================
    for epoch in range(num_epochs):
        print(f"\n===== Epoch {epoch+1}/{num_epochs} =====")
        epoch_loss_total = 0.0
        epoch_loss_sft = 0.0
        epoch_loss_distill = 0.0
        
        for i, sample in enumerate(training_samples):
            video_path = sample['video_path']
            question = sample['question']
            answer = sample['answer']
            
            # Load frames (List of PIL Images)
            frames_pil = load_video_pil(video_path, num_frames=num_frames, target_resolution=target_resolution)
            if frames_pil is None: continue
                
            # --- Teacher Forward ---
            # Pass frames to teacher (batch size 1 wrapper needs list of list of frames)
            # TeacherWrapper.forward expects [ [frame1, frame2...] ]
            with torch.no_grad():
                f_4d_teacher, p_m_teacher = teacher.forward([frames_pil])
                # print(f"Teacher extracted 4D features shape: {f_4d_teacher.shape}")
                # print("Teacher Explicit Signals Shapes:")
                # for k, v in p_m_teacher.items():
                #     print(f"  {k}: {v.shape}")
                
            # --- Student Input Prep ---
            # Standard Qwen3-VL processing
            messages = [
                {
                    "role": "user",
                    "content": [
                        *([{"type": "image"} for _ in frames_pil]),
                        {"type": "text", "text": question}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": answer}]
                }
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            
            # 手动调整像素尺寸限制
            inputs = processor(
                text=[text],
                images=frames_pil,
                return_tensors="pt",
                padding=True
            )
            # 在某些版本的 transformer 中，videos 可能需要特定的参数来不被压缩
            # 尤其是 'fps' 或 'video_resolution'
            # 但 Qwen2Processor 通常只看 inputs
            
            # # Check grid size (关键)
            # if "image_grid_thw" in inputs:
            #     print(f"DEBUG: Grid THW: {inputs['image_grid_thw']}")
            # elif "video_grid_thw" in inputs:
            #      print(f"DEBUG: Grid Video THW: {inputs['video_grid_thw']}")
            
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
    # Create Labels
            labels = inputs["input_ids"].clone()
            
            # --- SFT Masking Logic: Mask out the user prompt ---
            im_start_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
            # If not found directly, fallback to heuristic or skip precise masking if risky.
            
            if im_start_id is not None:
                # Iterate over batch
                for batch_idx in range(labels.shape[0]):
                    # Find indices of <|im_start|>
                    start_indices = (labels[batch_idx] == im_start_id).nonzero(as_tuple=True)[0]
                    if len(start_indices) > 0:
                        last_start_idx = start_indices[-1]
                        mask_len = last_start_idx + 2 
                        labels[batch_idx, :mask_len] = -100
            
            # Also mask padding
            if processor.tokenizer.pad_token_id is not None:
                labels[labels == processor.tokenizer.pad_token_id] = -100
            
            inputs["labels"] = labels
            
            # =========================================================================
            # Debug: Print sample information and shapes for distilation check
            # =========================================================================
            print(f"-" * 40)
            print(f"Sample {i+1}/{len(training_samples)}")
            # print(f"Video Path: {video_path}")
            # print(f"Question: {question}")
            # print(f"Ground Truth: {answer}")
            
            # --- Student Forward ---
            # Enable output_hidden_states to get embeddings
            outputs = model(**inputs, output_hidden_states=True)
            loss_sft = outputs.loss
            
            # --- Distillation Logic ---
            # 1. Extract Student Hidden States corresponding to Video using Special Tokens
            last_hidden_state = outputs.hidden_states[-1] # (B, Seq_Len, Dim)
            
            # 获取 ID
            vision_start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
            vision_end_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
            
            # 定位视觉 Token
            ids = inputs["input_ids"][0] # (Seq_Len,)
            locs_start = (ids == vision_start_id).nonzero(as_tuple=True)[0]
            locs_end = (ids == vision_end_id).nonzero(as_tuple=True)[0]
            
            loss_ld = 0.0
            loss_ed = 0.0
            loss_distill = 0.0
            
            f_4d_student = None
            
            # Use same logic as qwen_train.py to extract all visual tokens
            if len(locs_start) > 0 and len(locs_end) > 0:
                visual_tokens_list = []
                # Collect tokens from all image segments
                for s, e in zip(locs_start, locs_end):
                    segment = last_hidden_state[:, s+1 : e, :]
                    visual_tokens_list.append(segment)
                
                # Concatenate all visual segments
                student_visual_hidden = torch.cat(visual_tokens_list, dim=1) # (B, Total_Vis, C)
                
                # print(f"Extracted Visual Tokens Count: {student_visual_hidden.shape}")
                
                # 2. D4DP Projection & Input
                # D4DP internally handles arbitrary length if configured, or we pass dimensions
                # Pass grid dimensions to D4DP if available to handle dynamic reshape
                grid_thw = inputs.get("image_grid_thw", None)
                
                # Try to run D4DP
                try:
                    f_4d_student = d4dp(student_visual_hidden, grid_thw=grid_thw) # (B, T_s*H_s*W_s -> T_t*H_t*W_t, C_out)
                    # print(f"D4DP Output Latent 4D Shape (Student): {f_4d_student.shape}")
                    
                    # 3. Explicit Decoding
                    p_m_student = teacher.decode_student_features(f_4d_student)
                    # print("Student Explicit Signals Shapes:")
                    # for k, v in p_m_student.items():
                    #     print(f"  {k}: {v.shape}")

                    # 4. Compute Distillation Loss
                    loss_distill, loss_components = compute_distillation_loss(
                        f_4d_student, 
                        f_4d_teacher, 
                        p_m_student, 
                        p_m_teacher, 
                        alpha=alpha, 
                        beta=beta
                    )
                    
                    if isinstance(loss_components["loss_ld"], torch.Tensor):
                        loss_ld = loss_components["loss_ld"].item()
                    else:
                        loss_ld = loss_components["loss_ld"]
                        
                    if isinstance(loss_components["loss_ed"], torch.Tensor):
                        loss_ed = loss_components["loss_ed"].item()
                    else:
                        loss_ed = loss_components["loss_ed"]
                except RuntimeError as e:
                    print(f"D4DP/Distill Error (Shape Mismatch?): {e}")
                    loss_distill = 0.0 # Skip this sample
            else:
                print("Warning: Could not find vision_start/end tokens. Skipping distill.")
            
            # --- Total Loss & Backward ---
            total_loss = loss_sft + loss_distill
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(params_to_optimize, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            # Logging
            current_lr = optimizer.param_groups[0]['lr']
            # if i % 1 == 0:
            loss_distill_val = loss_distill.item() if isinstance(loss_distill, torch.Tensor) else loss_distill
            print(f"Loss Total: {total_loss.item():.4f} | SFT: {loss_sft.item():.4f} | Distill: {loss_distill_val:.4f} (LD: {loss_ld:.4f}, ED: {loss_ed:.4f}) | LR: {current_lr:.2e}")
            
            epoch_loss_total += total_loss.item()
            epoch_loss_sft += loss_sft.item()
            epoch_loss_distill += loss_distill_val
            
        avg_total = epoch_loss_total / len(training_samples)
        avg_sft = epoch_loss_sft / len(training_samples)
        avg_distill = epoch_loss_distill / len(training_samples)
        print(f"Epoch {epoch+1} Avg Loss: {avg_total:.4f} | Avg SFT: {avg_sft:.4f} | Avg Distill: {avg_distill:.4f}")
        
        # Save Checkpoint (Model + D4DP)
        save_dir = os.path.join("checkpoints_distill", f"epoch_{epoch+1}")
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        processor.save_pretrained(save_dir)
        # Save separate D4DP weights
        torch.save(d4dp.state_dict(), os.path.join(save_dir, "d4dp.pt"))
        print(f"Saved to {save_dir}")

if __name__ == "__main__":
    main()
