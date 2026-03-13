import os
import sys
import torch
import cv2
import json
import glob
import logging
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.student_model import FullStudentModel
from src.models.teacher_model import TeacherWrapper
from src.training.loss_utils import DistillationLoss

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoDataset(Dataset):
    def __init__(self, json_path, data_root, num_frames=16):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.data_root = data_root
        self.num_frames = num_frames
        logger.info(f"Dataset: Loaded {len(self.data)} samples from {json_path}.")
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        video_rel_path = item.get('video', '')
        path = os.path.join(self.data_root, video_rel_path)
        
        try:
            frames, duration = self._load_frames(path)
            
            # Extract prompt and target
            conversations = item.get('conversations', [])
            prompt = "Describe this video."
            target_text = ""
            
            if len(conversations) > 0:
                if conversations[0].get('from') == 'human':
                    prompt = conversations[0].get('value', '').replace("<video>\n", "").replace("<video>", "")
                if len(conversations) > 1 and conversations[1].get('from') == 'assistant':
                    target_text = conversations[1].get('value', '')
            
            return {
                "frames": frames, 
                "duration": duration,
                "path": path,
                "prompt": prompt,
                "target_text": target_text
            }  
        except Exception as e:
            # logger.warning(f"Failed to load video {path}: {e}")
            return None
            
    def _load_frames(self, path):
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total / fps if fps > 0 else 1.0 # Default 1s if fps is weird or unknown

        if total < 1: 
            cap.release()
            raise ValueError("Empty video")
        
        # Uniform sampling logic matching my_demo.py
        if total < self.num_frames:
             # Loop padding like my_demo.py
             indices = list(range(total))
             while len(indices) < self.num_frames:
                 indices.append(indices[-1])
             indices = indices[:self.num_frames]
        else:
             indices = np.linspace(0, total-1, self.num_frames).astype(int)
             
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, f = cap.read()
            if ret:
                f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                # Resize consistent with qwen_time.py default (336)
                f = cv2.resize(f, (336, 336), interpolation=cv2.INTER_CUBIC)
                frames.append(f)
            else:
                # Fallback black frame
                frames.append(np.zeros((336, 336, 3), dtype=np.uint8))
        cap.release()
        return frames, duration

def collate_fn(batch):
    # Filter None
    batch = [b for b in batch if b is not None]
    return batch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "L4P-main"))
import l4p

def main():
    # --- Configuration ---
    # Paths (adjust as needed)
    qwen_path = "./src/models/Qwen3-VL-8B-Instruct" 
    l4p_ckpt = "L4P-main/weights/l4p_depth_flow_2d3dtrack_camray_dynseg_v1.ckpt"
    l4p_config = "L4P-main/configs/model.yaml"
    video_dir = "RoboFAC/simulation_data" 
    json_path = "RoboFAC/training_qa.json"
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    
    # Hyperparameters per paper / request
    epochs = 5
    accum_steps = 1024 # Distributed Batch Size simulation
    learning_rate = 1e-5
    warmup_ratio = 0.03
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # --- Initialize Models ---
    
    # 1. Teacher (Frozen, same as my_demo)
    logger.info("Initializing Teacher (L4P)...")
    teacher = TeacherWrapper(l4p_ckpt, l4p_config, device=device)
    
    # 2. Student (Qwen + LoRA + Ep + Dm)
    logger.info("Initializing Student (Qwen3-VL)...")
    # TBD: Check if Qwen3 is available, else fallback handled in stud.py logic inside wrapper
    student = FullStudentModel(qwen_path, teacher_embed_dim=1024)
    # student.to(device) # device_map='auto' handles placement for llm. Submodules use llm device.
    student.train()
    
    # 3. Processor
    processor = AutoProcessor.from_pretrained(qwen_path, trust_remote_code=True, use_fast=False)
    
    # 4. Optimizer & Scheduler
    # Parameters to optimize: Ep (Projector) + LLM (LoRA) + D4DP (Decoder)
    # Filter parameters with requires_grad=True
    trainable_params = [p for p in student.parameters() if p.requires_grad]
    logger.info(f"Trainable Parameters Count: {sum(p.numel() for p in trainable_params)}")
    
    # Debug: Print trainable module names
    logger.info("Trainable Modules:")
    trainable_names = set()
    for name, param in student.named_parameters():
        if param.requires_grad:
            # Simplify name to module level
            clean_name = name.split('.')[0]
            if "d4dp" in name: clean_name = "D4DP"
            elif "lora" in name: clean_name = "LLM_LoRA"
            elif "merger" in name or "attn_pool" in name: clean_name = "Ep (Projector)"
            elif "bias" in name: clean_name = "Bias" # LoRA might train biases
            trainable_names.add(clean_name)
    logger.info(f" - {trainable_names}")
    
    optimizer = optim.AdamW(trainable_params, lr=learning_rate)
    
    dataset = VideoDataset(json_path, video_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn, num_workers=4)
    
    total_steps = (len(dataloader) // accum_steps) * epochs
    if total_steps < 1: total_steps = 1
    
    scheduler = OneCycleLR(
        optimizer, 
        max_lr=learning_rate, 
        total_steps=total_steps, 
        pct_start=warmup_ratio, 
        div_factor=10, 
        final_div_factor=100
    )
    
    criterion = DistillationLoss(weights={"depth": 1.0, "flow": 0.1, "mask": 0.05, "camray": 0.05})
    
    logger.info("Starting Training...")
    
    global_step = 0
    running_loss = 0.0
    optimizer.zero_grad()
    
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}/{epochs}")
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for step, batch in enumerate(pbar):
            if not batch: continue # Empty batch
            
            # Process sample (Assume batch size 1 for simplicity in processing inputs)
            sample = batch[0]
            frames = sample["frames"]
            prompt = sample["prompt"] 
            target_text = sample["target_text"]
            duration = sample["duration"]
            
            # ---------------------------
            # 1. Teacher Targets
            # ---------------------------
            with torch.no_grad():
                # Teacher takes list of frames -> tensor
                teacher_targets = teacher.get_targets(frames)
                
            if teacher_targets.get("latent") is None:
                continue
            
            # Debug Shapes (Teacher)
            print("="*40)
            print(f"Video Path: {sample['path']}")
            print(f"Ground Truth Text: {target_text}")
            
            t_lat = teacher_targets.get("latent")
            if isinstance(t_lat, list) and len(t_lat) > 0 and isinstance(t_lat[0], list):
                 # Flatten check
                 print(f"Teacher E4D Latent Feature (Sample 0, Frame 0) Shape: {t_lat[0][0].shape}")
            elif isinstance(t_lat, torch.Tensor):
                 print(f"Teacher E4D Latent Feature Shape: {t_lat.shape}")

            for t_task in ["depth", "flow_2d_backward", "dyn_mask", "camray"]:
                val = teacher_targets.get(t_task)
                if val is not None:
                     if isinstance(val, list):
                        s_shp = val[0].shape if len(val)>0 else "Empty"
                        print(f"Teacher Dm Output ({t_task}) List[0] Shape: {s_shp}")
                     else:
                        print(f"Teacher Dm Output ({t_task}) Shape: {val.shape}")

            # ---------------------------
            # 2. Student Inputs & Forward
            # ---------------------------
            text_prompt = prompt + " Answer concisely in a single sentence."
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": frames}, # Processor handles list of HWC numpy arrays? Yes for Qwen2-VL.
                        {"type": "text", "text": text_prompt},
                    ],
                }
            ]

             # Add assistant response for SFT if available
            if target_text:
                messages.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": target_text}]
                })
            
            # Prepare Inputs
            # 1. Full Text: User Prompt + Assistant Response (if any)
            # Use assistant messages if target text exists
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            
            # 2. Tokenize Full Input
            processor_outputs = processor(
                text=[text], 
                videos=[frames], 
                padding=True, 
                return_tensors="pt"
            )
            
            # 3. Create Labels (Mask User Prompt)
            labels = processor_outputs["input_ids"].clone()
            
            # Get length of prompt part alone to mask
            if target_text:
                # Prompt-only part (User message + Generation Prompt)
                # Note: We re-use messages[0] which is the user prompt
                prompt_messages = [messages[0]]
                prompt_text = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
                
                # Tokenize prompt alone with same images to get correct token length
                with torch.no_grad():
                    prompt_inputs = processor(
                        text=[prompt_text],
                        videos=[frames],
                        padding=True,
                        return_tensors="pt"
                    )
                prompt_len = prompt_inputs["input_ids"].shape[1]
                
                # Mask prompt tokens
                labels[:, :prompt_len] = -100
            else:
                # If no target text, mask everything (no SFT loss)
                labels[:] = -100

            model_inputs = {k: v.to(device) for k, v in processor_outputs.items()}
            model_inputs["labels"] = labels.to(device)
            # Pass duration scalar (list if batch > 1, but here batch=1)
            model_inputs["duration"] = [duration]
            
            # Student Forward
            student_out = student(**model_inputs)
            
            # Predict Text for Debug
            if step % 10 == 0: 
                # Print argmax text
                logits = student_out["logits"]
                pred_ids = torch.argmax(logits, dim=-1)
                # Decode first batch
                pred_text = processor.decode(pred_ids[0][-20:], skip_special_tokens=True) # Last 20 tokens
                print(f"Model Output Text (Last 20 tokens): {pred_text}")

            # ---------------------------
            # 3. Loss & Optimization
            # ---------------------------
            # Loss expects list of targets for batch. Only 1 in batch.
            # Wrap targets in list
            batch_targets = {k: [v] if v is not None else [None] for k, v in teacher_targets.items()}
            
            loss_dict = criterion(student_out, batch_targets, labels=model_inputs["labels"])
            loss = loss_dict["total_loss"]
            
            # Normalize loss for accumulation
            loss = loss / accum_steps
            loss.backward()
            
            running_loss += loss.item() * accum_steps # Scale back for logging
            
            if (step + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
                pbar.set_postfix({"loss": f"{running_loss / accum_steps:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})
                running_loss = 0.0
        
        # Save Checkpoint per Epoch
        ckpt_path = os.path.join(save_dir, f"student_epoch_{epoch+1}.pt")
        torch.save(student.state_dict(), ckpt_path)
        logger.info(f"Saved checkpoint to {ckpt_path}")

    logger.info("Training Complete.")

if __name__ == "__main__":
    main()
