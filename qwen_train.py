import os
import sys

# Logger to capture all output
class Logger(object):
    def __init__(self, filename="output1.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Set environment variables for better performance
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['VECLIB_MAXIMUM_THREADS'] = '4'

import json
import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from transformers import AutoProcessor, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

# User requested specifically for Qwen3 model class
try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    # Fallback to AutoModel if Qwen3 is not explicitly available but registered
    # Or try Qwen2VLForConditionalGeneration as Qwen3 is often compatible
    try:
        from transformers import Qwen2VLForConditionalGeneration as Qwen3VLForConditionalGeneration
    except ImportError:
        from transformers import AutoModel as Qwen3VLForConditionalGeneration

import warnings
warnings.filterwarnings("ignore")

# Load Video Function
def load_video(video_path, num_frames=16, target_resolution=336):

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
    # Redirect stdout to capture all prints
    sys.stdout = Logger("output.txt")
    
    # Configuration
    local_model_path = "./src/models/Qwen3-VL-8B-Instruct"
    data_json_path = "./RoboFAC/training_qa.json"
    data_root = "./RoboFAC/simulation_data"
    num_frames = 16
    target_resolution = 448
    limit_samples = 10
    num_epochs = 10
    learning_rate = 2e-5
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load Model and Processor
    print("Loading processor and model...")
    try:
        processor = AutoProcessor.from_pretrained(local_model_path, trust_remote_code=True)
        # Use Qwen3VLForConditionalGeneration as imported/aliased above
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            local_model_path,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Model loaded.")

    # Enable Gradient Checkpointing for memory efficiency
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("Gradient Checkpointing enabled.")

    # 2. Freeze Visual Encoder
    if hasattr(model, "visual"):
        model.visual.requires_grad_(False)
        print("Frozen 'visual' module.")
    elif hasattr(model, "model") and hasattr(model.model, "visual"):
        model.model.visual.requires_grad_(False)
        print("Frozen 'model.model.visual' module.")
    else:
        print("Warning: Could not identify visual module to freeze.")

    # 3. Configure LoRA
    # Detect projector name to be fully fine-tuned (added to modules_to_save)
    projector_target_modules = []
    
    # Check for merger or attn_pool in visual module
    visual_module = getattr(model, "visual", None)
    if visual_module is None and hasattr(model, "model"):
        visual_module = getattr(model.model, "visual", None)
        
    if visual_module:
        if hasattr(visual_module, "merger"):
            projector_target_modules.append("merger")
            print("Identified 'merger' as projector. Will be fully fine-tuned.")
        elif hasattr(visual_module, "attn_pool"):
            projector_target_modules.append("attn_pool")
            print("Identified 'attn_pool' as projector. Will be fully fine-tuned.")
        else:
            print("Warning: Could not identify projector (merger/attn_pool).")
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=projector_target_modules
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # 4. Load Data
    print("Loading dataset...")
    try:
        with open(data_json_path, 'r') as f:
            full_data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON: {e}")
        return

    training_samples = []
    # Buffer to hold all valid samples before random selection
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
    
    # Randomly select samples
    if len(all_valid_samples) > limit_samples:
        import random
        random.seed(42)
        training_samples = random.sample(all_valid_samples, limit_samples)
        print(f"Randomly selected {limit_samples} samples from {len(all_valid_samples)} valid valid samples.")
    else:
        training_samples = all_valid_samples
        print(f"Found {len(training_samples)} samples (less than limit {limit_samples}). Using all.")
            
    print(f"Prepared {len(training_samples)} samples for training.")
    if len(training_samples) == 0:
        return

    # 5. Training Loop
    model.train()
    
    # Enable grad requirement for inputs if checkpointing is on
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
        print("Enabled input_require_grads for gradient checkpointing compatibility.")

    # Calculate global steps
    num_training_steps = num_epochs * len(training_samples)
    print(f"Total training steps: {num_training_steps}")
    
    # Add Scheduler with Warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps
    )
    
    for epoch in range(num_epochs):
        print(f"\n===== Epoch {epoch+1}/{num_epochs} =====")
        epoch_loss = 0.0
        
        for i, sample in enumerate(training_samples):
            video_path = sample['video_path']
            question = sample['question']
            answer = sample['answer']
            
            print(f"-" * 40)
            print(f"Processing sample {i+1}/{len(training_samples)}")
            print(f"Video Path: {video_path}")
            print(f"Question: {question}")
            print(f"Ground Truth: {answer}")
            
            # Load frames
            frames = load_video(video_path, num_frames=num_frames, target_resolution=target_resolution)
            if frames is None:
                print("Failed to load video frames.")
                continue
                
            # Construct Training Prompt
            messages = [
                {
                    "role": "user",
                    "content": [
                        *([{"type": "image"} for _ in frames]),
                        {"type": "text", "text": question}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": answer}
                    ]
                }
            ]
            
            # Prepare inputs
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            
            # Tokenize & encode
            inputs = processor(
                text=[text],
                images=frames,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Create Labels
            labels = inputs["input_ids"].clone()
            
            # --- SFT Masking Logic: Mask out the user prompt ---
            # Strategy: Find the position of "<|im_start|>assistant" which marks the start of the answer
            # Qwen2-VL specific tokens: <|im_start|> is usually 151644, "assistant" follows
            # A more robust way without hardcoding IDs is to re-tokenize the prompt part, 
            # but that's tricky with images.
            # We will search for the control token sequence.
            
            # Typical sequence for start of answer in Qwen2-VL:
            # <|im_start|> assistant \n 
            # ids: [151644, 77091, 198] (Example, depends on tokenizer)
            
            # Let's inspect the tokenizer to find the ID for "<|im_start|>"
            im_start_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
            # If not found directly, fallback to heuristic or skip precise masking if risky.
            
            # Heuristic: Since we only have 1 turn, we can find the last occurrence of "<|im_start|>" 
            # (which should be the assistant's turn) and mask everything before it + header length.
            
            # However, Qwen2 tokenizer treats <|im_start|> as a special token.
            if im_start_id is not None:
                # Iterate over batch
                for batch_idx in range(labels.shape[0]):
                    # Find indices of <|im_start|>
                    start_indices = (labels[batch_idx] == im_start_id).nonzero(as_tuple=True)[0]
                    if len(start_indices) > 0:
                        # The last <|im_start|> is typically for the assistant in a single-turn setup
                        last_start_idx = start_indices[-1]
                        
                        # We also want to mask the header itself "<|im_start|>assistant\n"
                        # It's safer to just mask up to this start index + a few tokens, or just up to last_start_idx
                        # The Assistant answer starts AFTER the header.
                        # Let's verify the next few tokens are "assistant"
                        # But for now, masking up to the last <|im_start|> is a safe baseline 
                        # (model will learn to predict 'assistant' which is fine, or we mask slightly more)
                        
                        # Let's mask everything BEFORE the last <|im_start|> + 2 tokens (approx header length)
                        # Actually, predicting the start of assistant turn is okay.
                        # Masking up to last_start_idx ensures we don't train on image/user text
                        
                        # Find the newline after assistant? Too complex.
                        # Conservative approach: Mask everything up to last_start_idx + 2
                        mask_len = last_start_idx + 2 
                        labels[batch_idx, :mask_len] = -100
            
            # Also mask padding
            if processor.tokenizer.pad_token_id is not None:
                labels[labels == processor.tokenizer.pad_token_id] = -100
            
            inputs["labels"] = labels
            
            # Debug: Check valid tokens
            valid_tokens = (labels != -100).sum().item()
            total_tokens = labels.numel()
            if i % 5 == 0:
                print(f"  Valid tokens for loss: {valid_tokens}/{total_tokens}")
            
            if valid_tokens == 0:
                print(f"  Warning: No valid tokens for sample {i}! Skipping...")
                continue
            
            # Forward
            optimizer.zero_grad()
            outputs = model(**inputs, output_hidden_states=True)
            loss = outputs.loss
            
            # Extract and Count Visual Tokens
            last_hidden_state = outputs.hidden_states[-1] # (B, Seq_Len, Dim)
            
            # Find Vision Token Range
            vision_start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
            vision_end_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
            
            # We assume batch size 1 for simplicity in this script
            ids = inputs["input_ids"][0] 
            locs_start = (ids == vision_start_id).nonzero(as_tuple=True)[0]
            locs_end = (ids == vision_end_id).nonzero(as_tuple=True)[0]
            
            if len(locs_start) > 0 and len(locs_end) > 0:
                # Assuming single continuous video block or taking the first block
                # For multiple images/video parts, this logic might need adjustment (e.g. summing ranges)
                # Here we just check the total count of tokens inside vision tags.
                
                # Careful: If multiple images are used, we might have multiple start/end pairs.
                # Let's count total visual tokens across all pairs found.
                total_visual_tokens = 0
                for s_idx, e_idx in zip(locs_start, locs_end):
                    # Tokens between start and end (exclusive of tags)
                    count = (e_idx - s_idx - 1).item()
                    total_visual_tokens += count
                
                print(f"  Standard LLM Visual Tokens Count: {total_visual_tokens}")
            else:
                print("  Warning: vision_start/end tokens not found in input_ids.")
            
            if torch.isnan(loss):
                print(f"  Warning: Loss is NaN in sample {i}! Skipping step.")
                continue
                
            loss.backward()
            
            # Clip Gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Loss: {loss.item():.4f} | LR: {current_lr:.2e}")
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(training_samples)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        
        # Save Checkpoint
        save_dir = os.path.join("checkpoints", f"epoch_{epoch+1}")
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        processor.save_pretrained(save_dir)
        print(f"Saved weights to {save_dir}")

if __name__ == "__main__":
    main()
