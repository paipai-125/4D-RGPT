import os
# Set environment variables for better performance
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['VECLIB_MAXIMUM_THREADS'] = '4'

import json
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor
# Peft import for loading adapter
from peft import PeftModel

# User requested specifically for Qwen3 model class
try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    try:
        from transformers import Qwen2VLForConditionalGeneration as Qwen3VLForConditionalGeneration
    except ImportError:
        from transformers import AutoModel as Qwen3VLForConditionalGeneration

import warnings
warnings.filterwarnings("ignore")

# Load Video Function (Same as training)
def load_video(video_path, num_frames=8, target_resolution=224):
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
    # Configuration
    local_model_path = "./src/models/Qwen3-VL-8B-Instruct"
    adapter_path = "checkpoints/epoch_10"
    data_json_path = "./RoboFAC/training_qa.json"
    data_root = "./RoboFAC/simulation_data"
    
    # Must match training config
    num_frames = 16
    target_resolution = 448
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load Base Model and Processor
    print("Loading processor and base model...")
    try:
        processor = AutoProcessor.from_pretrained(local_model_path, trust_remote_code=True)
        # Load base model in bfloat16 as per training
        base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            local_model_path,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Error loading base model: {e}")
        return

    # 2. Load LoRA Adapter
    print(f"Loading LoRA adapter from {adapter_path}...")
    try:
        model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map="auto"
        )
        print("Adapter loaded and merged successfully.")
    except Exception as e:
        print(f"Error loading adapter: {e}")
        return
        
    model.eval()

    # 3. Load Data
    print("Loading dataset...")
    try:
        with open(data_json_path, 'r') as f:
            full_data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON: {e}")
        return

    # -------------------------------------------------------------------------
    # MODIFIED: Use the SAME sampling logic and seed as training to test the SAME 10 samples
    # -------------------------------------------------------------------------
    
    # Buffer to hold all valid samples before random selection
    all_valid_samples = []
    
    for i, item in enumerate(full_data):
        vid_rel_path = item.get('video')
        if not vid_rel_path: continue
        
        vid_path = os.path.join(data_root, vid_rel_path)
        
        # IMPORTANT: qwen_train.py checks for existence. We must do the same to match the pool.
        if os.path.exists(vid_path):
            # Parse Q&A
            conversations = item.get('conversations', [])
            if len(conversations) >= 2:
                question = conversations[0]['value'].replace("<video>\n", "").replace("<video>", "")
                answer = conversations[1]['value']
                all_valid_samples.append({
                    "id": item.get('id', f'sample_{i}'),
                    "video_path": vid_path,
                    "question": question,
                    "ground_truth": answer
                })
    
    # Randomly select samples (Same seed as training)
    import random
    random.seed(42)
    limit_samples = 10
    
    if len(all_valid_samples) > limit_samples:
        test_samples = random.sample(all_valid_samples, limit_samples)
        print(f"Randomly selected {limit_samples} samples from {len(all_valid_samples)} valid valid samples (Same seed as training).")
    else:
        test_samples = all_valid_samples
        print(f"Found {len(test_samples)} samples (less than limit {limit_samples}). Using all.")
            
    print(f"Prepared {len(test_samples)} samples for testing.")

    # 4. Inference Loop
    for i, sample in enumerate(test_samples):
        print(f"\n===== Test Sample {i+1}/{len(test_samples)} =====")
        print(f"Video Path: {sample['video_path']}")
        print(f"Question: {sample['question']}")
        print(f"Ground Truth: {sample['ground_truth']}")
        
        # Load frames
        frames = load_video(sample['video_path'], num_frames=num_frames, target_resolution=target_resolution)
        if frames is None:
            print("Error: Failed to load video frames.")
            continue
            
        # Construct Inference Prompt
        messages = [
            {
                "role": "user",
                "content": [
                    *([{"type": "image"} for _ in frames]),
                    {"type": "text", "text": sample['question']}
                ]
            }
        ]
        
        # Prepare inputs
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = processor(
            text=[text],
            images=frames,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,  # Greedy search for deterministic evaluation
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id
            )
        
        # Decode
        # We need to trim the input prompts from the output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.get("input_ids"), generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        print(f"Model Output: {output_text}")

if __name__ == "__main__":
    main()
