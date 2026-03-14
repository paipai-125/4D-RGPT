import os
# 在导入任何可能使用OpenMP的库之前设置环境变量
os.environ['OMP_NUM_THREADS'] = '4'

import sys
import json
import cv2
import torch
import numpy as np
import random
from PIL import Image
from transformers import AutoProcessor
from peft import PeftModel

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

# Load Video Function (matches training logic)
def load_video_pil(video_path, num_frames=16, target_resolution=448):
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
    # ==========================
    # Configuration
    # ==========================
    base_model_path = "../4D-Data/models/Qwen3-VL-2B-Instruct"
    adapter_path = "../4D-Data/checkpoints/epoch_5" # 指向第5轮权重
    data_root = "../4D-Data/RoboFAC/simulation_data"
    data_json_path = "../4D-Data/RoboFAC/training_qa.json"
    
    num_frames = 16
    target_resolution = 448
    limit_samples = 10
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ==========================
    # 1. Load Model & Adapter
    # ==========================
    print("Loading base model...")
    try:
        # Load Base Model
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32, # 推理通常使用 fp16
            device_map="auto",
            trust_remote_code=True
        )
        processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading base model: {e}")
        return

    print(f"Loading LoRA adapter from {adapter_path}...")
    try:
        # Load LoRA Adapter
        model = PeftModel.from_pretrained(model, adapter_path)
    except Exception as e:
        print(f"Error loading adapter: {e}")
        print("Note: Please ensure the 'epoch_10' checkpoint exists. Proceeding with base model only for demonstration if missing.")
    
    model.eval()

    # ==========================
    # 2. Load Data (Same subset logic)
    # ==========================
    print("Loading dataset...")
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
    
    # Random Select (Same seed as training to ensure same data)
    random.seed(42)
    if len(all_valid_samples) > limit_samples:
        test_samples = random.sample(all_valid_samples, limit_samples)
    else:
        test_samples = all_valid_samples
    print(f"Prepared {len(test_samples)} samples for inference.")

    # ==========================
    # 3. Inference Loop
    # ==========================
    print("\nStarting Inference...")
    
    for i, sample in enumerate(test_samples):
        video_path = sample['video_path']
        question = sample['question']
        ground_truth = sample['answer']
        
        frames_pil = load_video_pil(video_path, num_frames=num_frames, target_resolution=target_resolution)
        if frames_pil is None:
            print(f"Skipping {video_path} (Load failed)")
            continue

        # Prepare Chat Format
        messages = [
            {
                "role": "user",
                "content": [
                    *([{"type": "image"} for _ in frames_pil]),
                    {"type": "text", "text": question}
                ]
            }
        ]
        
        # Prepare Inputs
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text],
            images=frames_pil,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False, # Deterministic (Greedy Search)
                temperature=0.0,  
            )
            
        # Decode
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        # Print Result
        print(f"-"*60)
        print(f"Sample {i+1}:")
        print(f"Video: {video_path}")
        print(f"Q: {question}")
        print(f"A (GT)  : {ground_truth}")
        print(f"A (Pred): {output_text}")

if __name__ == "__main__":
    main()
