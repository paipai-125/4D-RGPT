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
from bert_score import score as bert_score
from rouge import Rouge

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

def compute_similarity(gt, pred):
    # BERTScore
    P, R, F1 = bert_score([pred], [gt], lang="zh", verbose=False)
    bert_f1 = F1[0].item()
    # ROUGE
    rouge = Rouge()
    scores = rouge.get_scores(pred, gt)
    rouge_l = scores[0]['rouge-l']['f']
    return bert_f1, rouge_l

def main():
    # ==========================
    # Configuration
    # ==========================
    base_model_path = "./src/models/Qwen3-VL-8B-Instruct"
    adapter_path = "../4D-Data/checkpoints_distill/epoch_5" # 指向第5轮权重
    data_root = "./RoboFAC/simulation_data"
    test_json_dir = "./RoboFAC/test_qa_sim"
    num_frames = 16
    target_resolution = 448
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ==========================
    # 1. Load Model & Adapter
    # ==========================
    print("Loading base model...")
    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading base model: {e}")
        return

    print(f"Loading LoRA adapter from {adapter_path}...")
    try:
        model = PeftModel.from_pretrained(model, adapter_path)
    except Exception as e:
        print(f"Error loading adapter: {e}")
        print("Note: Please ensure the 'epoch_5' checkpoint exists. Proceeding with base model only for demonstration if missing.")
    model.eval()

    # ==========================
    # 2. Load Test Data
    # ==========================
    print("Loading test set...")
    test_samples = []
    for fname in os.listdir(test_json_dir):
        if not fname.endswith(".json"): continue
        fpath = os.path.join(test_json_dir, fname)
        with open(fpath, 'r') as f:
            items = json.load(f)
            for item in items:
                vid_rel_path = item.get('video')
                if not vid_rel_path: continue
                vid_path = os.path.join(data_root, vid_rel_path)
                if os.path.exists(vid_path):
                    conversations = item.get('conversations', [])
                    if len(conversations) >= 2:
                        question = conversations[0]['value'].replace("<video>\n","").replace("<video>","")
                        answer = conversations[1]['value']
                        test_samples.append({
                            "video_path": vid_path,
                            "question": question,
                            "answer": answer
                        })
    print(f"Prepared {len(test_samples)} samples for inference.")

    # ==========================
    # 3. Inference Loop & Similarity
    # ==========================
    print("\nStarting Inference...")
    bert_scores = []
    rouge_scores = []
    for i, sample in enumerate(test_samples):
        video_path = sample['video_path']
        question = sample['question']
        ground_truth = sample['answer']
        frames_pil = load_video_pil(video_path, num_frames=num_frames, target_resolution=target_resolution)
        if frames_pil is None:
            print(f"Skipping {video_path} (Load failed)")
            continue
        messages = [
            {
                "role": "user",
                "content": [
                    *([{"type": "image"} for _ in frames_pil]),
                    {"type": "text", "text": question}
                ]
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text],
            images=frames_pil,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                temperature=0.0,
            )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        bert_f1, rouge_l = compute_similarity(ground_truth, output_text)
        bert_scores.append(bert_f1)
        rouge_scores.append(rouge_l)
        print(f"-"*60)
        print(f"Sample {i+1}:")
        print(f"Video: {video_path}")
        print(f"Q: {question}")
        print(f"A (GT)  : {ground_truth}")
        print(f"A (Pred): {output_text}")
        print(f"BERTScore-F1: {bert_f1:.4f} | ROUGE-L: {rouge_l:.4f}")
    if bert_scores:
        avg_bert = sum(bert_scores) / len(bert_scores)
        avg_rouge = sum(rouge_scores) / len(rouge_scores)
        print(f"\nAverage BERTScore-F1 over {len(bert_scores)} samples: {avg_bert:.4f}")
        print(f"Average ROUGE-L over {len(rouge_scores)} samples: {avg_rouge:.4f}")
    else:
        print("No valid samples for evaluation.")

if __name__ == "__main__":
    main()
