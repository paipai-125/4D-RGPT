import os
# 在导入任何可能使用OpenMP的库之前设置环境变量
os.environ['OMP_NUM_THREADS'] = '4'

import sys
import json
import re
import cv2
import torch
import torch.distributed as dist
import numpy as np
import random
from PIL import Image
from transformers import AutoProcessor
from peft import PeftModel
from tqdm import tqdm

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

def parse_float(text):
    if not text:
        return None
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    if matches:
        try:
            return float(matches[0])
        except:
            return None
    return None

def calculate_vsti_metrics(all_samples, pred_answers):
    # all_samples: full list of sample dicts (ground truths)
    # pred_answers: flat list of predictions
    
    mc_correct = 0
    mc_total = 0
    
    numeric_accuracies = []
    
    for sample, pred in zip(all_samples, pred_answers):
        q_type = sample.get('type', 'generic')
        
        if q_type == 'mc':
            gt = sample['answer'] # Should be "A", "B", etc.
            if not gt: continue
            
            # Simple check: Does normalized pred start with gt?
            pred_clean = pred.strip()
            
            # Extract first letter A-D
            match = re.search(r'\b([A-D])\b', pred_clean, re.IGNORECASE)
            pred_letter = match.group(1).upper() if match else ""
            
            if pred_letter == gt:
                mc_correct += 1
            mc_total += 1
            
        elif q_type == 'numeric':
            try:
                gt_val = float(sample['answer'])
                pred_val = parse_float(pred)
                
                if pred_val is not None:
                    if gt_val == 0:
                        rel_acc = 0.0 # Define behavior for gt=0?
                    else:
                        rel_err = abs(pred_val - gt_val) / abs(gt_val)
                        rel_acc = max(0.0, 1.0 - rel_err)
                    numeric_accuracies.append(rel_acc)
                else:
                    numeric_accuracies.append(0.0)
            except ValueError:
                pass # GT wasn't a number
                
    mc_acc = mc_correct / mc_total if mc_total > 0 else 0.0
    avg_num_acc = np.mean(numeric_accuracies) if numeric_accuracies else 0.0
    
    return mc_acc, avg_num_acc, mc_total, len(numeric_accuracies)

def main():
    # ==========================
    # DDP Initialization / 多卡初始化
    # ==========================
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    ddp_enabled = local_rank != -1

    if ddp_enabled:
        dist.init_process_group(backend="nccl")
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        rank = 0
        world_size = 1
        
    # Only rank 0 prints
    def log_print(*args, **kwargs):
        if rank == 0:
            print(*args, **kwargs)

    if rank == 0:
        log_print(f"Distributed testing enabled: {ddp_enabled}")
        log_print(f"Global Rank: {rank}, World Size: {world_size}")
        log_print(f"Using device: {device}")

    # ==========================
    # Configuration
    # ==========================
    base_model_path = "../4D-Data/models/Qwen3-VL-2B-Instruct"
    adapter_path = "../4D-Data/checkpoints_distill/epoch_1"
    data_root = "../4D-Data/VSTI-Bench"
    test_qa_path = "../4D-Data/VSTI-Bench/test.json"
    
    num_frames = 32
    target_resolution = 448
    
    # ==========================
    # 1. Load Model & Adapter
    # ==========================
    # log_print("Loading base model...")
    try:
        # Load Base Model
        # DDP/Multi-process: Manually load to device to avoid auto-map issues
        if ddp_enabled:
             model = Qwen3VLForConditionalGeneration.from_pretrained(
                base_model_path,
                torch_dtype=torch.float16 if str(device).startswith("cuda") else torch.float32, 
                device_map=None,
                trust_remote_code=True
            )
             model.to(device)
        else:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                base_model_path,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32, 
                device_map="auto",
                trust_remote_code=True
            )
            
        processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    except Exception as e:
        log_print(f"Error loading base model: {e}")
        return

    log_print(f"Loading LoRA adapter from {adapter_path}...")
    try:
        # Load LoRA Adapter
        model = PeftModel.from_pretrained(model, adapter_path)
    except Exception as e:
        log_print(f"Error loading adapter: {e}")
    
    model.eval()

    # ==========================
    # 2. Load Data (Full Test Set)
    # ==========================
    log_print("Loading dataset...")
    all_valid_samples = []

    # All ranks load file structure (fast enough)
    target_files = []
    if os.path.isfile(test_qa_path):
        target_files = [test_qa_path]
    
    for file_path in target_files:
        with open(file_path, 'r') as f:
            full_data = json.load(f)

        # Handle VSTI-Bench format (list of dicts)
        if isinstance(full_data, list):
            for item in full_data:
                # Video path handling
                vid_rel_path = item.get('video_path')
                if not vid_rel_path:
                    continue
                
                vid_path = os.path.join(data_root, vid_rel_path)
                
                # Question & Answer handling
                question = item.get('question', '')
                
                options = item.get('options')
                mc_answer = item.get('mc_answer')
                
                if options:
                    q_type = 'mc'
                    answer = mc_answer
                    # Format Question
                    formatted_q = question + "\n"
                    for opt in options:
                        formatted_q += f"{opt}\n"
                    formatted_q += "Answer:"
                    question = formatted_q
                elif mc_answer is None and options is None:
                    q_type = 'numeric'
                    answer = item.get('ground_truth', '')
                else:
                    q_type = 'generic'
                    answer = item.get('ground_truth', '')
                
                # Clean up <image> and <video> tags if present in question
                question = question.replace("<video>\n", "").replace("<video>", "")
                question = question.replace("<image>\n", "").replace("<image>", "")
                
                all_valid_samples.append({
                    "video_path": vid_path,
                    "question": question,
                    "answer": answer,
                    "type": q_type,
                    "original_gt": item.get('ground_truth')
                })

    # Sort to ensure consistent order across all processes before sharding
    all_valid_samples.sort(key=lambda x: x['video_path'] + x['question'])

    # Only test the first 10 samples
    # all_valid_samples = all_valid_samples[:10]

    log_print(f"Total samples: {len(all_valid_samples)}")

    # Shard Data
    my_samples = all_valid_samples[rank::world_size]
    # log_print(f"Rank {rank} processing {len(my_samples)} samples")

    # ==========================
    # 3. Inference Loop
    # ==========================
    log_print("\nStarting Inference...")
    gt_answers = []
    pred_answers = []

    # Use tqdm on all ranks
    # position=rank helps to prevent overlap in some terminals, but might still be messy
    iterator = tqdm(my_samples, desc=f"Rank {rank}", position=rank)

    for sample in iterator:
        video_path = sample['video_path']
        question = sample['question']
        ground_truth = sample['answer']
        
        frames_pil = load_video_pil(video_path, num_frames=num_frames, target_resolution=target_resolution)
        if frames_pil is None:
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
                do_sample=False, 
                temperature=0.0,  
            )
            
        # Decode
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        
        # Collect Results
        gt_answers.append(sample)
        pred_answers.append(output_text)

    # ==========================
    # 4. Gather Results & Metrics
    # ==========================
    if ddp_enabled:
        # Gather results from all ranks
        all_gt_lists = [None for _ in range(world_size)]
        all_pred_lists = [None for _ in range(world_size)]
        
        dist.all_gather_object(all_gt_lists, gt_answers)
        dist.all_gather_object(all_pred_lists, pred_answers)
        
        if rank == 0:
            final_gt = []
            final_pred = []
            for gts in all_gt_lists:
                if gts:
                    final_gt.extend(gts)
            for preds in all_pred_lists:
                if preds:
                    final_pred.extend(preds)
        else:
            final_gt = []
            final_pred = []
    else:
        final_gt = gt_answers
        final_pred = pred_answers

    if rank == 0:
        print("\nCalculating Metrics...")
        if len(final_gt) > 0:
            mc_acc, num_acc, mc_total, num_total = calculate_vsti_metrics(final_gt, final_pred)
            
            # Print to stdout
            print(f"Total Samples Evaluated: {len(final_gt)}")
            print(f"MC Accuracy: {mc_acc:.4f} ({mc_total} samples)")
            print(f"Numeric Relative Accuracy: {num_acc:.4f} ({num_total} samples)")
            
            # Save to file
            try:
                with open("./infer_output.txt", "w", encoding='utf-8') as f:
                    f.write(f"Total Samples Evaluated: {len(final_gt)}\n")
                    f.write(f"MC Accuracy: {mc_acc:.4f} ({mc_total} samples)\n")
                    f.write(f"Numeric Relative Accuracy: {num_acc:.4f} ({num_total} samples)\n")
                print("Results saved to ./infer_output.txt")
            except Exception as e:
                print(f"Error saving results to file: {e}")
        else:
            print("No samples evaluated.")
            
    if ddp_enabled:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
