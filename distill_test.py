import os
# 在导入任何可能使用OpenMP的库之前设置环境变量
os.environ['OMP_NUM_THREADS'] = '4'

import sys
import json
import cv2
import torch
import torch.distributed as dist
import numpy as np
import random
from PIL import Image
from transformers import AutoProcessor
from peft import PeftModel
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
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

def calculate_metrics(gt_answers, pred_answers):
    bleu_scores = []
    rouge = Rouge()
    rouge_scores = []
    
    # Use SmoothingFunction for BLEU to handle short sequences better
    smoothing = SmoothingFunction().method1

    for gt, pred in zip(gt_answers, pred_answers):
        # Handle empty prediction to avoid errors
        if not pred or not pred.strip():
            pred = " " # Empty string workaround
            
        # BLEU Score
        reference = [gt.split()]
        candidate = pred.split()
        if not candidate:
            candidate = [""]
            
        # Smoothing function can be added if needed, but default is fine
        bleu = sentence_bleu(reference, candidate, smoothing_function=smoothing)
        bleu_scores.append(bleu)

        # ROUGE Score
        try:
            rouge_score = rouge.get_scores(pred, gt)[0]
        except Exception:
            rouge_score = {
                'rouge-1': {'f': 0.0},
                'rouge-2': {'f': 0.0},
                'rouge-l': {'f': 0.0}
            }
        rouge_scores.append(rouge_score)

    avg_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
    if rouge_scores:
        avg_rouge = {
            "rouge-1": np.mean([score['rouge-1']['f'] for score in rouge_scores]),
            "rouge-2": np.mean([score['rouge-2']['f'] for score in rouge_scores]),
            "rouge-l": np.mean([score['rouge-l']['f'] for score in rouge_scores]),
        }
    else:
        avg_rouge = {"rouge-1": 0.0, "rouge-2": 0.0, "rouge-l": 0.0}

    return avg_bleu, avg_rouge

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
    adapter_path = "../4D-Data/checkpoints_distill/epoch_1"  #修改epoch
    data_root = "../4D-Data/RoboFAC/simulation_data"
    test_qa_path = "../4D-Data/RoboFAC/test_qa_sim"
    
    num_frames = 16
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
    for file_name in os.listdir(test_qa_path):
        file_path = os.path.join(test_qa_path, file_name)
        if not file_name.endswith('.json'):
            continue

        with open(file_path, 'r') as f:
            full_data = json.load(f)

        if not isinstance(full_data, dict):
            continue

        for key, item in full_data.items():
            if not isinstance(item, dict):
                continue

            vid_rel_path = item.get('video')
            if not vid_rel_path:
                continue

            vid_path = os.path.join(data_root, vid_rel_path)
            
            # Support both 'conversations' (legacy) and 'annos' (new format)
            conversations_list = []
            
            if 'conversations' in item:
                 conversations_list.append(item['conversations'])
            
            if 'annos' in item and isinstance(item['annos'], dict):
                 for k, v in item['annos'].items():
                      conversations_list.append(v)
            
            for conversations in conversations_list:
                if len(conversations) < 2:
                    continue

                q_text = conversations[0]['value']
                # Clean up <image> and <video> tags
                q_text = q_text.replace("<video>\n", "").replace("<video>", "")
                q_text = q_text.replace("<image>\n", "").replace("<image>", "")
                
                question = q_text
                answer = conversations[1]['value']
                all_valid_samples.append({
                    "video_path": vid_path,
                    "question": question,
                    "answer": answer
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
        gt_answers.append(ground_truth)
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
            avg_bleu, avg_rouge = calculate_metrics(final_gt, final_pred)
            
            # Print to stdout
            print(f"Total Samples Evaluated: {len(final_gt)}")
            print(f"Average BLEU Score: {avg_bleu:.4f}")
            print("Average ROUGE Scores:")
            print(f"  ROUGE-1: {avg_rouge['rouge-1']:.4f}")
            print(f"  ROUGE-2: {avg_rouge['rouge-2']:.4f}")
            print(f"  ROUGE-L: {avg_rouge['rouge-l']:.4f}")
            
            # Save to file
            try:
                with open("./infer_output.txt", "w", encoding='utf-8') as f:
                    f.write(f"Total Samples Evaluated: {len(final_gt)}\n")
                    f.write(f"Average BLEU Score: {avg_bleu:.4f}\n")
                    f.write("Average ROUGE Scores:\n")
                    f.write(f"  ROUGE-1: {avg_rouge['rouge-1']:.4f}\n")
                    f.write(f"  ROUGE-2: {avg_rouge['rouge-2']:.4f}\n")
                    f.write(f"  ROUGE-L: {avg_rouge['rouge-l']:.4f}\n")
                print("Results saved to ./infer_output.txt")
            except Exception as e:
                print(f"Error saving results to file: {e}")
        else:
            print("No samples evaluated.")
            
    if ddp_enabled:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
