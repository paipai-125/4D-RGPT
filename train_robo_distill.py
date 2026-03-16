import os
os.environ['OMP_NUM_THREADS'] = '1'

import sys
import json
import cv2
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 开启 TF32 以加速
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
    # 只有 rank 0 打印错误，避免刷屏，但在 import 失败时所有进程都应该退出
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
        # resize 到 448 给 Student, TeacherWrapper 内部会再 resize 适配 L4P
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
    # DDP Initialization / 多卡初始化
    # ==========================
    # 检查是否是 DDP 运行环境 (通过 torchrun 启动会有这些环境变量)
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    ddp_enabled = local_rank != -1

    if ddp_enabled:
        # 初始化进程组，使用 nccl 后端（GPU 推荐）
        dist.init_process_group(backend="nccl")
        # 设置当前设备
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        # 单卡或非分布式环境
        device = "cuda" if torch.cuda.is_available() else "cpu"
        rank = 0
        world_size = 1
        print(f"Using device: {device} (Non-Distributed Mode)")

    # ==========================
    # Logger Setup
    # ==========================
    # 只有主进程 (Global Rank 0) 负责写日志
    if rank == 0:
        sys.stdout = Logger("output_dis.txt")
        print(f"Distributed Training: {ddp_enabled}")
        print(f"Global Rank: {rank}, World Size: {world_size}")
    else:
        sys.stdout = open(os.devnull, 'w')

    # Configuration
    local_model_path = "../4D-Data/models/Qwen3-VL-2B-Instruct"
    data_json_path = "../4D-Data/RoboFAC/training_qa.json"
    data_root = "../4D-Data/RoboFAC/simulation_data"
    num_frames = 16
    target_resolution = 448
    num_epochs = 5
    learning_rate = 2e-5
    
    # Distillation Params
    alpha = 0.5 # Latent Loss
    beta = 0.1  # Explicit Loss
    accumulation_steps = 32 # Gradient Accumulation Steps
    
    # ==========================
    # 1. Load Student Model & Processor
    # ==========================
    if rank == 0:
        print("Loading processor and student model...")
    
    try:
        processor = AutoProcessor.from_pretrained(local_model_path, trust_remote_code=True)
        # DDP 模式下，必须设置为 device_map=None，并手动 to(device)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            local_model_path,
            torch_dtype=torch.float32,
            device_map=None, 
            trust_remote_code=True
        )
        model.to(device)
    except Exception as e:
        if rank == 0: 
            print(f"Error loading student model: {e}")
        return
        
    if rank == 0:
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
        modules_to_save=["merger", "attn_pool"] 
    )
    
    # Auto-detect hidden size
    if hasattr(model.config, "text_config") and hasattr(model.config.text_config, "hidden_size"):
        student_hidden_dim = model.config.text_config.hidden_size
    elif hasattr(model.config, "hidden_size"):
        student_hidden_dim = model.config.hidden_size
    else:
        student_hidden_dim = 3136 # Fallback
        if rank == 0:
            print(f"Warning: Could not auto-detect hidden size, using default {student_hidden_dim}")
    
    model = get_peft_model(model, peft_config)
    
    # Wrap Student with DDP
    if ddp_enabled:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    
    # ==========================
    # 2. Initialize Teacher Model
    # ==========================
    if rank == 0:
        print("Initializing Teacher Model (L4P)...")
    
    # TeacherWrapper 内部可能使用 Fabric
    # 我们希望它在当前 device 上运行且不干扰 DDP
    try:
        # 注意: TeacherWrapper 内部初始化 Fabric 时需确保不会重新初始化 group
        # 如果 L4P 代码中 fabric 设置为 auto，可能会通过环境变量检测到
        teacher = TeacherWrapper(device=device) 
    except RuntimeError as e:
        # 如果 Fabric 报错，通常需要在 teacher wrapper 里调整初始化逻辑
        # 但如果是按我们修改过的 l4p/utils.py (devices=1)，配合 torchrun 应该正常
        raise e
        
    if rank == 0:
        print("Teacher Model initialized.")

    # ==========================
    # 3. Initialize D4DP (4D Perception Decoder)
    # ==========================
    if rank == 0:
        print("Initializing D4DP Decoder...")
        
    d4dp = D4DPerception(input_dim=student_hidden_dim, output_dim=1408)
    d4dp.to(device)
    # Convert to float32
    d4dp = d4dp.to(dtype=torch.float32, device=device)
    d4dp.train()
    
    # Wrap D4DP with DDP
    if ddp_enabled:
        d4dp = DDP(d4dp, device_ids=[local_rank])
    
    # ==========================
    # 4. Optimizer & Data
    # ==========================
    params_to_optimize = list(model.parameters()) + list(d4dp.parameters())
    optimizer = torch.optim.AdamW(params_to_optimize, lr=learning_rate)

    if rank == 0:
        print("Loading dataset...")
        
    all_valid_samples = []
    # 只有主进程读取文件没必要，为了简单可以都读，耗时不多
    with open(data_json_path, 'r') as f:
        full_data = json.load(f)

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
    
    # Shuffle (确保所有 rank 使用相同的种子，这样 shuffle 结果一致)
    import random
    random.seed(42)
    random.shuffle(all_valid_samples)
    
    # Debug: 只训练前10个数据
    # if rank == 0:
    #     print("Limiting to first 10 samples.")
    # all_valid_samples = all_valid_samples[:10]

    # 确保数据总量能被 world_size 整除，防止最后一个 epoch 因为步数对不齐而卡死
    if len(all_valid_samples) % world_size != 0:
        new_len = len(all_valid_samples) - (len(all_valid_samples) % world_size)
        if rank == 0:
            print(f"Trimming dataset from {len(all_valid_samples)} to {new_len} to fit uniformly across {world_size} GPUs.")
        all_valid_samples = all_valid_samples[:new_len]

    # Data Sharding (关键: 多机多卡/单机多卡 都通过 rank 和 world_size 分片)
    # 简单的分片策略：step = world_size
    # Rank 0: [0, 4, 8...]
    # Rank 1: [1, 5, 9...]
    # ...
    # 这样每个样本只被一个 GPU 处理
    training_samples = all_valid_samples[rank::world_size]
    
    if rank == 0:
        print(f"Total JSON entries: {len(full_data)}")
        print(f"Total Valid samples: {len(all_valid_samples)}")
        print(f"Samples per GPU (approx): {len(training_samples)}")
        print(f"Prepared {len(training_samples)} samples for this process (Rank {rank}).")
        print(f"Accumulation steps: {accumulation_steps}")

    # Training Setup
    num_training_steps = num_epochs * (len(training_samples) // accumulation_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps
    )
    
    model.train()
    optimizer.zero_grad() # Initialize gradients
    
    global_step_counter = 0

    # ==========================
    # 5. Training Loop
    # ==========================
    for epoch in range(num_epochs):
        if rank == 0:
            print(f"\n===== Epoch {epoch+1}/{num_epochs} =====")
        
        # 为了 DDP shuffle 更好，通常使用 DistributedSampler 并在每个 epoch set_epoch
        # 这里使用了简单的静态切片，每个 epoch 处理相同的数据子集
        
        epoch_loss_total = 0.0
        epoch_loss_sft = 0.0
        epoch_loss_distill = 0.0
        
        for i, sample in enumerate(training_samples):
            video_path = sample['video_path']
            question = sample['question']
            answer = sample['answer']
            
            # Load frames
            frames_pil = load_video_pil(video_path, num_frames=num_frames, target_resolution=target_resolution)
            if frames_pil is None: continue
                
            # --- Teacher Forward ---
            with torch.no_grad():
                f_4d_teacher, p_m_teacher = teacher.forward([frames_pil])
                
            # --- Student Input Prep ---
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
            
            inputs = processor(
                text=[text],
                images=frames_pil,
                return_tensors="pt",
                padding=True
            )
            
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
    # Create Labels
            labels = inputs["input_ids"].clone()
            
            # SFT Masking
            im_start_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
            if im_start_id is not None:
                for batch_idx in range(labels.shape[0]):
                    start_indices = (labels[batch_idx] == im_start_id).nonzero(as_tuple=True)[0]
                    if len(start_indices) > 0:
                        last_start_idx = start_indices[-1]
                        mask_len = last_start_idx + 2 
                        labels[batch_idx, :mask_len] = -100
            
            if processor.tokenizer.pad_token_id is not None:
                labels[labels == processor.tokenizer.pad_token_id] = -100
            
            inputs["labels"] = labels
            
            # --- Student Forward ---
            outputs = model(**inputs, output_hidden_states=True)
            loss_sft = outputs.loss
            
            # --- Distillation Logic ---
            last_hidden_state = outputs.hidden_states[-1] 
            
            vision_start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
            vision_end_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
            
            ids = inputs["input_ids"][0]
            locs_start = (ids == vision_start_id).nonzero(as_tuple=True)[0]
            locs_end = (ids == vision_end_id).nonzero(as_tuple=True)[0]
            
            loss_distill = torch.tensor(0.0, device=device)
            loss_ld = 0.0
            loss_ed = 0.0
            
            if len(locs_start) > 0 and len(locs_end) > 0:
                visual_tokens_list = []
                for s, e in zip(locs_start, locs_end):
                    segment = last_hidden_state[:, s+1 : e, :]
                    visual_tokens_list.append(segment)
                
                student_visual_hidden = torch.cat(visual_tokens_list, dim=1)
                
                grid_thw = inputs.get("image_grid_thw", None)
                
                try:
                    f_4d_student = d4dp(student_visual_hidden, grid_thw=grid_thw)
                    
                    p_m_student = teacher.decode_student_features(f_4d_student)

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
                    if rank == 0:
                        print(f"D4DP/Distill Error (Shape Mismatch?): {e}")
                    loss_distill = torch.tensor(0.0, device=device)
            
            # --- Total Loss & Backward ---
            total_loss = loss_sft + loss_distill
            
            # Gradient Accumulation
            loss = total_loss / accumulation_steps
            loss.backward()
            
            global_step_counter += 1

            if global_step_counter % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(params_to_optimize, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # Logging (Rank 0 only)
            if rank == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                print(f"Sample {i+1}/{len(training_samples)} | Loss Total: {total_loss.item():.4f} | SFT: {loss_sft.item():.4f} | Distill: {loss_distill.item():.4f} (LD: {loss_ld:.4f}, ED: {loss_ed:.4f} | LR: {current_lr:.2e})")

            epoch_loss_total += total_loss.item()
            epoch_loss_sft += loss_sft.item()
            epoch_loss_distill += loss_distill.item()
        
        # Calculate Average for this Process
        avg_total = epoch_loss_total / max(len(training_samples), 1)
        avg_sft = epoch_loss_sft / max(len(training_samples), 1)
        avg_distill = epoch_loss_distill / max(len(training_samples), 1)
        
        if rank == 0:
            print(f"Epoch {epoch+1} Avg Loss (Rank 0 view): {avg_total:.4f} | Avg SFT: {avg_sft:.4f} | Avg Distill: {avg_distill:.4f}")
            
            # Save Checkpoint (Only Rank 0 saves)
            # Make sure it's adjacent to 4D-RGPT in 4D-Data
            save_root = os.path.abspath(os.path.join(os.getcwd(), "..", "4D-Data", "checkpoints_distill"))
            save_dir = os.path.join(save_root, f"epoch_{epoch+1}")
            os.makedirs(save_dir, exist_ok=True)
            
            # Unwrap DDP
            model_to_save = model.module if ddp_enabled else model
            model_to_save.save_pretrained(save_dir)
            processor.save_pretrained(save_dir)
            
            d4dp_to_save = d4dp.module if ddp_enabled else d4dp
            torch.save(d4dp_to_save.state_dict(), os.path.join(save_dir, "d4dp.pt"))
            print(f"Saved checkpoint to {save_dir}")
            
    # Cleanup
    if ddp_enabled:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
