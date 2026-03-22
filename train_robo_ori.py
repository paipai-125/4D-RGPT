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
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from transformers import AutoProcessor, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

# 添加 L4P 路径以供后续导入
sys.path.append(os.path.abspath("./L4P-main"))
# 添加 Orient-Anything-V2 路径
sys.path.append(os.path.abspath("./Orient-Anything-V2"))
sys.path.append(os.path.abspath(".")) # 当前目录

# 尝试导入自定义的蒸馏工具
try:
    from distill_utils import TeacherWrapper, D4DPerception, compute_distillation_loss
except ImportError as e:
    # 只有 rank 0 打印错误，避免刷屏，但在 import 失败时所有进程都应该退出
    print(f"Error importing distill_utils: {e}")
    sys.exit(1)

# Orient-Anything-V2 Imports
try: 
    from vision_tower import VGGT_OriAny_Ref
    from utils.app_utils import preprocess_images
except ImportError as e:
    print(f"Error importing Orient-Anything-V2 modules: {e}")
    print("Make sure 'Orient-Anything-V2' is in the path and requirements are installed.")
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rank = 0
        local_rank = 0  # Default to 0 for single GPU
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
    lambda_ori = 0.2 # Orient Loss Weight
    accumulation_steps = 32 # Gradient Accumulation Steps
    
    # Orient-Anything Checkpoint
    ori_ckpt_path = "../4D-Data/OriAnyV2_ckpt/rotmod_realrotaug_best.pt"
    
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
    
    # Get hidden size
    if hasattr(model.config, "hidden_size"):
        student_hidden_dim = model.config.hidden_size
    elif hasattr(model.config, "text_config"):
        student_hidden_dim = model.config.text_config.hidden_size
    else:
        # Fallback for 2B model (2048) or similar
        student_hidden_dim = 2048 
        if rank == 0:
            print(f"Warning: Could not detect hidden_size, using default {student_hidden_dim}")

    # --- Pose Token Setup (Student Side) ---
    # 定义可学习的 pose tokens: [1, 2, 1, hidden_dim]
    # Index 0: Reference pose token
    # Index 1: Target pose token
    pose_tokens = nn.Parameter(torch.randn(1, 2, 1, student_hidden_dim).to(device))
    
    # 定义 Projection Layer: Student Dim -> Teacher Dim (OriAnyV2 uses 1024 / 900 dims normally, check below)
    # 假设 Teacher 输出 hidden state (1024-d, DINOv2 style) 或者 logits (900-d)
    # 我们先设置为 Teacher 的 hidden dim，稍后加载 Teacher 时会确认维度
    # 先暂定 1024，后面初始化 teacher 后如果有变可以改
    TEACHER_DIM = 900 # 根据 demo_inference, out_dim=900
    pose_proj = nn.Linear(student_hidden_dim, TEACHER_DIM).to(device)

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
    
    model = get_peft_model(model, peft_config)
    
    # Wrap Student with DDP
    if ddp_enabled:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    
    # ==========================
    # 2. Initialize Teacher Model (L4P)
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
    # 2.5 Initialize Teacher 2 (Orient-Anything-V2)
    # ==========================
    if rank == 0:
        print("Initializing Teacher Model (OriAnyV2)...")
    
    try:
        # 1. Init
        dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.get_device_capability(local_rank)[0] >= 8 else torch.float16
        # out_dim=900, nopretrain=True
        teacher_ori = VGGT_OriAny_Ref(out_dim=900, dtype=dtype, nopretrain=True)
        
        # 2. Load Checkpoint
        if os.path.exists(ori_ckpt_path):
             state_dict = torch.load(ori_ckpt_path, map_location='cpu')
             teacher_ori.load_state_dict(state_dict)
             if rank == 0:
                 print(f"Loaded OriAnyV2 checkpoint from local: {ori_ckpt_path}")
        else:
             # Try HF Hub
             from huggingface_hub import hf_hub_download
             if rank == 0: print(f"Downloading checkpoint from HuggingFace...")
             try:
                 ckpt_path = hf_hub_download(repo_id="Viglong/OriAnyV2_ckpt", filename="rotmod_realrotaug_best.pt", cache_dir='./')
                 state_dict = torch.load(ckpt_path, map_location='cpu')
                 teacher_ori.load_state_dict(state_dict)
             except Exception as e:
                 print(f"Failed to download/load checkpoint: {e}")
                 sys.exit(1)
        
        teacher_ori.eval()
        teacher_ori = teacher_ori.to(device)
        
        # Teacher Dim Check (Optional)
        # 这里 teacher 输出的是 900 维的 classification logits
        if rank == 0: print(f"Teacher OriAnyV2 initialized. Target Dim: 900")
        
    except Exception as e:
        print(f"Error initializing OriAnyV2: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

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
    # 将 pose_tokens 和 pose_proj 也加入优化器
    params_to_optimize = list(model.parameters()) + list(d4dp.parameters()) + [pose_tokens] + list(pose_proj.parameters())
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
        epoch_loss_ori = 0.0

        for i, sample in enumerate(training_samples):
            video_path = sample['video_path']
            question = sample['question']
            answer = sample['answer']
            
            # Load frames
            frames_pil = load_video_pil(video_path, num_frames=num_frames, target_resolution=target_resolution)
            if frames_pil is None: continue
                
            # --- Teacher 1 Forward ---
            with torch.no_grad():
                f_4d_teacher, p_m_teacher = teacher.forward([frames_pil])


            # --- Teacher 2 Forward (OriAnyV2) ---
            with torch.no_grad():
                # 修改后的逻辑：分别预测16帧中每一帧的3D朝向 (Absolute Pose Estimation)
                # 而不是 Ref-Target 模式
                
                # 1. Preprocess Images (PIL -> Tensor [S, 3, H, W])
                try:
                    # mode="pad" 是 demo里的默认行为
                    # preprocess_images 返回的是 [S, 3, H, W]
                    ts_frames = preprocess_images(frames_pil, mode="pad").to(device).to(teacher_ori.dtype)
                    
                    # 扩展维度为 Batch: [1, S, 3, H, W] -> [S, 1, 3, H, W]
                    # 这样每一帧都被当作一个独立的 Reference Frame 处理 (Batch Size = S, Seq Len = 1)
                    # Vision Tower 内部 forward 会进入 S=1 的分支，调用 ref_sampler 得到绝对姿态
                    ts_frames = ts_frames.unsqueeze(1)
                    
                    # Forward
                    # teacher_ori_pose output shape: [S, 900] (这里 S=NumFrames, 900=OutputDim)
                    teacher_ori_pose = teacher_ori(ts_frames)
                    
                    # 为了兼容后续 loss 计算逻辑 (期望 [1, S, 900])，我们在第0维加一个 batch 维度
                    teacher_ori_pose = teacher_ori_pose.unsqueeze(0) # [1, S, 900]
                    
                    # 转为 float32 用于后续 loss 计算
                    teacher_pose_targets = teacher_ori_pose.float() 
                    
                except Exception as e_ori:
                    print(f"Error in OriAnyV2 forward: {e_ori}")
                    teacher_pose_targets = None

                
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
            
            # --- Student Pose Token Insertion Logic ---
            # 我们需要把 pose_tokens 插入到 input_embeddings 里
            # 插入位置：每个 <|vision_end|> token 之后
            
            input_ids = inputs["input_ids"] # [B, L]
            
            # --- DEBUG BLOCK ---
            # print(f"DEBUG: model type: {type(model)}")
            # if hasattr(model, "module"):
            #     print(f"DEBUG: model.module type: {type(model.module)}")
            # print(f"DEBUG: model.model type (if exists): {type(model.model) if hasattr(model, 'model') else 'N/A'}")
            # -------------------

            # Unrap model for access
            base_model = model.module if hasattr(model, "module") else model
            # specific fetch for Qwen2VL
            if hasattr(base_model, "model"):
                llm_model = base_model.model 
                # If PeftModel, base_model.model might still be PeftModel's attribute if it exists, or the underlying model
                if hasattr(llm_model, "model"): # Double wrapping?
                     llm_model = llm_model.model
            else:
                llm_model = base_model
            
            # 1. Get Base Embeddings
            # text 部分
            if hasattr(llm_model, "embed_tokens"):
                inputs_embeds = llm_model.embed_tokens(input_ids)
            elif hasattr(llm_model, "wte"):
                inputs_embeds = llm_model.wte(input_ids)
            elif hasattr(llm_model, "get_input_embeddings"):
                inputs_embeds = llm_model.get_input_embeddings()(input_ids)
            elif hasattr(base_model, "get_input_embeddings"):
                inputs_embeds = base_model.get_input_embeddings()(input_ids)
            else:
                raise AttributeError(f"Could not find embedding layer in {type(llm_model)}")

            # 2. Vision Embeddings
            # 这一步通常发生在 model.forward 内部，但我们需要手动做
            pixel_values = inputs["pixel_values"]
            grid_thw = inputs["image_grid_thw"]
            
            # Qwen3VL 的 visual encoder 输出
            # 注意：新版本 transformers 可能把 `visual` 放在 `model.visual`
            # 并且不同版本 forward 接口不同 (pixel_values + grid_thw)
            
            # 获取 visual_embeds
            # Qwen2VL/3VL 的 visual forward 比较复杂，通常是:
            # embeds = model.visual(pixel_values, grid_thw=grid_thw)
            # 再经过 projector
            
            # 为了避免手写复杂的 merge 逻辑，我们利用一个 Trick:
            # 我们先让 model 跑一遍只为了拿到 `image_embeds` 或者 `merged_input_embeds`
            # 但是 transformers 库很难 hook 中间结果。
            
            # Alternative: 修改 Input_Ids 插入占位符，然后替换 Embedding
            # 我们假设 <|vision_end|> 后面可以插入一个特殊 token，然后在 embedding 层替换
            
            # 查找 <|vision_end|> ID
            vision_end_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
            
            # 创建新的 input_ids，在每个 vision_end 后面插入一个占位符 (使用 pad_token_id 或者其他 unused)
            PAD = processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else 0
            
            batch_size, seq_len = input_ids.shape
            
            new_input_ids_list = []
            new_labels_list = []
            pose_token_indices = [] # 记录 pose token 在新序列中的位置
            
            for b in range(batch_size):
                curr_ids = input_ids[b]
                curr_labels = labels[b]
                
                # 找到 vision_end 的位置
                end_indices = (curr_ids == vision_end_id).nonzero(as_tuple=True)[0]
                
                # 构建新序列
                new_ids = []
                new_l = []
                last_pos = 0
                curr_pose_indices = []
                
                for k, idx in enumerate(end_indices):
                    # 复制直到 vision_end (包含)
                    valid_len = (idx + 1) - last_pos
                    new_ids.append(curr_ids[last_pos : idx+1])
                    new_l.append(curr_labels[last_pos : idx+1])
                    
                    # 插入 Pose Token 占位符 (用 PAD 代替，Embedding 层会替换)
                    new_ids.append(torch.tensor([PAD], device=device))
                    new_l.append(torch.tensor([-100], device=device)) # Label -100
                    
                    # 记录这个占位符在新序列中的索引
                    # 当前长度 = sum(len of parts)
                    current_len = sum(len(x) for x in new_ids)
                    curr_pose_indices.append(current_len - 1)
                    
                    last_pos = idx + 1
                    
                # 复制剩余部分
                if last_pos < len(curr_ids):
                    new_ids.append(curr_ids[last_pos:])
                    new_l.append(curr_labels[last_pos:])
                    
                # Concat
                combined_ids = torch.cat(new_ids)
                combined_labels = torch.cat(new_l)
                
                new_input_ids_list.append(combined_ids)
                new_labels_list.append(combined_labels)
                pose_token_indices.append(curr_pose_indices)
                
            # Pad to max length in batch
            max_len = max(len(x) for x in new_input_ids_list)
            padded_ids = torch.full((batch_size, max_len), PAD, device=device, dtype=input_ids.dtype)
            padded_labels = torch.full((batch_size, max_len), -100, device=device, dtype=labels.dtype)
            
            # Attention Mask
            attention_mask = torch.zeros((batch_size, max_len), device=device, dtype=torch.long)
            
            for b in range(batch_size):
                l = len(new_input_ids_list[b])
                padded_ids[b, :l] = new_input_ids_list[b]
                padded_labels[b, :l] = new_labels_list[b]
                attention_mask[b, :l] = 1 # 简单 mask, Qwen 会自己处理 4D Mask
            
            # 生成 Base Input Embeddings (包含 Image 替换逻辑)
            # 这一步调用 Qwen3 forward 的一部分逻辑：
            # 由于 inputs_embeds 参数互斥，如果你传了 inputs_embeds，Qwen 不会处理 pixel_values
            # 所以我们必须手动做 "Image -> Embedding" 和 "Merge"
            
            # --- Manual Embedding Generation ---
            # 1. Text Embeds
            # Robustly fetch embedding layer
            base_model = model.module if hasattr(model, "module") else model
            llm_model = base_model.model if hasattr(base_model, "model") else base_model
            if hasattr(llm_model, "model"): llm_model = llm_model.model

            if hasattr(llm_model, "embed_tokens"):
                input_embeds = llm_model.embed_tokens(padded_ids)
            elif hasattr(llm_model, "wte"):
                input_embeds = llm_model.wte(padded_ids)
            elif hasattr(llm_model, "get_input_embeddings"):
                input_embeds = llm_model.get_input_embeddings()(padded_ids)
            elif hasattr(base_model, "get_input_embeddings"):
                input_embeds = base_model.get_input_embeddings()(padded_ids)
            else:
                # If still failing, try accessing via config or print dir
                raise AttributeError(f"Could not find embedding layer in {type(llm_model)}")
            
            # 2. Vision Embeds & Replacement
            # 获取 visual_embeds (合并 grid_thw)
            if hasattr(llm_model, "visual"):
                visual_module = llm_model.visual
            elif hasattr(base_model, "visual"):
                visual_module = base_model.visual
            elif hasattr(base_model, "model") and hasattr(base_model.model, "visual"):
                visual_module = base_model.model.visual
            else:
                 raise AttributeError(f"Could not find visual module in {type(base_model)} or {type(llm_model)}")

            visual_embeds = visual_module(pixel_values, grid_thw=grid_thw)
            
            # Robustly extract Tensor from output (Tuple, List, ModelOutput)
            # 1. Handle ModelOutput (has last_hidden_state)
            if hasattr(visual_embeds, "last_hidden_state"):
                visual_embeds = visual_embeds.last_hidden_state
            
            # 2. Handle Tuple/List (recursive unwrap to find first Tensor)
            # Qwen2VL might return (hidden_states, pooled_output) -> take [0]
            while isinstance(visual_embeds, (tuple, list)):
                if len(visual_embeds) > 0:
                    visual_embeds = visual_embeds[0]
                else:
                    break # Empty tuple/list?
            
            # Final check
            if not isinstance(visual_embeds, torch.Tensor):
                 if rank == 0:
                     print(f"Warning: visual_embeds is {type(visual_embeds)}, expected Tensor. Trying to cast or proceed.")

            # Apply merger if dimensions mismatch (e.g. 1024 -> 2048)
            if visual_embeds.shape[-1] != student_hidden_dim:
                merger = None
                if hasattr(visual_module, "merger"):
                    merger = visual_module.merger
                elif hasattr(llm_model, "visual") and hasattr(llm_model.visual, "merger"):
                    merger = llm_model.visual.merger
                elif hasattr(base_model, "visual") and hasattr(base_model.visual, "merger"):
                    merger = base_model.visual.merger
                elif hasattr(model, "visual") and hasattr(model.visual, "merger"):
                    merger = model.visual.merger

                if merger is not None:
                     visual_embeds = merger(visual_embeds)
                elif rank == 0:
                    print(f"Warning: Visual embeds dim {visual_embeds.shape[-1]} != student dim {student_hidden_dim}, but no merger found.")

            
            # 3. 把 visual embeds 填回 input_embeds
            # Qwen2VL/3VL 逻辑：input_ids 里有一些特殊 token (vision_start...vision_end 之间的占位符)
            # 它是直接把 visual_embeds 放到那些位置
            # 这部分逻辑在 modeling_qwen2_vl.py 里比较复杂。
            # 为了复用，我们其实可以先跑一次 model() 得到 hidden_states[0] (embedding output) 吗？不行，forward 不返回 embedding 层输出。
            
            # 妥协方案：调用 model 的私有方法 _merge_input_ids_with_image_features (如果有)
            # 或者复刻 Qwen2VL 的 merge 逻辑
            
            # 简化版 Merge 逻辑 (假设 Qwen 代码结构):
            # Qwen 处理逻辑: 找到 <|vision_start|> 和 <|vision_end|> 之间的区域，替换为 visual_embeds
            # 并且会去除占位符，导致序列变短。
            # 这使得我们在 input_ids 里插的 pose token 索引失效。
            
            # --- 修正策略 ---
            # 直接使用 model(input_ids=padded_ids, pixel_values=..., labels=...)
            # 然后在 input_ids 里我们已经通过上面的循环，在 vision_end 后面加了 PAD
            # 现在我们需要 hook model 的 embedding layer? 不行 DDP 很难 hook
            
            # 最佳方案: 使用 inputs_embeds 参数
            # 我们需要自己实现 merge。
            
            # 由于手写 merge 风险大，我们尝试利用 model 自身的 forward。
            # 只有当 inputs_embeds 不为 None 时，model 才会跳过 visual 处理。
            
            # 让我们尝试一种 "Post-Merge Injection" 
            # 1. 正常 forward 拿到 embedding 层的输出？做不到。
            # 2. 正常 forward，但在 inputs_embeds 进入 layer 0 之前修改。
            
            # 让我们回到最原始的 "Hack"：
            # 我们手动构建 inputs_embeds。
            # 1. 拿到 text embeddings
            # 2. 拿到 visual embeddings
            # 3. 按照 input_ids 拼起来。
            
            # Qwen3 的 visual embeddings 是经过 Patch Merging 的，长度变短了 (H/2 * W/2)。
            # input_ids 里的 visual placeholder 长度是 H*W 还是已经缩短了？
            # 实际上 processor 处理后的 input_ids 里的 placeholder 长度已经匹配 visual feature 长度了。
            
            # 所以我们可以相对放心地替换。
            
            # --- 实现 Manual Merge ---
            vision_start_token_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
            input_embeds_final = input_embeds.clone()
            
            # 需要用到 model.model.rotary_pos_emb 计算位置编码吗？不用，这里是 input embedding
            
            # 替换 Image Features
            # 找到所有 vision_start
            image_idx = 0
            
            # visual_embeds shape: [Total_Patches, D] (flatted batch)
            # 我们需要 split back to batch? 
            # Qwen2VL visual output is List of tensors? No, usually concatenated.
            # 需要根据 grid_thw 拆分
            
            # 假设 visual_embeds 是 [Total_Patches, D]
            current_patch_idx = 0
            
            for b in range(batch_size):
                # 找到当前 batch 的 visual placeholders
                # input_ids: ... <|vision_start|> [PAD] [PAD] ... <|vision_end|> ...
                # 我们寻找 vision_start
                start_locs = (padded_ids[b] == vision_start_token_id).nonzero(as_tuple=True)[0]
                
                # 每一帧
                for frame_idx in range(len(start_locs)):
                    # grid thw: [B*N, 3] (T, H, W)
                    # 对应的 THW
                    # 注意 grid_thw 是 flatten 的
                    # 全局 visual 索引
                    
                    # 计算该帧的 patch 数量 (Merge 后)
                    # Qwen2VL merge: usually (H//2)*(W//2)
                    t, h, w = grid_thw[image_idx] # image_idx 是全局的第几张图
                    patches_count = t * (h // 2) * (w // 2)
                    
                    # 取出对应的 visual features
                    vis_feat = visual_embeds[current_patch_idx : current_patch_idx + patches_count]
                    
                    # 放入 input_embeds
                    # 位置: start_locs[i] + 1 ... start_locs[i] + 1 + patches_count
                    # 这里的 input_ids 原本就是填充了 placeholder 的
                    # 直接替换
                    start_pos = start_locs[frame_idx] + 1
                    end_pos = start_pos + patches_count
                    
                    input_embeds_final[b, start_pos:end_pos] = vis_feat.to(input_embeds_final.dtype)
                    
                    current_patch_idx += patches_count
                    image_idx += 1

            # 替换 Pose Token
            # 注意：我们在 padded_ids 里插入了 PAD 占位符
            # 我们之前记录了 pose_token_indices
            
            for b in range(batch_size):
                indices = pose_token_indices[b] # list of positions
                
                for k, pos_idx in enumerate(indices):
                    # 第 k 个 pose token
                    # 如果 k=0 -> Ref (Frame 0)
                    # 如果 k>0 -> Target (Frame k)
                    
                    # 修改后：所有帧都使用相同的 Pose Token (Abs Pose)
                    # 也可以根据需要为不同帧使用不同的token，但为了简单，这里复用 index 0 的 token
                    # 这样学生模型就会学习这一个 Token 代表 "Pose" 并输出对应的绝对姿态
                    token_feat = pose_tokens[0, 0] # [1, D]
                    
                    input_embeds_final[b, pos_idx] = token_feat.to(input_embeds_final.dtype)
            
            
            # --- Student Forward ---
            # 使用 inputs_embeds
            # 注意：传入 inputs_embeds 时，不需要 input_ids (或者作为辅助)
            # 但是 Qwen 需要 input_ids 来计算 position_ids 吗？
            # 最好传入 attention_mask
            
            outputs = model(
                inputs_embeds=input_embeds_final,
                attention_mask=attention_mask,
                labels=padded_labels,
                output_hidden_states=True
            )
            loss_sft = outputs.loss
            
            # --- Distillation Logic ---
            last_hidden_state = outputs.hidden_states[-1] 
            
            # 1. 提取 Pose Token Outputs
            # 位置在 pose_token_indices
            
            student_pose_outputs = [] # Shape [All_Pose_In_Batch, D]
            teacher_pose_targets_flat = [] # Shape [All_Pose_In_Batch, D_teacher]

            
            if teacher_pose_targets is not None:
                # teacher_pose_targets: [1, S, 900] (因为我们只用了一个 batch 的 vision input)
                # student batch size = 1 typically (rank 0) or real batch size
                # 这里的代码假设 batch_size=1 (sample by sample loop)
                # training_samples loop is "one sample at a time"
                
                # 遍历 batch (其实只有 1 个 sample)
                for b in range(batch_size):
                    indices = pose_token_indices[b]
                    # indices 长度应该是 16 (num_frames)
                    
                    for k, pos_idx in enumerate(indices):
                        # Student Output
                        s_out = last_hidden_state[b, pos_idx] # [D]
                        s_proj = pose_proj(s_out) # [Teacher_Dim]
                        student_pose_outputs.append(s_proj)
                        
                        # Teacher Target
                        # Teacher shape: [1, S, 900]. S=16
                        # Batch index 0 (Assuming teacher processed 1 sample)
                        if k < teacher_pose_targets.shape[1]: 
                            t_out = teacher_pose_targets[0, k]
                            teacher_pose_targets_flat.append(t_out)
            
            # 2. Compute Orient Distill Loss
            if len(student_pose_outputs) > 0 and len(teacher_pose_targets_flat) > 0:
                s_stack = torch.stack(student_pose_outputs)
                t_stack = torch.stack(teacher_pose_targets_flat).to(s_stack.device)
                # print(f"Student Pose Proj Shape: {s_stack.shape}")
                # print(f"Teacher Pose Targets Shape: {t_stack.shape}")

                # MSE Loss for Features
                loss_ori = nn.MSELoss()(s_stack, t_stack)
            else:
                loss_ori = torch.tensor(0.0, device=device)
            
            
            # --- L4P Distillation Logic (Original) ---
            # 原有的 L4P 逻辑寻找 vision_start/end
            # 但是注意：我们修改了 input_ids 的长度和内容！用 padded_ids 替代 ids 查找
            
            ids = padded_ids[0] # assuming b=0
            locs_start = (ids == vision_start_token_id).nonzero(as_tuple=True)[0]
            vision_end_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>") # re-get just in case
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
                
                # grid_thw 需要调整吗？不需要，visual token 数量没变
                
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
                    
                    # ... logging values ...
                    if isinstance(loss_components["loss_ld"], torch.Tensor):
                        loss_ld = loss_components["loss_ld"].item()
                    else:
                        loss_ld = loss_components["loss_ld"]
                        
                    if isinstance(loss_components["loss_ed"], torch.Tensor):
                        loss_ed = loss_components["loss_ed"].item()
                    else:
                        loss_ed = loss_components["loss_ed"]
                except RuntimeError as e:
                    pass
            
            # --- Total Loss & Backward ---
            # Add Orient Loss
            total_loss = loss_sft + loss_distill + (lambda_ori * loss_ori)
            
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
            if rank == 0 and global_step_counter % accumulation_steps == 0:
                current_lr = optimizer.param_groups[0]["lr"]

                print(f"Sample {i+1}/{len(training_samples)} | Total: {total_loss.item():.4f} | SFT: {loss_sft.item():.4f} | Dis: {loss_distill.item():.4f} (LD: {loss_ld:.4f}, ED: {loss_ed:.4f} | Ori: {loss_ori.item():.4f} | LR: {current_lr:.2e})")


            epoch_loss_total += total_loss.item()
            epoch_loss_sft += loss_sft.item()
            epoch_loss_distill += loss_distill.item()
            epoch_loss_ori += loss_ori.item()
        
        # Calculate Average for this Process
        count = max(len(training_samples), 1)
        avg_total = epoch_loss_total / count
        avg_sft = epoch_loss_sft / count
        avg_distill = epoch_loss_distill / count
        avg_ori = epoch_loss_ori / count
        
        if rank == 0:
            print(f"Epoch {epoch+1} Avg (Rank0): Total {avg_total:.4f} | SFT {avg_sft:.4f} | Distill {avg_distill:.4f} | Ori {avg_ori:.4f}")
            
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
            
            # Save Pose Token & Proj
            torch.save({'pose_tokens': pose_tokens, 'pose_proj': pose_proj.state_dict()}, os.path.join(save_dir, "pose_adapter.pt"))
            
            print(f"Saved checkpoint to {save_dir}")
            
    # Cleanup
    if ddp_enabled:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
