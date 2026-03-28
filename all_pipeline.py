import os
# 在导入任何可能使用OpenMP的库之前设置环境变量
os.environ['OMP_NUM_THREADS'] = '1'

import sys
import json
import cv2
import torch
import logging
from transformers import logging as hf_logging
hf_logging.set_verbosity_error() # Prevent BertModel LOAD REPORT Output

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 开启 TF32 以加速
torch.set_float32_matmul_precision('high') 

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from transformers import AutoProcessor, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))

# 添加各种路径以供后续导入
sys.path.append(os.path.join(current_dir, "L4P-main"))
sys.path.append(os.path.join(current_dir, "Orient-Anything-V2"))
sys.path.append(os.path.join(current_dir, "GroundingDINO"))
sys.path.append(current_dir) # 当前目录

# 尝试导入自定义的蒸馏工具
try:
    from distill_utils import TeacherWrapper, GroundingDINOSAM2Teacher, OrientTeacher, D4DPerception, compute_distillation_loss
    from multi_teacher_loss import compute_multi_teacher_loss
    from l4p.utils.vis import flow_video_to_color_with_bounds, colormap_image
    from l4p.utils.misc import apply_fn
    from dependency_weights import DependencyAwareWeighter, DependencyWeightManager
    from consistency_losses import compute_consistency_losses
except ImportError as e:
    # 只有 rank 0 打印错误，避免刷屏，但在 import 失败时所有进程都应该退出
    print(f"Error importing distill_utils or l4p: {e}")
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
        
    def isatty(self):
        return getattr(self.terminal, "isatty", lambda: False)()
    
    def fileno(self):
        return getattr(self.terminal, "fileno", lambda: -1)()
        
    def isatty(self):
        return getattr(self.terminal, "isatty", lambda: False)()
    
    def fileno(self):
        return getattr(self.terminal, "fileno", lambda: -1)()
        
    def isatty(self):
        return getattr(self.terminal, "isatty", lambda: False)()
    
    def fileno(self):
        return getattr(self.terminal, "fileno", lambda: -1)()


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
        return None, None
        
    cap = cv2.VideoCapture(video_path)
    total_frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30.0
    
    if total_frames_count <= 0:
        return None, None
        
    indices = np.linspace(0, total_frames_count - 1, num_frames, dtype=int)
    frames = []
    timestamps = []
    
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
        ts = idx / fps
        timestamps.append(ts)
        
    cap.release()
    
    while len(frames) < num_frames:
        if frames:
            frames.append(frames[-1])
            timestamps.append(timestamps[-1] + (1.0/fps) if len(timestamps)>0 else 0.0)
        else:
            frames.append(Image.new('RGB', (target_resolution, target_resolution)))
            timestamps.append(0.0)
    
    return frames, timestamps

def save_image(img_path, img_np):
    """
    Save numpy array as image.
    img_np: (H, W, 3) or (H, W), values in [0, 1] or [0, 255]
    """
    if img_np.max() <= 1.05 and img_np.min() >= -0.05: # Assume float 0-1
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
    elif img_np.dtype != np.uint8:
        img_np = img_np.clip(0, 255).astype(np.uint8)
    
    if len(img_np.shape) == 2:
        img = Image.fromarray(img_np, mode='L')
    else:
        img = Image.fromarray(img_np)
    
    img.save(img_path)

def visualize_teacher_outputs(out_dir, p_m_teacher, step_idx):
    """
    Visualizes teacher outputs (Depth, Flow, Motion, Camray)
    """
    os.makedirs(out_dir, exist_ok=True)
    prefix = f"step_{step_idx:04d}"
    
    # 1. Depth: (B, C, T, H, W) -> (T, H, W, 3)
    if "depth" in p_m_teacher:
        depth_est = p_m_teacher["depth"][0] # (1, T, H, W)
        
        # Logic from l4p/utils/vis.py: generate_video_visualizations
        vis_min_depth, vis_max_depth = 0.05, 20.0
        
        # Calculate range
        valid_mask = depth_est > 0
        if valid_mask.any():
            min_val = torch.min(depth_est[valid_mask]).item()
            max_val = torch.max(depth_est[valid_mask]).item()
        else:
            min_val, max_val = 0.0, 1.0

        depth_range = (max(min_val, vis_min_depth), min(max_val, vis_max_depth))
        depth_est_clamped = torch.clamp(depth_est, min=depth_range[0], max=depth_range[1])
        depth_est_vis, _, _ = colormap_image(depth_est_clamped, vmin=depth_range[0], vmax=depth_range[1])
        
        # (3, T, H, W) -> (T, H, W, 3)
        depth_est_vis_np = depth_est_vis.permute(1, 2, 3, 0).cpu().numpy()
        
        depth_dir = os.path.join(out_dir, f"{prefix}_depth")
        os.makedirs(depth_dir, exist_ok=True)
        for i in range(depth_est_vis_np.shape[0]):
            save_image(os.path.join(depth_dir, f"frame_{i:02d}.png"), depth_est_vis_np[i])

    # 2. Flow: (B, 2, T, H, W)
    if "flow" in p_m_teacher:
        flow_est = p_m_teacher["flow"].cpu() # (B, 2, T, H, W)
        bflow_est_vis_b3thw, _ = flow_video_to_color_with_bounds(flow_est, None, max_flow_mag=25.0)
        # [B, 3, T, H, W] -> [T, H, W, 3] of first batch
        if bflow_est_vis_b3thw is not None:
            flow_vis_np = bflow_est_vis_b3thw[0].permute(1, 2, 3, 0).numpy()
            
            flow_dir = os.path.join(out_dir, f"{prefix}_flow")
            os.makedirs(flow_dir, exist_ok=True)
            for i in range(flow_vis_np.shape[0]):
                save_image(os.path.join(flow_dir, f"frame_{i:02d}.png"), flow_vis_np[i])

    # 3. Motion: (B, 1, T, H, W) - already sigmoided in TeacherWrapper
    if "motion" in p_m_teacher:
        motion_est = p_m_teacher["motion"][0] # (1, T, H, W)
        vis_thr = 0.85
        motion_est_bin = (motion_est > vis_thr).to(dtype=torch.float32)
        # (1, T, H, W) -> (T, H, W, 3)
        motion_vis_np = motion_est_bin[0, ..., None].repeat(1, 1, 1, 3).cpu().numpy()
        
        motion_dir = os.path.join(out_dir, f"{prefix}_motion")
        os.makedirs(motion_dir, exist_ok=True)
        for i in range(motion_vis_np.shape[0]):
            save_image(os.path.join(motion_dir, f"frame_{i:02d}.png"), motion_vis_np[i])

    # 4. Camray: (B, 6, T, H, W)
    if "camray" in p_m_teacher:
        camray = p_m_teacher["camray"][0].float().cpu().numpy() # (6, T, H, W)
        camray_dir = camray[:3].transpose(1, 2, 3, 0) # (T, H, W, 3)
        camray_mom = camray[3:].transpose(1, 2, 3, 0) # (T, H, W, 3)
        
        # Visualize direction
        camray_vis_dir = (camray_dir + 1.0) / 2.0
        
        # Visualize moment
        camray_vis_mom = np.zeros_like(camray_mom)
        for t in range(camray_mom.shape[0]):
            mom_t = camray_mom[t]
            vmin, vmax = mom_t.min(), mom_t.max()
            if vmax - vmin > 1e-8:
                camray_vis_mom[t] = (mom_t - vmin) / (vmax - vmin)
            else:
                camray_vis_mom[t] = 0.5 
        
        camray_out_dir = os.path.join(out_dir, f"{prefix}_camray")
        os.makedirs(camray_out_dir, exist_ok=True)
        for i in range(camray_vis_dir.shape[0]):
            save_image(os.path.join(camray_out_dir, f"frame_{i:02d}_dir.png"), camray_vis_dir[i])
            save_image(os.path.join(camray_out_dir, f"frame_{i:02d}_mom.png"), camray_vis_mom[i])


class ConditionalAttnDecoder(nn.Module):
    def __init__(self, query_dim=1408, kv_dim=1536, num_heads=8):
        super().__init__()
        self.kv_proj = nn.Linear(kv_dim, query_dim)
        self.attn = nn.MultiheadAttention(embed_dim=query_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)
        self.mlp = nn.Sequential(
            nn.Linear(query_dim, query_dim * 4),
            nn.GELU(),
            nn.Linear(query_dim * 4, query_dim)
        )
    def forward(self, query, kv):
        if kv is None or kv.shape[1] == 0:
            return query
        kv_proj = self.kv_proj(kv)
        attn_out, _ = self.attn(query, kv_proj, kv_proj)
        x = self.norm1(query + attn_out)
        x = self.norm2(x + self.mlp(x))
        return x

class SegDecoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # initialize to small weights
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)

    def forward(self, seg_hiddens, visual_hiddens, grid_thw, target_hw=(448, 448)):
        # seg_hiddens: (N, T, C)
        # visual_hiddens: (1, T*H*W, C)
        N, T, C = seg_hiddens.shape
        t_grid, h_grid, w_grid = grid_thw[0][0].item(), grid_thw[0][1].item(), grid_thw[0][2].item()
        
        seq_len = visual_hiddens.shape[1]
        
        # In Qwen VL, vision tokens are usually merged by 2x2, so h_feat = h_grid // 2
        # We calculate the actual spatial dimension dynamically
        num_patches_per_frame = seq_len // T
        import math
        spatial_dim = int(math.sqrt(num_patches_per_frame))
        
        if spatial_dim * spatial_dim != num_patches_per_frame:
            # If not perfectly square, just fallback to whatever aspect ratio it was approximately
            h_feat = h_grid // 2
            w_feat = w_grid // 2
            if h_feat * w_feat * t_grid != seq_len:
                # Then grid_thw might not match seq_len directly
                # We simply reshape to (T, seq_len/T, C) -> (T, C, H, W)
                pass # usually square
        else:
            h_feat = spatial_dim
            w_feat = spatial_dim
            
        v_h = visual_hiddens.view(T, h_feat, w_feat, C).permute(0, 3, 1, 2) # (T, C, h_feat, w_feat)
        seg_h = self.mlp(seg_hiddens) # (N, T, C)
        
        logits = (seg_h.unsqueeze(-1).unsqueeze(-1) * v_h.unsqueeze(0)).sum(dim=2) # (N, T, h, w)
        logits = F.interpolate(logits, size=target_hw, mode='bilinear', align_corners=False)
        return logits

class OriDecoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # 先投影到 900 维隐式特征 (对应 teacher 的 pose_enc)
        self.pose_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 900)
        )
        # 将 900 维特征进一步解码为 6 维显式角度
        self.mlp = nn.Sequential(
            nn.Linear(900, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 6)
        )
        
    def forward(self, ori_hiddens):
        pose_tokens = self.pose_proj(ori_hiddens)  # (N, T, 900)
        orients_6d = self.mlp(pose_tokens)         # (N, T, 6)
        return orients_6d, pose_tokens

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
    data_parquet_path = "../4D-Data/SIMS-VSI/sims_vsi_3q_40k.parquet"
    data_root = "../4D-Data/SIMS-VSI"
    num_frames = 16
    target_resolution = 448
    num_epochs = 3
    learning_rate = 2e-5
    
    # Object-level Tokens Config
    num_obj_tokens = 3 # N
    
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

    special_tokens = ["<seg_start>", "<seg_end>", "<ori_start>", "<ori_end>"]
    if num_obj_tokens > 0:
        # 增加 s_1..s_N, o_1..o_N 占位符
        special_tokens += [f"<seg_{i}>" for i in range(num_obj_tokens)]
        special_tokens += [f"<ori_{i}>" for i in range(num_obj_tokens)]
        
    num_added = processor.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    if num_added > 0:
        model.resize_token_embeddings(len(processor.tokenizer))
        # if rank == 0:
        #     print(f"Added {num_added} special tokens.")
            
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
    
    # 2. 新增可学习embedding参数：seg_tokens, ori_tokens
    if num_obj_tokens > 0:
        # [N, hidden_dim]
        seg_tokens = torch.nn.Parameter(torch.randn(num_obj_tokens, student_hidden_dim, dtype=torch.float32, device=device))
        ori_tokens = torch.nn.Parameter(torch.randn(num_obj_tokens, student_hidden_dim, dtype=torch.float32, device=device))
        model.register_parameter("seg_tokens", seg_tokens)
        model.register_parameter("ori_tokens", ori_tokens)
    else:
        seg_tokens = None
        ori_tokens = None

    # Wrap Student with DDP
    if ddp_enabled:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    
    # ==========================
    # 2. Initialize Teacher Models
    # ==========================
    if rank == 0:
        print("Initializing Teacher Models (L4P, GroundingDINO+SAM2, Orient V2)...")
    
    try:
        l4p_teacher = TeacherWrapper(device=device) 
        sam2_teacher = GroundingDINOSAM2Teacher(device=device)
        orient_teacher = OrientTeacher(device=device)
    except RuntimeError as e:
        raise e
        
    if rank == 0:
        print("Teacher Models initialized.")

    # ==========================
    # 3. Initialize Decoders (D4DP, Seg, Ori)
    # ==========================
    if rank == 0:
        print("Initializing Decoders...")
        
    d4dp = D4DPerception(input_dim=student_hidden_dim, output_dim=1408).to(dtype=torch.float32, device=device)
    d4dp.train()
    
    cond_attn = ConditionalAttnDecoder(query_dim=1408, kv_dim=student_hidden_dim, num_heads=8).to(dtype=torch.float32, device=device)
    cond_attn.train()
    
    seg_decoder = SegDecoder(hidden_dim=student_hidden_dim).to(dtype=torch.float32, device=device)
    seg_decoder.train()
    
    ori_decoder = OriDecoder(hidden_dim=student_hidden_dim).to(dtype=torch.float32, device=device)
    ori_decoder.train()
    
    if ddp_enabled:
        d4dp = DDP(d4dp, device_ids=[local_rank])
        cond_attn = DDP(cond_attn, device_ids=[local_rank])
        seg_decoder = DDP(seg_decoder, device_ids=[local_rank])
        ori_decoder = DDP(ori_decoder, device_ids=[local_rank])
    
    # ==========================
    # 4. Optimizer & Data
    # ==========================
    params_to_optimize = list(model.parameters()) + list(d4dp.parameters()) + list(cond_attn.parameters()) + list(seg_decoder.parameters()) + list(ori_decoder.parameters())
    if num_obj_tokens > 0 and seg_tokens is not None and ori_tokens is not None:
        params_to_optimize.extend([seg_tokens, ori_tokens])
    optimizer = torch.optim.AdamW(params_to_optimize, lr=learning_rate)
    
    # Initialize DependencyWeightManager
    weighter = DependencyAwareWeighter(momentum=0.99, min_weight=0.1)
    weight_manager = DependencyWeightManager(weighter)

    if rank == 0:
        print("Loading dataset...")
        
    all_valid_samples = []
    # 只有主进程读取文件没必要，为了简单可以都读，耗时不多
    import pandas as pd
    df = pd.read_parquet(data_parquet_path)
    full_data = df.to_dict('records')

    for item in full_data:
        vid_rel_path = item.get('video')
        if not vid_rel_path: continue
        vid_path = os.path.join(data_root, vid_rel_path)
        if os.path.exists(vid_path):
            conversations = item.get('conversations', [])
            if isinstance(conversations, np.ndarray):
                conversations = conversations.tolist()
            if len(conversations) >= 2:
                q_val = conversations[0].get('value')
                if q_val is None:
                    q_val = conversations[0].get('pwdvalue')
                a_val = conversations[1].get('value')
                if a_val is None:
                    a_val = conversations[1].get('pwdvalue')
                    
                if q_val is not None and a_val is not None:
                    question = q_val.replace("<video>\n", "").replace("<video>", "").replace("<image>\n", "").replace("<image>", "")
                    answer = a_val.replace("<video>\n", "").replace("<video>", "").replace("<image>\n", "").replace("<image>", "") if isinstance(a_val, str) else a_val
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
            frames_pil, timestamps = load_video_pil(video_path, num_frames=num_frames, target_resolution=target_resolution)
            if frames_pil is None: continue
                
            # --- Teacher Forward ---
            # To save memory, we load them to CUDA only when needed, and release immediately.
            with torch.no_grad():
                # 1. L4P Teacher
                f_4d_teacher, p_m_teacher = l4p_teacher.forward([frames_pil])
                f_4d_teacher = f_4d_teacher.cpu() # Keep on CPU until needed
                if "depth" in p_m_teacher: p_m_teacher["depth"] = p_m_teacher["depth"].cpu()
                if "flow" in p_m_teacher: p_m_teacher["flow"] = p_m_teacher["flow"].cpu()
                if "motion" in p_m_teacher: p_m_teacher["motion"] = p_m_teacher["motion"].cpu()
                if "camray" in p_m_teacher: p_m_teacher["camray"] = p_m_teacher["camray"].cpu()
                torch.cuda.empty_cache()

                # 2. SAM2 Teacher
                # Predict top-N objects. N is configured in `num_obj_tokens`.
                batch_masks, batch_boxes = sam2_teacher([frames_pil], [question], N=num_obj_tokens) 
                # batch_masks: List of (N_objs, T, H, W)
                torch.cuda.empty_cache()

                # 3. Orient Teacher
                batch_orients, batch_pose_tokens = orient_teacher([frames_pil], batch_masks, batch_boxes)
                # batch_orients: List of (N_objs, T, 3)
                # batch_pose_tokens: List of (N_objs, T, 900)
                torch.cuda.empty_cache()

            # Prepare GT variables for loss
            masks_gt = batch_masks[0].to(device) if len(batch_masks) > 0 and len(batch_masks[0]) > 0 else torch.zeros((0, num_frames, 448, 448), device=device)
            orients_gt = batch_orients[0].to(device) if len(batch_orients) > 0 and len(batch_orients[0]) > 0 else torch.zeros((0, num_frames, 3), device=device)
            pose_tokens_gt = batch_pose_tokens[0].to(device) if len(batch_pose_tokens) > 0 and len(batch_pose_tokens[0]) > 0 else torch.zeros((0, num_frames, 900), device=device)

            # Move f_4d_teacher back for loss
            f_4d_teacher = f_4d_teacher.to(device)
            p_m_teacher = {k: v.to(device) for k, v in p_m_teacher.items()}

                
            # --- Student Input Prep ---
            user_content = []
            for frame_idx, fr_pil in enumerate(frames_pil):
                ts = timestamps[frame_idx]
                time_str = f" [{int(ts//60):02d}:{ts%60:05.2f}s]"
                user_content.append({"type": "text", "text": time_str})
                user_content.append({"type": "image"})
                if num_obj_tokens > 0:
                    obj_text = "<seg_start>" + "".join([f"<seg_{j}>" for j in range(num_obj_tokens)]) + "<seg_end>"
                    obj_text += "<ori_start>" + "".join([f"<ori_{j}>" for j in range(num_obj_tokens)]) + "<ori_end>"
                    user_content.append({"type": "text", "text": obj_text})
            user_content.append({"type": "text", "text": question})
            
            messages = [
                {
                    "role": "user",
                    "content": user_content
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
            
            # 为了在不覆盖 input_ids 的情况下替换 embedding，我们将 inputs_embeds
            # 传给 Qwen 的一个特性，或者如果 Qwen3VL 不支持，我们使用 hook 或类似手法。
            # Qwen2VL 其实在 forward 的时候，只有没有 inputs_embeds 才会从 input_ids 取。
            # 但是如果传入 inputs_embeds，Qwen2VL 无法知道 image_pad_positions。
            # 这里我们注册一个针对 get_input_embeddings 的 hook，来实现替换。
            
            hook_handle = None
            if num_obj_tokens > 0:
                def embed_hook(module, args, output):
                    # To avoid in-place operation on leaf variables, we clone the output
                    output_clone = output.clone()
                    batch_input_ids = args[0]
                    for batch_idx in range(output.shape[0]):
                        for j in range(num_obj_tokens):
                            seg_id = processor.tokenizer.convert_tokens_to_ids(f"<seg_{j}>")
                            seg_mask = (batch_input_ids[batch_idx] == seg_id)
                            if seg_mask.any():
                                output_clone[batch_idx][seg_mask] = getattr(model, 'module', model).seg_tokens[j].to(output.dtype)
                                
                            ori_id = processor.tokenizer.convert_tokens_to_ids(f"<ori_{j}>")
                            ori_mask = (batch_input_ids[batch_idx] == ori_id)
                            if ori_mask.any():
                                output_clone[batch_idx][ori_mask] = getattr(model, 'module', model).ori_tokens[j].to(output.dtype)
                    return output_clone

                # Register hook on the actual embedding layer
                embed_layer = getattr(model, 'module', model).get_input_embeddings()
                hook_handle = embed_layer.register_forward_hook(embed_hook)
                
            outputs = model(**inputs, output_hidden_states=True)
            
            if hook_handle is not None:
                hook_handle.remove()
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
            loss_ori_ld = 0.0
            loss_ori_ed = 0.0
            loss_seg_distill = torch.tensor(0.0, device=device)
            loss_ori_distill = torch.tensor(0.0, device=device)
            loss_l4p = torch.tensor(0.0, device=device)
            
            if len(locs_start) > 0 and len(locs_end) > 0:
                visual_tokens_list = []
                for s, e in zip(locs_start, locs_end):
                    segment = last_hidden_state[:, s+1 : e, :]
                    visual_tokens_list.append(segment)
                
                student_visual_hidden = torch.cat(visual_tokens_list, dim=1) # (1, T*H*W, C)
                grid_thw = inputs.get("image_grid_thw", None)
                
                # === Extract Object Tokens (<seg_i>, <ori_i>) ===
                seg_hiddens_extracted = None
                ori_hiddens_extracted = None
                
                if num_obj_tokens > 0:
                    seg_h_list, ori_h_list = [], []
                    for obj_idx in range(num_obj_tokens):
                        seg_id = processor.tokenizer.convert_tokens_to_ids(f"<seg_{obj_idx}>")
                        ori_id = processor.tokenizer.convert_tokens_to_ids(f"<ori_{obj_idx}>")
                        
                        s_mask = (ids == seg_id)
                        o_mask = (ids == ori_id)
                        
                        s_feat = last_hidden_state[0, s_mask, :]
                        o_feat = last_hidden_state[0, o_mask, :]
                        
                        if s_feat.shape[0] == num_frames and o_feat.shape[0] == num_frames:
                            seg_h_list.append(s_feat)
                            ori_h_list.append(o_feat)
                        else:
                            if rank == 0:
                                print(f"Warning: Missing tokens for obj {obj_idx}, shape: s={s_feat.shape}, o={o_feat.shape}")
                    
                    if len(seg_h_list) == num_obj_tokens:
                        seg_hiddens_extracted = torch.stack(seg_h_list, dim=0) # (N, T, C)
                        ori_hiddens_extracted = torch.stack(ori_h_list, dim=0) # (N, T, C)

                try:
                    f_4d_student = d4dp(student_visual_hidden, grid_thw=grid_thw)
                    
                    if seg_hiddens_extracted is not None:
                        # Reshape to (1, N*T, C) for cross-attention
                        N, T, C = seg_hiddens_extracted.shape
                        kv_tokens = seg_hiddens_extracted.reshape(1, N*T, C)
                        f_4d_cond = cond_attn(f_4d_student, kv_tokens)
                    else:
                        f_4d_cond = f_4d_student
                        
                    p_m_student = l4p_teacher.decode_student_features(f_4d_cond)
                    # We need to compute p_m_student_uncond for camray? 
                    # Prompt: "相机姿态保持不变，仍然通过潜在4D特征"， let's just use f_4d_cond for all or mix them.
                    # Wait, prompt says: "对于原有的L4P解码，融合后的特征通过Dm，分别解码出深度、光流、运动分割（即显式显式特征的distillation）。而相机姿态保持不变，仍然通过潜在4D特征，因为相机运动和全图有关"
                    
                    p_m_student_uncond = l4p_teacher.decode_student_features(f_4d_student)
                    # Merge them
                    p_m_student_merged = {}
                    for k in p_m_student.keys():
                        if k in ["pred_camray"]:
                            p_m_student_merged[k] = p_m_student_uncond[k]
                        else:
                            p_m_student_merged[k] = p_m_student[k]
                    
                    # Compute weight from masks gt
                    weight_map = None
                    if len(batch_masks) > 0 and batch_masks[0] is not None and batch_masks[0].shape[0] > 0:
                        union_m, _ = batch_masks[0].max(dim=0) # [T, H, W]
                        m_2d = union_m.to(torch.float32).unsqueeze(1).to(device) # [T, 1, H, W]
                        # Create boundary mask using morphological operations
                        dilated = torch.nn.functional.max_pool2d(m_2d, kernel_size=5, stride=1, padding=2)
                        eroded = -torch.nn.functional.max_pool2d(-m_2d, kernel_size=5, stride=1, padding=2)
                        boundary = dilated - eroded
                        interior = eroded
                        background = 1.0 - dilated
                        
                        # 权重设定：背景 0.1, 内部 1.0, 边界 2.0 (您可以根据需要调整)
                        w_map = background * 0.1 + interior * 1.0 + boundary * 2.0
                        weight_map = w_map.squeeze(1) # [T, H, W]
                    
                    loss_l4p, loss_components = compute_distillation_loss(
                        f_4d_student, f_4d_teacher, p_m_student_merged, p_m_teacher, alpha=alpha, beta=beta,
                        weight_map=weight_map
                    )
                    loss_distill = loss_l4p
                    
                    loss_ld = loss_components["loss_ld"].item() if isinstance(loss_components["loss_ld"], torch.Tensor) else loss_components["loss_ld"]
                    loss_ed = loss_components["loss_ed"].item() if isinstance(loss_components["loss_ed"], torch.Tensor) else loss_components["loss_ed"]
                    
                    if seg_hiddens_extracted is not None and ori_hiddens_extracted is not None:
                        pred_masks = seg_decoder(seg_hiddens_extracted, student_visual_hidden, grid_thw, target_hw=(448, 448)) # (N, T, 448, 448)
                        pred_orients, pred_pose_tokens = ori_decoder(ori_hiddens_extracted) # (N, T, 6), (N, T, 900)
                        
                        # get the results as a tuple and manually unpack based on length
                        loss_res = compute_multi_teacher_loss(
                            pred_masks=pred_masks, 
                            pred_orients=pred_orients, 
                            pred_pose_tokens=pred_pose_tokens,
                            gt_masks=masks_gt, 
                            gt_orients_3d=orients_gt,
                            gt_pose_tokens=pose_tokens_gt
                        )
                        if isinstance(loss_res, tuple):
                            if len(loss_res) == 2:
                                ls, lo = loss_res
                                lo_ld, lo_ed, avg_dice = 0.0, 0.0, 0.0
                            elif len(loss_res) == 3:
                                ls, lo, avg_dice = loss_res
                                lo_ld, lo_ed = 0.0, 0.0
                            elif len(loss_res) == 5:
                                ls, lo, lo_ld, lo_ed, avg_dice = loss_res
                            else:
                                raise ValueError(f"Unexpected return length from compute_multi_teacher_loss: {len(loss_res)}")
                        else:
                            raise ValueError(f"Unexpected return type from compute_multi_teacher_loss: {type(loss_res)}")

                        
                        task_stats = {
                            'seg': avg_dice,
                            'obj_depth': loss_components.get('loss_depth', torch.tensor(0.0)).item() if isinstance(loss_components.get('loss_depth', 0.0), torch.Tensor) else loss_components.get('loss_depth', 0.0),
                            'obj_flow': loss_components.get('loss_flow', torch.tensor(0.0)).item() if isinstance(loss_components.get('loss_flow', 0.0), torch.Tensor) else loss_components.get('loss_flow', 0.0),
                            'obj_mo': loss_components.get('loss_motion', torch.tensor(0.0)).item() if isinstance(loss_components.get('loss_motion', 0.0), torch.Tensor) else loss_components.get('loss_motion', 0.0)
                        }
                        dynamic_weights = weight_manager.step(task_stats)
                        
                        # Apply dynamic weights
                        w_depth = dynamic_weights.get('obj_depth', 1.0)
                        w_flow = dynamic_weights.get('obj_flow', 1.0)
                        w_mo = dynamic_weights.get('obj_mo', 1.0)
                        w_ori = dynamic_weights.get('orient', 1.0)
                        w_seg = dynamic_weights.get('seg', 1.0)
                        
                        # Recompute L4P loss components with dynamic weights
                        weighted_ed = 0.0
                        if 'loss_depth' in loss_components: weighted_ed += 1.0 * loss_components['loss_depth'] * w_depth
                        if 'loss_flow' in loss_components: weighted_ed += 0.1 * loss_components['loss_flow'] * w_flow
                        if 'loss_motion' in loss_components: weighted_ed += 0.05 * loss_components['loss_motion'] * w_mo
                        if 'loss_camray' in loss_components: weighted_ed += 0.05 * loss_components['loss_camray'] # Always weight 1.0 for global?
                        
                        loss_l4p_weighted = alpha * loss_components["loss_ld"] + beta * weighted_ed
                        
                        loss_seg_distill = ls * w_seg
                        loss_ori_distill = lo * w_ori
                        loss_ori_ld = (lo_ld * w_ori).item() if isinstance(lo_ld, torch.Tensor) else lo_ld * w_ori
                        loss_ori_ed = (lo_ed * w_ori).item() if isinstance(lo_ed, torch.Tensor) else lo_ed * w_ori
                        
                        # Add consistency loss
                        pred_depth_for_cons = p_m_student_merged.get('depth', torch.zeros(1, num_frames, 448, 448, device=device))
                        loss_cons_raw, loss_cons_geo_raw, loss_cons_temp_raw = compute_consistency_losses(pred_masks, pred_depth_for_cons, pred_orients, timestamps)
                        
                        # Use a small regularization weight to prevent dominating the main tasks (even after normalization)
                        w_geo = 0.01
                        w_temp = 0.1
                        loss_cons_geo = loss_cons_geo_raw * w_geo
                        loss_cons_temp = loss_cons_temp_raw * w_temp
                        loss_cons = loss_cons_geo + loss_cons_temp
                        
                        loss_distill = loss_l4p_weighted + loss_seg_distill + loss_ori_distill + loss_cons
                        
                except RuntimeError as e:
                    if rank == 0:
                        import traceback
                        traceback.print_exc()
                        print(f"Distill Error: {e}")
                    loss_distill = torch.tensor(0.0, device=device)
                    loss_seg_distill = 0.0
                    loss_ori_distill = 0.0
                    loss_ori_ld = 0.0
                    loss_ori_ed = 0.0
                    loss_cons_geo = torch.tensor(0.0)
                    loss_cons_temp = torch.tensor(0.0)
            
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
                seg_val = loss_seg_distill.item() if isinstance(loss_seg_distill, torch.Tensor) else loss_seg_distill
                ori_total_val = loss_ori_distill.item() if isinstance(loss_ori_distill, torch.Tensor) else loss_ori_distill
                l4p_val = loss_l4p.item() if isinstance(loss_l4p, torch.Tensor) else loss_l4p
                cons_geo_val = loss_cons_geo.item() if 'loss_cons_geo' in locals() and isinstance(loss_cons_geo, torch.Tensor) else 0.0
                cons_temp_val = loss_cons_temp.item() if 'loss_cons_temp' in locals() and isinstance(loss_cons_temp, torch.Tensor) else 0.0

                print(f"Sample {i+1}/{len(training_samples)} | Total: {total_loss.item():.4f} | SFT: {loss_sft.item():.4f} | L4P: {l4p_val:.4f} | Seg: {seg_val:.4f} | Ori: {ori_total_val:.4f} | Cons_Geo: {cons_geo_val:.4f} | Cons_Temp: {cons_temp_val:.4f} | LR: {current_lr:.2e}")

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
            save_root = os.path.abspath(os.path.join(os.getcwd(), "..", "4D-Data", "checkpoints_all"))
            save_dir = os.path.join(save_root, f"epoch_{epoch+1}")
            os.makedirs(save_dir, exist_ok=True)
            
            # Unwrap DDP
            model_to_save = model.module if ddp_enabled else model
            model_to_save.save_pretrained(save_dir)
            processor.save_pretrained(save_dir)
            
            d4dp_to_save = d4dp.module if ddp_enabled else d4dp
            torch.save(d4dp_to_save.state_dict(), os.path.join(save_dir, "d4dp.pt"))
            
            cond_attn_to_save = cond_attn.module if ddp_enabled else cond_attn
            torch.save(cond_attn_to_save.state_dict(), os.path.join(save_dir, "cond_attn.pt"))
            
            seg_to_save = seg_decoder.module if ddp_enabled else seg_decoder
            torch.save(seg_to_save.state_dict(), os.path.join(save_dir, "seg_decoder.pt"))
            
            ori_to_save = ori_decoder.module if ddp_enabled else ori_decoder
            torch.save(ori_to_save.state_dict(), os.path.join(save_dir, "ori_decoder.pt"))
            
            torch.save(weighter.state_dict(), os.path.join(save_dir, "dependency_weighter.pt"))
            print(f"Saved checkpoint to {save_dir}")
            
    # Cleanup
    if ddp_enabled:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
