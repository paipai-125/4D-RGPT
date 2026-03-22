import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF
from typing import List, Dict, Tuple, Optional

# 尝试导入 L4P 相关模块，需要在主脚本中设置 sys.path
try:
    from l4p.models.utils import prepare_model
    from l4p.utils.geometry_utils import get_rays_plucker, normalize_intrinsics
    from l4p.utils.vis import flow_video_to_color_with_bounds, apply_fn
except ImportError:
    pass # 主程序会处理 sys.path

class TeacherWrapper(nn.Module):
    """
    TeacherWrapper 封装 L4P 模型
    负责:
    1. 加载 L4P 权重
    2. 预处理视频帧 (Resize, Normalize)
    3. 前向计算输出 F_4D (潜在特征) 和 P_m (显式4D信号)
    """
    def __init__(self, 
                 config_path="L4P-main/configs/model.yaml", 
                 ckpt_path="../4D-Data/l4p-weights/l4p_depth_flow_2d3dtrack_camray_dynseg_v1.ckpt",
                 device="cuda"):
        super().__init__()
        self.device = device
        
        # 优化: 获取设备索引，精准分配，避免所有进程都加载到 GPU0
        device_idx = [0] # 默认
        if isinstance(device, torch.device) and device.index is not None:
             device_idx = [device.index]
        elif isinstance(device, str) and ":" in device:
             try:
                 device_idx = [int(device.split(":")[-1])]
             except:
                 pass
        elif isinstance(device, int):
             device_idx = [device]

        # 加载 L4P 模型
        print(f"Loading L4P model from {ckpt_path} on device {device_idx}...")
        self.model = prepare_model(
            model_config_path=config_path,
            ckpt_path=ckpt_path,
            max_queries=64,
            precision="32", # 改为 "32" 以避免 distributed 模式下的 BF16/FP32 冲突
            accelerator="gpu" if device != "cpu" else "cpu",
            devices=device_idx # 显式传入设备列表
        )
        self.model.to(device)
        self.model.eval()
        
        # 冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False
            
        # 图像统计量 (ImageNet)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1).to(device)
        
        # 需要的显式任务
        self.tasks = ["depth", "flow_2d_backward", "dyn_mask", "camray"]
        # L4P 中 camray 可能对应 'traj3d' task head，如果包含则需要转换
        # 我们会检查模型的 task_heads keys
        
        # 预定义输入尺寸
        self.target_size = (224, 224) 

    def preprocess(self, frames_list_of_lists):
        """
        frames_list_of_lists: List[List[PIL.Image]], batch size B
        Returns: tensor (B, 3, T, H, W) normalized
        """
        batch_tensors = []
        for frames in frames_list_of_lists:
            # Resize and ToTensor
            # 这里的 frames 已经是 PIL Image
            frame_tensors = []
            for img in frames:
                # Resize 到 224x224
                img_resized = img.resize(self.target_size, resample=Image.BILINEAR)
                t = TF.to_tensor(img_resized) # (3, H, W), [0, 1]
                frame_tensors.append(t)
            
            # Stack T frames -> (T, 3, H, W)
            video_tensor = torch.stack(frame_tensors, dim=0)
            # Permute to (3, T, H, W)
            video_tensor = video_tensor.permute(1, 0, 2, 3)
            batch_tensors.append(video_tensor)
            
        # Stack Batch -> (B, 3, T, H, W)
        batch_tensors = torch.stack(batch_tensors, dim=0).to(self.device)

        # Normalize with float32
        batch_tensors = (batch_tensors - self.mean.to(batch_tensors.dtype)) / self.std.to(batch_tensors.dtype)

        # Force conversion to float32 for model input (avoid BF16/FP32 mixture)
        batch_tensors = batch_tensors.to(torch.float32)

        return batch_tensors

    @torch.no_grad()
    def forward(self, frames_list):
        """
        输入原始 PIL frames，返回教师模型的 4D 特征和显式信号
        """
        rgb_b3thw = self.preprocess(frames_list)
        B, _, T, H, W = rgb_b3thw.shape
        
        # 构造输入字典
        batch = {
            "rgb_b3thw": rgb_b3thw,
            # 如果需要 intrinsics，通常设为默认或者 None，取决于 L4P 是否强依赖
            # demo 中只传了 rgb (通过 dataset)
        }
        
        # 前向传播
        # L4P forward 返回字典，包含 task outputs 和 enc_features_bpc_list
        out = self.model.forward(batch, self.tasks)
        
        # 提取 F_4D (潜在特征)
        # my_demo.py 中取最后一个 block 的特征
        # enc_features_bpc_list 是一个 list of tensors (B, N_tokens, C)
        # 或者是 enc_features_bpc_2dlist (multi-window)
        # 我们假设输入是单窗口 (16帧)
        f_4d = None
        if "enc_features_bpc_list" in out:
            f_4d = out["enc_features_bpc_list"][-1] # (B, 2048, 1408) for 16x224x224
        elif "enc_features_bpc_2dlist" in out:
            f_4d = out["enc_features_bpc_2dlist"][-1][-1]
            
        # 提取显式信号 P_m
        explicit_signals = {}
        
        # 1. Depth: (B, 1, T, H, W) -> permute to (B, T, H, W, 1) if needed, 
        # 但为了 Loss 计算方便，保持 (B, C, T, H, W) 
        if "depth_est_b1thw" in out:
            explicit_signals["depth"] = apply_fn(out["depth_est_b1thw"], fn_type="linear")
            
        # 2. Flow: (B, 2, T, H, W)
        if "flow_2d_backward_est_b2thw" in out:
            explicit_signals["flow"] = out["flow_2d_backward_est_b2thw"]
            
        # 3. Motion: (B, 1, T, H, W) -> Sigmoid 处理得到掩码
        if "dyn_mask_est_b1thw" in out:
            explicit_signals["motion"] = torch.sigmoid(out["dyn_mask_est_b1thw"])
            
        # 4. Camray: (B, 6, T, H, W)
        if "camray_est_b6thw" in out:
            rays = out["camray_est_b6thw"]
            # Upsample Teacher output if needed to match global target (T, H, W)
            if rays.shape[-3:] != (T, H, W):
               rays = F.interpolate(
                   rays,
                   size=(T, H, W),
                   mode='trilinear',
                   align_corners=False
               )
            explicit_signals["camray"] = rays
        elif "traj3d_est_b16t" in out and "traj3d_intrinsics_est_b16t" in out:
            # 从轨迹重建 Ray
            _traj = out["traj3d_est_b16t"]
            _intrinsics = out["traj3d_intrinsics_est_b16t"]
            
            _world_T_cam = _traj.reshape(B, 4, 4, T).permute(0, 3, 1, 2) # (B, T, 4, 4)
            _cam_T_world = torch.linalg.inv(_world_T_cam).permute(0, 2, 3, 1) # (B, 4, 4, T)
            
            _intrinsics_reshaped = _intrinsics.reshape(B, 4, 4, T).float()
            _intrinsics_norm = normalize_intrinsics(_intrinsics_reshaped, H, W)
            
            _camray_b6thw, _ = get_rays_plucker(
                _intrinsics_norm,
                _cam_T_world,
                (H, W),
                make_first_cam_ref=True,
                normalize_dist=False
            )
            explicit_signals["camray"] = _camray_b6thw
            
        return f_4d, explicit_signals

    def decode_student_features(self, student_f_4d):
        """
        利用教师模型的 Task Heads 解码学生生成的潜在特征
        Args:
            student_f_4d: (B, 2048, 1408) similar to f_4d from teacher
        Returns:
            explicit_signals: dict of predictions
        """
        # 注意：L4P 的 Heads (PixelwiseTaskWithDPT) 通常期望多层特征列表 (hooks_idx)
        # 由于 D4DP 仅输出一层最终特征，我们采用复制策略构建伪列表
        # Hooks 通常取 [16, 24, 32, 40] 层输出，我们假设学生特征对应最后一层，并用于所有层
        # 这是一种适配策略
        
        fake_features_list = [student_f_4d] * 40 # 创建足够长度的列表，或者只填需要的索引位置
        # 实际上 DPT Head 内部会按 hooks_idx 取值，只要列表长度够且对应位置有 Tensor 即可
        # L4P VideoMAE encoder depth=40. 
        
        img_info = (16, 224, 224) # 固定
        out = {}
        
        # 调用各个 Head
        # 注意: 传入包含 'rgb_b3thw' 的空 dict 可能会报错，如果 head 需要 img_info 以外的信息
        # 看代码 dense_heads.py，主要依赖 enc_features_bpc_list
        
        # Extra args (img_info is passed positionally)
        extra_args = {
            "win_id": 0 # Default window id for single window input
        }
        
        # Depth
        # Fixed: Access l4p_model for task_heads
        try:
            if "depth" in self.tasks and "depth" in self.model.l4p_model.task_heads:
                # print("Processing depth task head...")
                res = self.model.l4p_model.task_heads["depth"](fake_features_list, img_info, **extra_args)
                out.update(res)
                
            # Flow
            if "flow_2d_backward" in self.tasks and "flow_2d_backward" in self.model.l4p_model.task_heads:
                # print("Processing flow task head...")
                res = self.model.l4p_model.task_heads["flow_2d_backward"](fake_features_list, img_info, **extra_args)
                out.update(res)
                
            # Motion (dyn_mask)
            if "dyn_mask" in self.tasks and "dyn_mask" in self.model.l4p_model.task_heads:
                # print("Processing motion task head...")
                res = self.model.l4p_model.task_heads["dyn_mask"](fake_features_list, img_info, **extra_args)
                out.update(res)
            
            # Camray
            if "camray" in self.tasks and "camray" in self.model.l4p_model.task_heads:
                # Similar to traj3d, camray head might imply PnP or other non-differentiable ops depending on implementation.
                # Use direct task_head call to get continuous 6D Plucker rays.
                head = self.model.l4p_model.task_heads["camray"]
                if hasattr(head, "task_head"):
                    # Direct differentiable path (B, 6, T_out, H_out, W_out) usually (16, 16, 16)
                    rays = head.task_head(fake_features_list, img_info)
                    
                    # Upsample if needed
                    target_T, target_H, target_W = img_info
                    if rays.shape[-3:] != (target_T, target_H, target_W):
                        rays = torch.nn.functional.interpolate(
                            rays,
                            size=(target_T, target_H, target_W),
                            mode='trilinear',
                            align_corners=False
                        )
                    
                    out["camray_est_b6thw"] = rays
                else:
                    res = self.model.l4p_model.task_heads["camray"](fake_features_list, img_info, **extra_args)
                    out.update(res)
            # 如果 camray 是通过 traj3d header 预测的
            elif "traj3d" in self.model.l4p_model.task_heads and "camray" in self.tasks:
                # traj3d head 包含非微分几何操作 (solvePnP)，导致 student 梯度中断或 numpy 错误。
                # 我们直接跳过 forward，调用 task_head 获取 dense rays (camray_est_b6thw)。
                head = self.model.l4p_model.task_heads["traj3d"]
                if hasattr(head, "task_head"):
                    # Direct differentiable path (B, 6, T_out, H_out, W_out)
                    rays = head.task_head(fake_features_list, img_info)

                    # Upsample if needed
                    target_T, target_H, target_W = img_info
                    if rays.shape[-3:] != (target_T, target_H, target_W):
                        rays = torch.nn.functional.interpolate(
                            rays,
                            size=(target_T, target_H, target_W),
                            mode='trilinear',
                            align_corners=False
                        )
                    
                    out["camray_est_b6thw"] = rays
                else:
                    # Fallback (may fail with gradient)
                    res = head(fake_features_list, img_info, **extra_args)
                    out.update(res)
        except Exception as e:
            print(f"Error inside decode_student_features task head execution: {e}")
            raise e
             
        # 整理输出格式，与 forward 一致
        explicit_signals = {}
        if "depth_est_b1thw" in out:
            explicit_signals["depth"] = apply_fn(out["depth_est_b1thw"], fn_type="linear")
        if "flow_2d_backward_est_b2thw" in out:
            explicit_signals["flow"] = out["flow_2d_backward_est_b2thw"]
        if "dyn_mask_est_b1thw" in out:
            explicit_signals["motion"] = torch.sigmoid(out["dyn_mask_est_b1thw"])
            
        if "camray_est_b6thw" in out:
            explicit_signals["camray"] = out["camray_est_b6thw"]
        elif "traj3d_est_b16t" in out:
            # 同样转换 traj -> rays
            B = student_f_4d.shape[0]
            T, H, W = 16, 224, 224
            _traj = out["traj3d_est_b16t"]
            _intrinsics = out["traj3d_intrinsics_est_b16t"]
            
            _world_T_cam = _traj.reshape(B, 4, 4, T).permute(0, 3, 1, 2)
            _cam_T_world = torch.linalg.inv(_world_T_cam).permute(0, 2, 3, 1)
            
            _intrinsics_reshaped = _intrinsics.reshape(B, 4, 4, T).float()
            _intrinsics_norm = normalize_intrinsics(_intrinsics_reshaped, H, W)
            
            _camray_b6thw, _ = get_rays_plucker(
                _intrinsics_norm,
                _cam_T_world,
                (H, W),
                make_first_cam_ref=True,
                normalize_dist=False
            )
            explicit_signals["camray"] = _camray_b6thw
            
        return explicit_signals


class D4DPerception(nn.Module):
    """
    4D 感知解码器 (Student Side)
    将的学生 LLM 隐藏状态映射到教师的 4D 潜在特征空间
    """
    def __init__(self, input_dim=4096, output_dim=1408, hidden_dim=2560):
        super().__init__()
        # 3层 MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Xavier 初始化
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(self, x, grid_thw=None):
        """
        Args:
            x: (B, N_tokens, D_in) -> e.g. (B, 3136, 4096)
            grid_thw: Optional tensor (B, 3) containing [T, H, W] grid dimensions
        Returns:
            f_student_4d: (B, 2048, 1408)
        """
        # Ensure input dtype matches MLP weights (fix for Float16 vs Float mismatch)
        target_dtype = self.mlp[0].weight.dtype
        if x.dtype != target_dtype:
            x = x.to(target_dtype)

        B, N, C = x.shape
        
        # 1. 恢复时空结构 (B, T, H, W, C)
        # Teacher Target: T=8, H=16, W=16 -> 2048
        # Student Input: T=16, H=?, W=? (e.g. 14x14 for 3136)
        
        T_student = 16
        if grid_thw is not None:
             # 如果提供了 grid 信息 (但通常 Qwen 的 grid 是 patch 后的)
             pass 

        # 推断 H, W
        if N % T_student == 0:
            spatial = N // T_student
            H_s = int(np.sqrt(spatial))
            W_s = spatial // H_s
        else:
             # Fallback
             H_s = int(np.sqrt(N / T_student))
             W_s = H_s

        # 尝试 Reshape: (B, N, C) -> (B, T, H, W, C)
        # 此处先把 token 序列恢复成 4D (Batch, Time, Height, Width, Channel)
        try:
             x = x.view(B, T_student, H_s, W_s, C)
        except RuntimeError:
             # 如果形状不匹配 (view 失败)，退回到 1D 插值
             # (B, N, C) -> (B, C, N) -> interpolate -> (B, C, 2048) -> (B, 2048, C)
             x = x.permute(0, 2, 1) 
             x = F.interpolate(x, size=2048, mode='linear', align_corners=False)
             x = x.permute(0, 2, 1).contiguous()
             return self.mlp(x)

        # 2. 调整分辨率: Permute to (B, C, T, H, W) for trilinear interpolation
        x = x.permute(0, 4, 1, 2, 3) # (B, C, T, H, W)
        
        target_T, target_H, target_W = 8, 16, 16
        
        # 使用三线性插值调整 T, H, W 到 Teacher 的分辨率
        if x.shape[2:] != (target_T, target_H, target_W):
            x = F.interpolate(x, size=(target_T, target_H, target_W), mode='trilinear', align_corners=False)
        
        # 3. 恢复并展平 (B, C, 8, 16, 16) -> (B, 8, 16, 16, C) -> (B, 2048, C)
        x = x.permute(0, 2, 3, 4, 1)
        x = x.flatten(1, 3) # (B, 2048, 4096)
        
        # 4. MLP 映射 (B, 2048, 4096) -> (B, 2048, 1408)
        # 这里的 MLP 会作用在最后一个维度上
        x = self.mlp(x) # (B, 2048, 1408)
        
        # print(f"DEBUG: D4DP Final Output Shape: {x.shape}")
        
        return x


def compute_distillation_loss(f_pred, f_true, p_pred_dict, p_true_dict, alpha=0.5, beta=0.1):
    """
    计算总的蒸馏损失
    Args:
        f_pred: 学生预测的潜在特征
        f_true: 教师真实的潜在特征
        p_pred_dict: 学生显式预测字典
        p_true_dict: 教师显式真值字典
        alpha: 潜在损失权重
        beta: 显式损失权重
    """
    loss_dict = {}
    
    # 1. 潜在特征损失 (SmoothL1)
    loss_ld = F.smooth_l1_loss(f_pred, f_true, beta=1.0)
    loss_dict["loss_ld"] = loss_ld
    
    # 2. 显式信号损失 (SmoothL1)
    # Weights: depth: 1.0, flow: 0.1, motion: 0.05, camray: 0.05
    weights = {"depth": 1.0, "flow": 0.1, "motion": 0.05, "camray": 0.05}
    loss_ed = 0.0
    
    for key in weights:
        if key in p_pred_dict and key in p_true_dict:
            pred = p_pred_dict[key]
            gt = p_true_dict[key]
            
            # 确保 shape 一致 (可能存在广播或微小差异，通常 L4P 输出一致)
            if pred.shape != gt.shape:
                # 简单 Resize 对齐 (以 teacher 为准)
                if pred.dim() == 5: # (B, C, T, H, W)
                     gt = F.interpolate(gt, size=pred.shape[2:], mode='nearest')
            
            l = F.smooth_l1_loss(pred, gt, beta=1.0)
            loss_ed += weights[key] * l
            
    loss_dict["loss_ed"] = loss_ed
    
    total_distill_loss = alpha * loss_ld + beta * loss_ed
    loss_dict["loss_distill_total"] = total_distill_loss
    
    return total_distill_loss, loss_dict
