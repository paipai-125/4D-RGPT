import os
import sys
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms

# Add L4P path
current_dir = os.path.dirname(os.path.abspath(__file__))
# Try relative path first, then absolute fallback
l4p_path = os.path.abspath(os.path.join(current_dir, "../../L4P-main"))
if not os.path.exists(l4p_path):
    l4p_path = "/root/autodl-tmp/4D-RGPT/L4P-main"

if l4p_path not in sys.path:
    print(f"Adding L4P path: {l4p_path}")
    sys.path.append(l4p_path)

try:
    from l4p.models.utils import prepare_model
    from l4p.utils.geometry_utils import get_rays_plucker, normalize_intrinsics
    print("L4P modules imported successfully.")
except ImportError as e:
    print(f"Warning: L4P modules not found. Error: {e}")
    print(f"Current sys.path: {sys.path}")
    prepare_model = None

class TeacherWrapper(nn.Module): 
    def __init__(self, ckpt_path, config_path, device="cuda"):
        super().__init__()
        self.device = device
        
        print(f"Loading Teacher Model from {ckpt_path}...")
        try:
            self.model = prepare_model(
                model_config_path=config_path,
                ckpt_path=ckpt_path,
                max_queries=64,
                precision="16-mixed", 
                accelerator="gpu" if "cuda" in device else "cpu",
            )
            self.model.eval()
            self.model.to(device)
            # Freeze completely
            for param in self.model.parameters():
                param.requires_grad = False
                
        except Exception as e:
            print(f"Failed to load teacher: {e}")
            self.model = None

        # Preprocessing: Standard L4P (ImageNet Norm)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        self.resize = transforms.Resize((224, 224))
        
    def preprocess(self, frames_list):
        # frames_list: List of length 16, each [H, W, 3] uint8 or float (0-1)
        # Convert to tensor: (1, 3, T, H, W)
        
        tensors = []
        for f in frames_list:
            # Check if numpy
            if isinstance(f, np.ndarray):
                f = torch.from_numpy(f).permute(2, 0, 1) # HWC -> CHW
            elif isinstance(f, torch.Tensor):
                if f.ndim == 3 and f.shape[-1] == 3: # HWC
                    f = f.permute(2, 0, 1)
            
            # Ensure float 0-1
            if f.dtype == torch.uint8:
                f = f.float() / 255.0
            
            # Resize
            # f is (C, H, W)
            f = self.resize(f)
            tensors.append(f)
            
        stack = torch.stack(tensors, dim=1).to(self.device) # (C, T, H, W)
        
        # Normalize
        # self.mean is (1, 3, 1, 1). Squeeze the batch dim to get (3, 1, 1)
        # stack is (3, T, H, W)
        mean = self.mean.view(3, 1, 1, 1)
        std = self.std.view(3, 1, 1, 1)
        stack = (stack - mean) / std
        
        return stack.unsqueeze(0) # (1, C, T, H, W)

    def forward(self, batch_tensor, tasks=["flow_2d_backward", "depth", "dyn_mask", "camray"]):
        if self.model is None:
            return {}
            
        batch_input = {
            "rgb_b3thw": batch_tensor,
            "dataset_idx": torch.zeros(batch_tensor.shape[0]),
        }
        
        with torch.no_grad():
            out = self.model(batch_input, tasks)
        return out


    def get_targets(self, frames_list):
        """
        frames_list: List of 16 frames (numpy arrays or tensors)
        Returns dict of targets.
        """
        # Batch inference is expensive. Use with torch.no_grad is not enough if graph is built.
        # But this function is usually called under torch.no_grad in distillation.py
        # Let's ensure returns are detached.
        
        batch_tensor = self.preprocess(frames_list)
        out = self.forward(batch_tensor)
        
        results = {}
        
        # Latent
        if "enc_features_bpc_list" in out:
            results["latent"] = out["enc_features_bpc_list"][-1].float().detach().cpu() # Move to CPU to save GPU Mem
        elif "enc_features_bpc_2dlist" in out:
            # Some L4P configs use 2dlist for hierarchical
            results["latent"] = out["enc_features_bpc_2dlist"][-1][-1].float().detach().cpu()
            
        # Explicit (Direct Outputs or Processed)
        # 1. Depth (Direct)
        if "depth_est_b1thw" in out:
            results["depth"] = out["depth_est_b1thw"].float().detach().cpu()
            
        # 2. Flow (Direct)
        if "flow_2d_backward_est_b2thw" in out:
            results["flow"] = out["flow_2d_backward_est_b2thw"].float().detach().cpu()

        # 3. Mask
        if "dyn_mask_est_b1thw" in out:
            results["dyn_mask"] = out["dyn_mask_est_b1thw"].float().detach().cpu()
            
        # 4. Camray
        if "camray_est_b9thw" in out:
            results["camray"] = out["camray_est_b9thw"].float().detach().cpu()
            
        # Clean up out to free memory immediately
        del out
        del batch_tensor
        torch.cuda.empty_cache()
            
        return results
        
        # If I use sigmoid, range is (0, 1). SmoothL1 is fine.
        # But Distillation is usually Feature Matching. Logits contain more info than saturated sigmoid.
        # Let's use LOGITS everywhere. my_demo.py uses sigmoid only for VISUALIZATION.

        # if "dyn_mask_est_b1thw" in out:
        #     results["mask"] = out["dyn_mask_est_b1thw"].float()
            
        # # 4. Camray (Direct or reconstruct)
        # if "camray_est_b6thw" in out:
        #     results["camera"] = out["camray_est_b6thw"].float()
        # elif "traj3d_est_b16t" in out and "traj3d_intrinsics_est_b16t" in out:
        #     # Reconstruct logic from my_demo.py
        #     _traj = out["traj3d_est_b16t"]
        #     _intrinsics = out["traj3d_intrinsics_est_b16t"]
        #     _B, _, _T = _traj.shape
        #     _H, _W = 224, 224 # Fixed output size used in L4P teacher
            
        #     _world_T_cam = _traj.reshape(_B, 4, 4, _T).permute(0, 3, 1, 2)
        #     _cam_T_world = torch.linalg.inv(_world_T_cam).permute(0, 2, 3, 1)
            
        #     _intrinsics_reshaped = _intrinsics.reshape(_B, 4, 4, _T).float()
        #     _intrinsics_norm = normalize_intrinsics(_intrinsics_reshaped, _H, _W)
            
        #     _camray_b6thw, _ = get_rays_plucker(
        #         _intrinsics_norm,
        #         _cam_T_world,
        #         (_H, _W),
        #         make_first_cam_ref=True,
        #         normalize_dist=False
        #     )
        #     results["camera"] = _camray_b6thw.float().to(self.device)
                
        # return results
