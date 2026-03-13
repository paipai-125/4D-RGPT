import sys
import os
# 在导入任何可能使用OpenMP的库之前设置环境变量
os.environ['OMP_NUM_THREADS'] = '4'

import argparse
import numpy as np
import torch
import cv2
import mediapy as media
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
from lightning.fabric import Fabric

# Add parent directory to sys.path to import l4p modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from l4p.models.utils import prepare_model
from l4p.data.video_dataset import VideoDataset
from l4p.utils.vis import flow_video_to_color_with_bounds, colormap_image
from l4p.utils.misc import apply_fn
from l4p.utils.geometry_utils import get_rays_plucker, normalize_intrinsics

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

def colormap_depth(depth, vmin=None, vmax=None, cmap="turbo"):
    # depth: (H, W)
    if vmin is None:
        vmin = depth.min()
    if vmax is None:
        vmax = depth.max()
    
    # Avoid division by zero
    denom = vmax - vmin
    if denom < 1e-8:
        denom = 1e-8
        
    depth_norm = (depth - vmin) / denom
    depth_norm = np.clip(depth_norm, 0, 1)
    
    cm = matplotlib.colormaps.get_cmap(cmap).reversed()
    depth_vis = cm(depth_norm)[:, :, :3] # (H, W, 3)
    return depth_vis

def main():
    parser = argparse.ArgumentParser(description="Run L4P on a custom video")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video")
    parser.add_argument("--output_dir", type=str, default="./results", help="Parent directory for results")
    args = parser.parse_args()

    video_path = args.video_path
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    # Create specific result directory for this video
    out_dir = os.path.join(args.output_dir, video_name)
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Processing video: {video_path}")
    print(f"Output directory: {out_dir}")

    # 1. Read video and uniformly sample 16 frames
    sampled_frames = []
    
    # Use mediapy (which uses ffmpeg) to read video
    try:
        with media.VideoReader(video_path) as reader:
            total_frames = reader.num_images
            print(f"Total frames in video: {total_frames}")
            
            if total_frames < 16:
                print("Video has fewer than 16 frames. Looping to fill 16 frames.")
                # Read all available frames
                frames = list(reader)
                # Pad
                while len(frames) < 16:
                    frames.append(frames[-1])
                sampled_frames = frames[:16]
            else:
                indices = np.linspace(0, total_frames - 1, 16).astype(int)
                print(f"Sampling frame indices: {indices}")
                
                # We need to iterate carefully
                current_frame_idx = 0
                target_indices = set(indices)
                
                for frame in reader:
                    if current_frame_idx in target_indices:
                        sampled_frames.append(frame)
                    current_frame_idx += 1
                    if len(sampled_frames) == 16:
                        break
        
    except Exception as e:
        print(f"Error using mediapy: {e}. Falling back to opencv.")
        sampled_frames = []
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 16:
            indices = list(range(total_frames)) + [total_frames-1]*(16-total_frames)
        else:
            indices = np.linspace(0, total_frames - 1, 16).astype(int)
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                sampled_frames.append(frame_rgb)
        cap.release()

    if len(sampled_frames) != 16:
        print(f"Warning: Expected 16 frames, got {len(sampled_frames)}. Padding if necessary.")
        while len(sampled_frames) < 16:
             sampled_frames.append(np.zeros_like(sampled_frames[0]))
        sampled_frames = sampled_frames[:16]

    # Clean up numpy arrays (mediapy returns them, opencv too)
    sampled_frames = [np.array(f) for f in sampled_frames]

    # Save input frames
    input_dir = os.path.join(out_dir, "input")
    os.makedirs(input_dir, exist_ok=True)
    for i, frame in enumerate(sampled_frames):
        img_path = os.path.join(input_dir, f"frame_{i:02d}.png")
        save_image(img_path, frame)
    print(f"Saved input frames to {input_dir}")

    # Create temp video for VideoDataset
    temp_video_path = os.path.join(out_dir, "temp_input.mp4")
    # Resize to have even dimensions if necessary for codec, but dataset will resize anyway
    # VideoDataset expects a path
    media.write_video(temp_video_path, sampled_frames, fps=10)
    
    # 2. Load Model
    precision = "16-mixed"
    accelerator = "gpu"
    model_name = "l4p_depth_flow_2d3dtrack_camray_dynseg_v1"
    # Assuming running from L4P-main/demo
    ckpt_path = f"weights/{model_name}.ckpt"
    model_config_path = "configs/model.yaml"
    
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found at {ckpt_path}. Please check your current directory.")
        return

    print("Loading model...")
    model = prepare_model(
        model_config_path=model_config_path,
        ckpt_path=ckpt_path,
        max_queries=64, 
        precision=precision,
        accelerator=accelerator,
    )
    
    # 3. Predict
    dataset = VideoDataset(
        video_paths=[temp_video_path],
        crop_size=(16, 224, 224), 
        resize_size=(224, 224),
        estimation_directions=[1],
    )
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    fabric = Fabric(precision=precision, accelerator=accelerator)
    dataloader = fabric.setup_dataloaders(dataloader)
    
    batch = next(iter(dataloader)) # Get the single batch (1 video)
    tasks = ["depth", "flow_2d_backward", "dyn_mask", "camray"] 
    # Note: 'camray' matches the name in model.yaml task_heads (actually task_name is 'traj3d' in yaml init_args but key is 'camray')
    # Let's check model code. l4p_model.py forwards tasks.
    # The keys in the ModuleDict are what matters. In model.yaml it is keys: flow_2d_backward, depth, camray, dyn_mask
    
    print("Running inference...")
    with torch.no_grad():
        out = model.forward(batch, tasks)

    # --- ADDED FOR DISTILLATION ---
    # Unified 4D encoder output latent 4D features
    # enc_features_bpc_list: List of features from each transformer block.
    # We take the last block's output as the latent representation.
    latent_4d_features = None
    if "enc_features_bpc_list" in out:
        latent_4d_features = out["enc_features_bpc_list"][-1]
    elif "enc_features_bpc_2dlist" in out:
        # Multi-window case: taking the last window's features for now
        latent_4d_features = out["enc_features_bpc_2dlist"][-1][-1]

    # Explicit Signals for Smooth-L1 Loss (H, W, C) per frame
    # Stored as (B, T, H, W, C) tensors so P(n) is accessible via slicing.

    # Depth (Pdepth): (H, W, 1) per frame
    P_depth = out["depth_est_b1thw"].permute(0, 2, 3, 4, 1) if "depth_est_b1thw" in out else None

    # Flow (Pflow): (H, W, 2) per frame
    P_flow = out["flow_2d_backward_est_b2thw"].permute(0, 2, 3, 4, 1) if "flow_2d_backward_est_b2thw" in out else None

    # Motion (Pmotion): (H, W, 1) per frame
    # Applying sigmoid to turn logits into [0, 1] mask suitable for loss
    P_motion = torch.sigmoid(out["dyn_mask_est_b1thw"]).permute(0, 2, 3, 4, 1) if "dyn_mask_est_b1thw" in out else None

    # Camray (Pcamray): (H, W, 6) per frame
    P_camray = None
    if "camray_est_b6thw" in out:
        P_camray = out["camray_est_b6thw"].permute(0, 2, 3, 4, 1)
    elif "traj3d_est_b16t" in out and "traj3d_intrinsics_est_b16t" in out:
        # Reconstruct rays for distillation variable
        _traj = out["traj3d_est_b16t"]
        _intrinsics = out["traj3d_intrinsics_est_b16t"]
        _B, _, _T = _traj.shape
        _H, _W = 224, 224 # Fixed output size used in this script
        
        _world_T_cam = _traj.reshape(_B, 4, 4, _T).permute(0, 3, 1, 2)
        _cam_T_world = torch.linalg.inv(_world_T_cam).permute(0, 2, 3, 1)
        
        _intrinsics_reshaped = _intrinsics.reshape(_B, 4, 4, _T).float()
        _intrinsics_norm = normalize_intrinsics(_intrinsics_reshaped, _H, _W)
        
        _camray_b6thw, _ = get_rays_plucker(
            _intrinsics_norm,
            _cam_T_world,
            (_H, _W),
            make_first_cam_ref=True,
            normalize_dist=False
        )
        P_camray = _camray_b6thw.permute(0, 2, 3, 4, 1)

    print("Distillation Variables Prepared:")
    print(f"  Latent Features: {latent_4d_features.shape if latent_4d_features is not None else 'Missing'}")
    print(f"  P_depth: {P_depth.shape if P_depth is not None else 'Missing'}")
    print(f"  P_flow: {P_flow.shape if P_flow is not None else 'Missing'}")
    print(f"  P_motion: {P_motion.shape if P_motion is not None else 'Missing'}")
    print(f"  P_camray: {P_camray.shape if P_camray is not None else 'Missing'}")
    # ------------------------------

    # 4. Save Outputs
    print("Saving results...")
    
    # Depth
    if "depth_est_b1thw" in out:
        depth_dir = os.path.join(out_dir, "depth")
        os.makedirs(depth_dir, exist_ok=True)
        
        # Logic from l4p/utils/vis.py: generate_video_visualizations
        vis_min_depth, vis_max_depth = 0.05, 20.0
        depth_est_1thw = apply_fn(out["depth_est_b1thw"][0], fn_type="linear")
        
        # Calculate range identically to infer.py (with safety check for empty)
        valid_mask = depth_est_1thw > 0
        if valid_mask.any():
            min_val = torch.min(depth_est_1thw[valid_mask]).item()
            max_val = torch.max(depth_est_1thw[valid_mask]).item()
        else:
            min_val, max_val = 0.0, 1.0

        depth_range = (
            max(min_val, vis_min_depth),
            min(max_val, vis_max_depth),
        )
        depth_est_1thw = torch.clamp(depth_est_1thw, min=depth_range[0], max=depth_range[1])
        depth_est_vis, _, _ = colormap_image(depth_est_1thw, vmin=depth_range[0], vmax=depth_range[1])
        
        # Returns (3, T, H, W) float tensor -> (T, H, W, 3)
        depth_est_vis_np = depth_est_vis.permute(1, 2, 3, 0).cpu().numpy()

        for i in range(16):
            # Vis function results are [0, 1] floats
            save_image(os.path.join(depth_dir, f"frame_{i:02d}.png"), depth_est_vis_np[i])
        print(f"Saved depth maps to {depth_dir}")

    # Flow
    if "flow_2d_backward_est_b2thw" in out:
        flow_dir = os.path.join(out_dir, "flow_2d_backward")
        os.makedirs(flow_dir, exist_ok=True)
        
        flow_2d_backward_est_b2thw = out["flow_2d_backward_est_b2thw"].cpu()
        bflow_est_vis_b3thw, flow_bounds = flow_video_to_color_with_bounds(
            flow_2d_backward_est_b2thw, None, max_flow_mag=25.0
        )
        # [B, 3, T, H, W] -> [T, H, W, 3]
        bflow_est_vis_thw3 = bflow_est_vis_b3thw[0].permute(1, 2, 3, 0).numpy()
        
        for i in range(16):
            save_image(os.path.join(flow_dir, f"frame_{i:02d}.png"), bflow_est_vis_thw3[i])
        print(f"Saved flow maps to {flow_dir}")

    # Motion Segmentation
    if "dyn_mask_est_b1thw" in out:
        mask_dir = os.path.join(out_dir, "dyn_mask")
        os.makedirs(mask_dir, exist_ok=True)
        
        dyn_mask_est_1thw = out["dyn_mask_est_b1thw"][0]
        dyn_mask_est_1thw = apply_fn(dyn_mask_est_1thw, fn_type="sigmoid")
        vis_thr = 0.85
        dyn_mask_est_1thw = (dyn_mask_est_1thw > vis_thr).to(dtype=torch.float32)
        # [1, T, H, W] -> [T, H, W, 3]
        dyn_mask_est_thw3 = dyn_mask_est_1thw[0, ..., None].repeat(1, 1, 1, 3).cpu().numpy()
        
        for i in range(16):
            save_image(os.path.join(mask_dir, f"frame_{i:02d}.png"), dyn_mask_est_thw3[i])
        print(f"Saved motion masks to {mask_dir}")

    # Camera Pose Rays
    camray_vis_dir = None
    camray_vis_mom = None
    
    # Check for direct ray output
    if "camray_est_b6thw" in out:
        camray = out["camray_est_b6thw"][0].float().cpu().numpy() # (6, T, H, W)
        
        # Split into direction and moment
        camray_dir = camray[:3].transpose(1, 2, 3, 0) # (T, H, W, 3)
        camray_mom = camray[3:].transpose(1, 2, 3, 0) # (T, H, W, 3)
        
        # Visualize direction: (-1, 1) -> (0, 1)
        camray_vis_dir = (camray_dir + 1.0) / 2.0
        
        # Visualize moment: Normalize min-max per frame for better visualization
        # Moment values can be arbitrary range depending on scene geometric scale
        # Simple normalization to [0, 1]
        camray_vis_mom = np.zeros_like(camray_mom)
        for t in range(camray_mom.shape[0]):
            mom_t = camray_mom[t]
            vmin, vmax = mom_t.min(), mom_t.max()
            if vmax - vmin > 1e-8:
                camray_vis_mom[t] = (mom_t - vmin) / (vmax - vmin)
            else:
                camray_vis_mom[t] = 0.5 

    # Check for trajectory output (traj3d head) -> Reconstruct rays
    elif "traj3d_est_b16t" in out:
        print("Reconstructing camera rays from estimated trajectory...")
        traj3d_est_b16t = out["traj3d_est_b16t"]
        B, _, T = traj3d_est_b16t.shape
        world_T_cam_b44t = traj3d_est_b16t.reshape(B, 4, 4, T)
        
        if "traj3d_intrinsics_est_b16t" in out:
            intrinsics_est_b16t = out["traj3d_intrinsics_est_b16t"]
            intrinsics_b44t = intrinsics_est_b16t.reshape(B, 4, 4, T)
            
            # Use geometry utils to reconstruct rays
            H, W = 224, 224 # Output size
            
            # Need extrinsics (cam_T_world) for get_rays_plucker
            # world_T_cam = inv(cam_T_world)
            # Permute for inverse: (B, 4, 4, T) -> (B, T, 4, 4)
            world_T_cam_bt44 = world_T_cam_b44t.permute(0, 3, 1, 2)
            cam_T_world_bt44 = torch.linalg.inv(world_T_cam_bt44)
            cam_T_world_b44t = cam_T_world_bt44.permute(0, 2, 3, 1) # (B, 4, 4, T)
            
            # Normalize intrinsics
            intrinsics_b44t = intrinsics_b44t.float()
            intrinsics_norm_b44t = normalize_intrinsics(intrinsics_b44t, H, W)
            
            # Compute Plucker Rays
            # Returns (B, 6, T, H, W)
            camray_b6thw, _ = get_rays_plucker(
                intrinsics_norm_b44t,
                cam_T_world_b44t,
                (H, W),
                make_first_cam_ref=True,
                normalize_dist=False
            )
            
            # Split into direction and moment
            camray = camray_b6thw[0].float().cpu().numpy() # (6, T, H, W)
            camray_dir = camray[:3].transpose(1, 2, 3, 0) # (T, H, W, 3)
            camray_mom = camray[3:].transpose(1, 2, 3, 0) # (T, H, W, 3)
            
            # Visualize direction: (-1, 1) -> (0, 1)
            camray_vis_dir = (camray_dir + 1.0) / 2.0
            
            # Visualize moment: Normalize min-max
            camray_vis_mom = np.zeros_like(camray_mom)
            for t in range(camray_mom.shape[0]):
                mom_t = camray_mom[t]
                # Normalize globally for the frame or channel-wise? 
                # Let's do simple min-max per frame to show structure
                vmin, vmax = mom_t.min(), mom_t.max()
                if vmax - vmin > 1e-8:
                    camray_vis_mom[t] = (mom_t - vmin) / (vmax - vmin)
                else:
                    camray_vis_mom[t] = 0.5

        else:
             print("Warning: Trajectory intrinsics missing. Skipping ray viz.")
    else:
        print(f"Warning: No camera info in output. Keys: {list(out.keys())}")

    if camray_vis_dir is not None:
        camray_dir_path = os.path.join(out_dir, "cam_pose_rays")
        os.makedirs(camray_dir_path, exist_ok=True)
        for i in range(16):
            # Save Direction
            save_image(os.path.join(camray_dir_path, f"frame_{i:02d}_dir.png"), camray_vis_dir[i])
            # Save Moment
            if camray_vis_mom is not None:
                save_image(os.path.join(camray_dir_path, f"frame_{i:02d}_mom.png"), camray_vis_mom[i])
        print(f"Saved camera pose rays (dir & mom) to {camray_dir_path}")

    # Cleanup
    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)
    
    print("Done!")

if __name__ == "__main__":
    main()
