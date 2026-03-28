import torch
import torch.nn.functional as F
import math
from scipy.optimize import linear_sum_assignment

def dice_loss(inputs, targets, smooth=1e-5):
    """
    inputs: (1, T, H, W) logits
    targets: (1, T, H, W) binary (0/1)
    """
    inputs = torch.sigmoid(inputs)
    inputs = inputs.reshape(inputs.shape[0], -1)
    targets = targets.reshape(targets.shape[0], -1)
    intersection = (inputs * targets).sum(-1)
    dice = (2. * intersection + smooth) / (inputs.sum(-1) + targets.sum(-1) + smooth)
    return 1 - dice

def compute_multi_teacher_loss(pred_masks, pred_orients, pred_pose_tokens, gt_masks, gt_orients_3d, gt_pose_tokens):
    """
    pred_masks: (N_pred, T, H, W)  logits
    pred_orients: (N_pred, T, 6)   6D representation from orientation token MLP
    pred_pose_tokens: (N_pred, T, 900) Latent orientation features from student
    gt_masks: (N_gt, T, H, W)      binary (0/1) masks from SAM2
    gt_orients_3d: (N_gt, T, 3)    Degree representations (yaw, pitch, roll) from Orient V2
    gt_pose_tokens: (N_gt, T, 900) Latent orientation features from Orient V2
    """
    N_pred, T = pred_masks.shape[0], pred_masks.shape[1]
    N_gt = gt_masks.shape[0] if gt_masks is not None else 0
    device = pred_masks.device
    
    # 1. Convert Orient V2 3D (degree) to 6D (sin/cos)
    if N_gt > 0:
        # gt_orients_3d: (N_gt, T, 3) 
        # convert to radians
        angles_rad = gt_orients_3d * (math.pi / 180.0)
        sin_vals = torch.sin(angles_rad)
        cos_vals = torch.cos(angles_rad)
        # interleave to shape (N_gt, T, 6)
        # [sin_az, cos_az, sin_el, cos_el, sin_ro, cos_ro]
        gt_orients_6d = torch.stack([
            sin_vals[..., 0], cos_vals[..., 0], 
            sin_vals[..., 1], cos_vals[..., 1], 
            sin_vals[..., 2], cos_vals[..., 2]
        ], dim=-1)
    else:
        gt_orients_6d = None

    if N_gt == 0:
        # All unmatched
        loss_seg = F.binary_cross_entropy_with_logits(pred_masks, torch.zeros_like(pred_masks))
        loss_orient = torch.tensor(0.0, device=device, requires_grad=True)
        return loss_seg, loss_orient

    # 2. Compute cost matrix by treating the mask over whole video (T, H, W)
    # Detach predictions for matching stability
    pred_masks_det = pred_masks.detach()
    pred_flat = torch.sigmoid(pred_masks_det).reshape(N_pred, -1)   # (N_pred, T*H*W)
    gt_flat = gt_masks.float().reshape(N_gt, -1)                    # (N_gt, T*H*W)

    intersection = torch.mm(pred_flat, gt_flat.T)                   # (N_pred, N_gt)
    pred_sum = pred_flat.sum(dim=-1).unsqueeze(1)                   # (N_pred, 1)
    gt_sum = gt_flat.sum(dim=-1).unsqueeze(0)                       # (1, N_gt)
    
    dice_scores = (2. * intersection + 1e-5) / (pred_sum + gt_sum + 1e-5)
    cost_matrix = 1.0 - dice_scores                                 # (N_pred, N_gt)
    
    # 3. Hungarian matching
    cost_matrix_np = cost_matrix.cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_matrix_np)
    
    matched_pairs = []
    unmatched_preds = set(range(N_pred))
    
    for r, c in zip(row_ind, col_ind):
        matched_pairs.append((r, c))
        unmatched_preds.remove(r)
            
    # 4. Compute Segmentation Loss
    loss_seg = 0.0
    for r, c in matched_pairs:
        p_m = pred_masks[r].unsqueeze(0)   # (1, T, H, W)
        g_m = gt_masks[c].unsqueeze(0).float()     # (1, T, H, W)
        bce = F.binary_cross_entropy_with_logits(p_m, g_m)
        dl = dice_loss(p_m, g_m).mean()
        loss_seg = loss_seg + bce + dl

    for r in unmatched_preds:
        p_m = pred_masks[r]                # (T, H, W)
        bce = F.binary_cross_entropy_with_logits(p_m, torch.zeros_like(p_m))
        loss_seg = loss_seg + bce

    loss_seg = loss_seg / max(1, N_pred)

    # 5. Compute Orient Loss via both 6D Angle and latent MSE
    loss_orient = 0.0
    loss_orient_ld = 0.0  # Latent Distillation
    loss_orient_ed = 0.0  # Explicit Distillation
    if len(matched_pairs) > 0:
        valid_pairs = 0
        for r, c in matched_pairs:
            # Explicit 6D Angle Loss (kept from before, if desired)
            p_o = pred_orients[r]    # (T, 6)
            g_o = gt_orients_6d[c]   # (T, 6)
            
            # To handle uncomputed objects which are [0,0,0], we can just compute smoothly
            loss_6d = F.mse_loss(p_o, g_o)
            
            # Latent Pose Token Distillation (Smooth L1 Loss as requested)
            p_pose = pred_pose_tokens[r] # (T, 900)
            g_pose = gt_pose_tokens[c]   # (T, 900)
            
            loss_latent = F.smooth_l1_loss(p_pose, g_pose)
            
            loss_orient_ed = loss_orient_ed + loss_6d
            loss_orient_ld = loss_orient_ld + loss_latent
            loss_orient = loss_orient + loss_6d + loss_latent
            valid_pairs += 1
            
        loss_orient = loss_orient / max(1, valid_pairs)
        loss_orient_ed = loss_orient_ed / max(1, valid_pairs)
        loss_orient_ld = loss_orient_ld / max(1, valid_pairs)
    else:
        loss_orient = torch.tensor(0.0, device=device, requires_grad=True)
        loss_orient_ed = torch.tensor(0.0, device=device, requires_grad=True)
        loss_orient_ld = torch.tensor(0.0, device=device, requires_grad=True)

    # Compute average dice score for the matched pairs to be used as quality metric
    avg_dice = 0.0
    if len(matched_pairs) > 0:
        sum_dice = sum([dice_scores[r, c].item() for r, c in matched_pairs])
        avg_dice = sum_dice / len(matched_pairs)
        
    return loss_seg, loss_orient, loss_orient_ld, loss_orient_ed, avg_dice

