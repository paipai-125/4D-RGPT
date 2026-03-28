import torch
import torch.nn.functional as F
import math

def _fix_depth_dim(d, target_hw=None):
    # Dims can come as (B, C, T, H, W), (1, T, H, W) etc.
    # We want robustly -> (T, H, W)
    if d.dim() == 5:
        d = d[0, 0] # (T, H, W)
    elif d.dim() == 4:
        if d.shape[0] == 1:
            d = d[0] # (T, H, W)
        elif d.shape[1] == 1: # (T, 1, H, W)
            d = d[:, 0] # (T, H, W)
        else:
            d = d[0] # Fallback
    elif d.dim() == 3:
        pass # Already (T, H, W)
        
    if target_hw is not None:
        T, H_d, W_d = d.shape
        H, W = target_hw
        if H_d != H or W_d != W:
            d = F.interpolate(d.unsqueeze(1).float(), size=(H, W), mode='bilinear', align_corners=False).squeeze(1)
            
    return d

def get_sobel_kernel(device):
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=device).view(1, 1, 3, 3)
    return sobel_x, sobel_y

def seg_depth_boundary_loss(pred_masks, pred_depth):
    # pred_masks: (N, T, H, W)
    N, T, H, W = pred_masks.shape
    device = pred_masks.device
    sobel_x, sobel_y = get_sobel_kernel(device)
    
    pred_depth = _fix_depth_dim(pred_depth, target_hw=(H, W)) # (T, H, W)
    depth_in = pred_depth.view(T, 1, H, W)
    
    dx = F.conv2d(depth_in, sobel_x, padding=1)
    dy = F.conv2d(depth_in, sobel_y, padding=1)
    depth_grad = torch.sqrt(dx**2 + dy**2 + 1e-6).view(T, H, W)
    
    total_loss = 0.0
    valid_count = 0
    
    for t in range(T):
        d_grad = depth_grad[t]
        d_mean, d_std = d_grad.mean(), d_grad.std()
        depth_edge = (d_grad > (d_mean + d_std)).float() # 二值化深度边缘
        
        for k in range(N):
            mask_k = pred_masks[k, t].view(1, 1, H, W)
            if mask_k.sum() < 100:
                continue
            mx = F.conv2d(mask_k, sobel_x, padding=1)
            my = F.conv2d(mask_k, sobel_y, padding=1)
            seg_edge_k = (torch.sqrt(mx**2 + my**2 + 1e-6) > 0.5).float().view(H, W)
            
            intersection = (seg_edge_k * depth_edge).sum()
            union = torch.clamp((seg_edge_k + depth_edge), 0, 1).sum()
            if union > 0:
                iou = intersection / union
                total_loss += (1.0 - iou)
                valid_count += 1
                
    return total_loss / max(valid_count, 1)

def seg_depth_distribution_loss(pred_masks, pred_depth):
    N, T, H, W = pred_masks.shape
    margin = 0.1
    total_loss = 0.0
    valid_count = 0
    
    pred_depth = _fix_depth_dim(pred_depth, target_hw=(H, W)) # (T, H, W)
    
    for t in range(T):
        depth_t = pred_depth[t] # (H, W)
        for k in range(N):
            mask_k = pred_masks[k, t] > 0.5
            if mask_k.sum() < 100:
                continue
            
            intra_vals = depth_t[mask_k]
            inter_vals = depth_t[~mask_k]
            
            if len(intra_vals) > 1 and len(inter_vals) > 1:
                intra_var = intra_vals.var()
                inter_var = inter_vals.var()
                
                loss_k = F.relu(intra_var - inter_var + margin)
                total_loss += loss_k
                valid_count += 1
                
    return total_loss / max(valid_count, 1)

def orient_seg_alignment_loss(pred_orients, pred_masks):
    # pred_orients: (N, T, 6)
    # pred_masks: (N, T, H, W)
    N, T, H, W = pred_masks.shape
    total_loss = 0.0
    valid_count = 0
    
    for t in range(T):
        for k in range(N):
            mask_k = pred_masks[k, t]
            if mask_k.sum() < 100:
                continue
            
            # 计算质心
            y_indices, x_indices = torch.where(mask_k > 0.5)
            if len(y_indices) < 50:
                continue
            cx = x_indices.float().mean()
            cy = y_indices.float().mean()
            
            # 解码 6D 到 3D
            o6d = pred_orients[k, t] # 6
            yaw   = torch.atan2(o6d[1], o6d[0])
            pitch = torch.atan2(o6d[3], o6d[2])
            roll  = torch.atan2(o6d[5], o6d[4])
            
            # 从yaw和pitch计算3D主轴并投影到2D
            axis_x = torch.cos(yaw) * torch.cos(pitch)
            axis_y = torch.sin(yaw) * torch.cos(pitch)
            len_2d = torch.sqrt(axis_x**2 + axis_y**2 + 1e-6)
            dir_x = axis_x / len_2d
            dir_y = axis_y / len_2d
            
            # 采样50点
            perm = torch.randperm(len(y_indices))[:50]
            sampled_y = y_indices[perm].float()
            sampled_x = x_indices[perm].float()
            
            # 点到直线距离 |A(x-cx) + B(y-cy)|, 这里的法向量是 (-dir_y, dir_x)
            distances = torch.abs(-dir_y * (sampled_x - cx) + dir_x * (sampled_y - cy))
            total_loss += (distances.mean() / float(max(H, W)))
            valid_count += 1
            
    return total_loss / max(valid_count, 1)

def temporal_consistency_loss(pred_masks, pred_depth, pred_orients, timestamps):
    N, T, H, W = pred_masks.shape
    if T < 2:
        return torch.tensor(0.0, device=pred_masks.device)
        
    pred_depth = _fix_depth_dim(pred_depth, target_hw=(H, W)) # (T, H, W)
    
    depth_diffs = 0.0
    orient_diffs = 0.0
    pos_diffs = 0.0
    valid_count = 0
    
    for t in range(T - 1):
        for k in range(N):
            mask_t = pred_masks[k, t] > 0.5
            mask_t1 = pred_masks[k, t+1] > 0.5
            
            if mask_t.sum() < 100 or mask_t1.sum() < 100:
                continue
                
            # 1. Depth change (normalize assuming typical diff in normalized scale ~5.0 units max)
            d_t = pred_depth[t][mask_t].mean()
            d_t1 = pred_depth[t+1][mask_t1].mean()
            depth_diff = torch.abs(d_t1 - d_t)
            depth_diffs += (depth_diff / 5.0)
            
            # 2. Orient change (pure radians, scaled down against PI)
            o_t = pred_orients[k, t]
            o_t1 = pred_orients[k, t+1]
            yaw_t = torch.atan2(o_t[1], o_t[0])
            yaw_t1 = torch.atan2(o_t1[1], o_t1[0])
            pitch_t = torch.atan2(o_t[3], o_t[2])
            pitch_t1 = torch.atan2(o_t1[3], o_t1[2])
            roll_t = torch.atan2(o_t[5], o_t[4])
            roll_t1 = torch.atan2(o_t1[5], o_t1[4])
            
            ori_diff = (torch.abs(yaw_t1 - yaw_t) + torch.abs(pitch_t1 - pitch_t) + torch.abs(roll_t1 - roll_t)) / math.pi
            orient_diffs += ori_diff
            
            # 3. Position change (Centroids mapped to 0.0 ~ 1.0 by dividing H, W)
            y_t, x_t = torch.where(mask_t)
            y_t1, x_t1 = torch.where(mask_t1)
            cx_t, cy_t = x_t.float().mean() / W, y_t.float().mean() / H
            cx_t1, cy_t1 = x_t1.float().mean() / W, y_t1.float().mean() / H
            pos_diff = torch.sqrt((cx_t1 - cx_t)**2 + (cy_t1 - cy_t)**2 + 1e-6)
            pos_diffs += pos_diff
            
            valid_count += 1
            
    if valid_count > 0:
        return (depth_diffs + orient_diffs + pos_diffs) / valid_count
    return torch.tensor(0.0, device=pred_masks.device)


def compute_consistency_losses(pred_masks, pred_depth, pred_orients, timestamps):
    l_bound = seg_depth_boundary_loss(pred_masks, pred_depth)
    l_dist = seg_depth_distribution_loss(pred_masks, pred_depth)
    l_align = orient_seg_alignment_loss(pred_orients, pred_masks)
    l_temp = temporal_consistency_loss(pred_masks, pred_depth, pred_orients, timestamps)
    
    total_loss = l_bound + l_dist + l_align + l_temp
    return total_loss, (l_bound + l_dist + l_align), l_temp
