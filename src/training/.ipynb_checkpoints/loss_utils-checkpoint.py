import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.1, weights={"depth": 1.0, "flow": 0.1, "mask": 0.05, "camray": 0.05}):
        super().__init__()
        self.alpha = alpha # Latent Distillation Coefficient
        self.beta = beta   # Explicit Distillation Coefficient
        self.weights = weights
        self.smooth_l1 = nn.SmoothL1Loss(beta=1.0, reduction='mean') 
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_out, teacher_targets, labels=None):
        """
        student_out: Dict from student model forward
        teacher_targets: Dict from teacher wrapper
        labels: Tensor [B, Seq] for SFT
        """
        losses = {
            "loss_sft": torch.tensor(0.0, device=student_out["logits"].device),
            "loss_ld": torch.tensor(0.0, device=student_out["logits"].device),
            "loss_ed": torch.tensor(0.0, device=student_out["logits"].device),
            "total_loss": torch.tensor(0.0, device=student_out["logits"].device)
        }
        
        # 1. SFT Loss (Language)
        if labels is not None:
             logits = student_out["logits"]
             # Shift logits and labels for causal LM loss
             shift_logits = logits[..., :-1, :].contiguous()
             shift_labels = labels[..., 1:].contiguous()
             
             losses["loss_sft"] = self.ce_loss(
                 shift_logits.view(-1, shift_logits.size(-1)), 
                 shift_labels.view(-1)
             )

        # -----------------------------------------------------
        # 2. Latent Distillation (LD) - Smooth L1
        # -----------------------------------------------------
        # student: List[Tensor] (one per sample in batch), [C, T, H, W]
        # teacher: List[Tensor] or Tensor [B, C, T, H, W]
        
        stud_latents = student_out.get("latent_features", [])
        teach_latents = teacher_targets.get("latent", []) 
        
        if stud_latents and teach_latents is not None:
            batch_loss_ld = 0.0
            valid_samples = 0
            
            for i in range(len(stud_latents)):
                if i >= len(stud_latents): break
                s_feat = stud_latents[i]
                if s_feat is None: continue
                
                # Get corresponding teacher feature
                if isinstance(teach_latents, list):
                    if i < len(teach_latents):
                        t_feat = teach_latents[i]
                    else:
                        continue
                elif isinstance(teach_latents, torch.Tensor):
                    if i < teach_latents.shape[0]:
                        t_feat = teach_latents[i]
                    else:
                        continue
                else:
                    continue
                    
                t_feat = t_feat.to(s_feat.device)
                
                # Handle Dimensions
                if s_feat.dim() == 4: s_feat = s_feat.unsqueeze(0)
                if t_feat.dim() == 4: t_feat = t_feat.unsqueeze(0)
                
                # Interpolate if shape mismatch
                if s_feat.shape != t_feat.shape:
                    if s_feat.shape[1] != t_feat.shape[1]: # Channels
                         # Cannot compute if channels differ without projection
                         continue
                         
                    # Interpolate Spatial/Temporal
                    t_feat = F.interpolate(
                        t_feat, 
                        size=s_feat.shape[2:], # [T, H, W]
                        mode='trilinear',
                        align_corners=False
                    )

                loss = self.smooth_l1(s_feat, t_feat)
                batch_loss_ld += loss
                valid_samples += 1
                
            if valid_samples > 0:
                losses["loss_ld"] = batch_loss_ld / valid_samples

        # -----------------------------------------------------
        # 3. Explicit Distillation (ED) - Smooth L1
        # -----------------------------------------------------
        task_map = {
            "depth": "depth",
            "mask": "dyn_mask",
            "flow": "flow_2d_backward",
            "camera": "camray"
        }
        
        stud_explicit = student_out.get("explicit_outputs", {})
        
        batch_loss_ed = 0.0
        
        for stud_key, teach_key in task_map.items():
            if stud_key not in stud_explicit or teach_key not in teacher_targets:
                continue
                
            s_preds_list = stud_explicit[stud_key] 
            t_targets_batch = teacher_targets[teach_key]
            
            task_loss = 0.0
            valid_task_samples = 0
            
            for i in range(len(s_preds_list)):
                s_pred = s_preds_list[i]
                if s_pred is None: continue
                
                # Get teacher target
                if isinstance(t_targets_batch, list):
                    if i < len(t_targets_batch):
                        t_target = t_targets_batch[i]
                    else: continue
                elif isinstance(t_targets_batch, torch.Tensor):
                    if i < t_targets_batch.shape[0]:
                        t_target = t_targets_batch[i]
                    else: continue
                else: continue
                
                t_target = t_target.to(s_pred.device)
                
                # Align Dimensions
                if s_pred.dim() == 4: s_pred = s_pred.unsqueeze(0)
                if t_target.dim() == 4: t_target = t_target.unsqueeze(0)
                
                if s_pred.shape != t_target.shape:
                     t_target = F.interpolate(
                        t_target,
                        size=s_pred.shape[2:], 
                        mode='trilinear',
                        align_corners=False
                    )
                
                if torch.isnan(s_pred).any() or torch.isnan(t_target).any():
                     continue
                     
                loss = self.smooth_l1(s_pred, t_target)
                task_loss += loss
                valid_task_samples += 1
            
            if valid_task_samples > 0:
                weight = self.weights.get(stud_key, 1.0)
                batch_loss_ed += weight * (task_loss / valid_task_samples)

        losses["loss_ed"] = batch_loss_ed
        
        # Total Loss
        losses["total_loss"] = losses["loss_sft"] + self.alpha * losses["loss_ld"] + self.beta * losses["loss_ed"]
        
        return losses
