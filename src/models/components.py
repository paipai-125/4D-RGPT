import torch
import torch.nn as nn
import math
import types

# -----------------------------------------------------------------------------
# 1. Define TimePositionalEncoding
# -----------------------------------------------------------------------------
class TimePositionalEncoding(nn.Module):
    def __init__(self, d_model=1152):
        super().__init__()
        self.d_model = d_model
        # T = 10000, D = d_model
        # div_term = 1 / T^(2i/D)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.register_buffer('div_term', div_term)

    def forward(self, timestamps):
        # timestamps: [N] (tensor of time values, e.g., t(n))
        # Ensure timestamps are float for calculation
        timestamps = timestamps.float()
        
        # Calculate phase: t / T^(2i/D)
        # timestamps: [N], div_term: [D/2] -> [N, D/2]
        phase = timestamps.unsqueeze(1) * self.div_term
        
        pe = torch.zeros(timestamps.size(0), self.d_model, device=timestamps.device, dtype=timestamps.dtype)
        pe[:, 0::2] = torch.sin(phase)
        pe[:, 1::2] = torch.cos(phase)
        
        return pe

# -----------------------------------------------------------------------------
# 2. Define D4DP Decoder
# -----------------------------------------------------------------------------
class D4DP_Decoder(nn.Module):
    """
    Training-only 4D Perception Decoder (D4DP).
    Maps LLM hidden states to a list of feature maps simulating Teacher's Latent Features.
    Teacher (L4P VideoMAE-L) features are typically 1024.
    """
    def __init__(self, input_dim=3584, output_dim=1024, hidden_dim=2560):
        super().__init__()
        # 3-layer MLP to project LLM Token Dim -> VideoMAE Channel Dim
        # Hidden layer dimension 2560, GELU activation
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim) 
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, T, H_patches, W_patches):
        """
        x: [B, Tokens, D_llm]
        Assuming x contains only visual tokens or we splice them out before calling this.
        Returns: list of tensors [B, C, T, H, W]
        """
        features = self.projector(x) # [B, T*P, C]
        
        B = features.shape[0] 
        C = features.shape[-1]
        
        # Reshape [B, T, H, W, C]
        features = features.view(B, T, H_patches, W_patches, C)
        
        # Permute to [B, C, T, H, W]
        features = features.permute(0, 4, 1, 2, 3) 
        
        # Replicate for compatibility with DPT hooks (usually expects 4 hierarchical features)
        return [features, features, features, features]


# -----------------------------------------------------------------------------
# 3. Patching Logic (Extended with Decoders)
# -----------------------------------------------------------------------------

class TimeAwareMerger(nn.Module):
    def __init__(self, original_merger, time_embed_module, parent_visual):
        super().__init__()
        self.merger = original_merger
        self.time_embed = time_embed_module
        # Store parent safely
        object.__setattr__(self, "parent", parent_visual) 
        
    def forward(self, hidden_states):
        # hidden_states: [TotalTokens, Dim]
        # grid_thw: [B, 3] (T, H, W) stored in self.parent.current_grid_thw
        
        grid_thw = getattr(self.parent, "current_grid_thw", None)
        
        if grid_thw is not None:
            # Add Time Embeddings
            all_time_embeds = []
            device = hidden_states.device
            dtype = hidden_states.dtype
            
            start_idx = 0
            for i in range(grid_thw.shape[0]):
                T, H, W = grid_thw[i]
                T, H, W = int(T.item()), int(H.item()), int(W.item())
                
                # Number of patches per frame = H * W
                num_patches_per_frame = H * W
                total_tokens = T * num_patches_per_frame
                
                # Create timestamps
                if T > 0:
                    # shape [T, H*W] -> flatten
                    times = torch.arange(T, device=device).unsqueeze(1).repeat(1, num_patches_per_frame).flatten()
                    # Get embeddings
                    t_embed = self.time_embed(times) # [Tokens, Dim]
                    all_time_embeds.append(t_embed)
                else:
                    # Should not happen for video but handle empty case
                    all_time_embeds.append(torch.zeros(total_tokens, hidden_states.shape[-1], device=device, dtype=dtype))
                
                start_idx += total_tokens
                    
            if all_time_embeds:
                time_embeds_tensor = torch.cat(all_time_embeds, dim=0).to(dtype)
                
                # Check shape match (sanity check)
                if time_embeds_tensor.shape[0] == hidden_states.shape[0]:
                     hidden_states = hidden_states + time_embeds_tensor
        
        return self.merger(hidden_states)

def apply_time_encoding_patch(model, device):
    """
    Monkey-patch the visual encoder to inject Time Positional Encoding 
    BEFORE the projector (Merger/MLP).
    """
    
    print("Applying Time Positional Encoding Patch...")

    # 1. Locate the Visual Module
    if hasattr(model, "visual"):
        visual_module = model.visual
    elif hasattr(model, "model") and hasattr(model.model, "visual"):
        visual_module = model.model.visual
    else:
        try:
             # Last resort
             visual_module = model.visual
        except:
             raise AttributeError("Could not find 'visual' module in model.")

    print(f"Found visual module.")
    
    # 2. Initialize Time Encoding Module
    vision_config = getattr(visual_module, "config", None)
    embed_dim = getattr(vision_config, "embed_dim", 1152) if vision_config else 1152
    
    time_embed_module = TimePositionalEncoding(d_model=embed_dim).to(device)
    model.time_embed_module = time_embed_module
    
    # 3. Capture the original forward of the visual module
    original_visual_forward = visual_module.forward
    
    # 4. Define robust wrapper for Visual Forward
    def captured_visual_forward_generic(self, *args, **kwargs):
        captured_grid = kwargs.get("grid_thw", None)
        if captured_grid is None:
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    if arg.dim() == 2 and arg.shape[-1] == 3:
                         captured_grid = arg
                         break
        
        self.current_grid_thw = captured_grid
        return original_visual_forward(*args, **kwargs)
        
    # Apply patches
    visual_module.forward = types.MethodType(captured_visual_forward_generic, visual_module)
    
    # Wrap merger
    target_merger_name = None
    if hasattr(visual_module, "merger"):
        target_merger_name = "merger"
    elif hasattr(visual_module, "attn_pool"):
        target_merger_name = "attn_pool"
    
    if target_merger_name:
        current_merger = getattr(visual_module, target_merger_name)
        if not isinstance(current_merger, TimeAwareMerger):
             wrapped_merger = TimeAwareMerger(current_merger, time_embed_module, visual_module)
             setattr(visual_module, target_merger_name, wrapped_merger)

    print("Time Encoding Patch applied successfully.")
    
    return model
