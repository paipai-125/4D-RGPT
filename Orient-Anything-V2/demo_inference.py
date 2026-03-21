import argparse
import os
import sys
import warnings

# specific for your environment, if you see an error about OMP:
os.environ["OMP_NUM_THREADS"] = "1"
# Suppress specific Numba TBB warning
warnings.filterwarnings('ignore', message='.*The TBB threading layer requires TBB version.*')

import torch
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
import traceback

# Add current directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from utils.app_utils import inf_single_case, background_preprocess
    from utils.utils import Get_target_azi_ele_rot
except ImportError as e:
    # Fallback import if utils.utils doesn't contain the function or fails
    try:
        from utils.app_utils import inf_single_case, background_preprocess, Get_target_azi_ele_rot
    except ImportError as inner_e:
         print(f"Error importing utils: {inner_e} (original error: {e})")
         print("Please make sure you have installed all requirements.")
         print("pip install -r requirements.txt")
         sys.exit(1)

import tempfile
from vision_tower import VGGT_OriAny_Ref
from utils.paths import LOCAL_CKPT_PATH, HF_CKPT_PATH, RENDER_FILE

try:
    from utils.axis_renderer import BlendRenderer
    HAS_BLEND_RENDERER = True
    print("Using Blender for rendering.")
except ImportError as e:
    print(f"Bpy not found/compatible ({e}). Switching to Matplotlib fallback.")
    
    # --- Matplotlib Renderer Fallback ---
    import io
    try:
        import matplotlib.pyplot as plt
        # Ensure 3D plotting is enabled
        from mpl_toolkits.mplot3d import Axes3D
        
        class MatplotlibRenderer:
            def __init__(self, blend_file_path=None):
                pass

            def render_axis(self, azi, ele, rot, alpha=1, save_path="output.png"):
                # Convert to radians
                azi_rad = np.radians(-azi) 
                ele_rad = np.radians(ele)
                rot_rad = np.radians(rot)

                # Rz (around z)
                c, s = np.cos(azi_rad), np.sin(azi_rad)
                Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

                # Ry (around y)
                c, s = np.cos(ele_rad), np.sin(ele_rad)
                Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

                # Rx (around x)
                c, s = np.cos(rot_rad), np.sin(rot_rad)
                Rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

                R = Rx @ Ry @ Rz

                origin = np.array([0, 0, 0])
                # Blender axes often: X=Red, Y=Green, Z=Blue
                x_axis = R @ np.array([1, 0, 0])
                y_axis = R @ np.array([0, 1, 0])
                z_axis = R @ np.array([0, 0, 1])

                fig = plt.figure(figsize=(4, 4), dpi=100)
                ax = fig.add_subplot(111, projection='3d')
                ax.set_axis_off()
                
                # Determine limits to center view
                ax.set_xlim([-1.2, 1.2])
                ax.set_ylim([-1.2, 1.2])
                ax.set_zlim([-1.2, 1.2])
                
                # Draw axes
                ax.quiver(0, 0, 0, x_axis[0], x_axis[1], x_axis[2], color='r', length=1.0, normalize=True, linewidth=3)
                ax.quiver(0, 0, 0, y_axis[0], y_axis[1], y_axis[2], color='g', length=1.0, normalize=True, linewidth=3)
                ax.quiver(0, 0, 0, z_axis[0], z_axis[1], z_axis[2], color='b', length=1.0, normalize=True, linewidth=3)
                
                ax.view_init(elev=20, azim=-60)
                
                plt.savefig(save_path, transparent=True, bbox_inches='tight', pad_inches=0)
                plt.close(fig)

        # Assign the fallback class to the variable expected by main()
        BlendRenderer = MatplotlibRenderer
        HAS_BLEND_RENDERER = True
        
    except ImportError:
        print("Matplotlib not found either. Visualization disabled.")
        HAS_BLEND_RENDERER = False

def get_args():
    parser = argparse.ArgumentParser(description="Orient Anything V2 Inference Demo")
    parser.add_argument("--ref_image", type=str, required=True, help="Path to the reference image")
    parser.add_argument("--tgt_image", type=str, default=None, help="Path to the target image (optional)")
    parser.add_argument("--ckpt_path", type=str, default="../../4D-Data/OriAnyV2_ckpt/rotmod_realrotaug_best.pt", help="Path to the model checkpoint")
    parser.add_argument("--remove_bg", action="store_true", help="Remove background from images (requires rembg)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for inference")
    parser.add_argument("--download_ckpt", action="store_true", help="Force download checkpoint from HuggingFace")
    return parser.parse_args()

def main():
    args = get_args()
    
    print(f"=== Orient Anything V2 Inference ===")
    
    # --- 1. Load Model Checkpoint ---
    ckpt_path = args.ckpt_path
    if ckpt_path is None:
        if os.path.exists(LOCAL_CKPT_PATH) and not args.download_ckpt:
            ckpt_path = LOCAL_CKPT_PATH
            print(f"Using local checkpoint found at: {ckpt_path}")
        else:
            print(f"Local checkpoint not found at {LOCAL_CKPT_PATH} or download forced.")
            print("Attempting to download from HuggingFace (Viglong/OriAnyV2_ckpt)...")
            try:
                ckpt_path = hf_hub_download(
                    repo_id="Viglong/OriAnyV2_ckpt", 
                    filename=HF_CKPT_PATH, 
                    repo_type="model", 
                    cache_dir='./', 
                    resume_download=True
                )
                print(f"Downloaded checkpoint to: {ckpt_path}")
            except Exception as e:
                print(f"Error downloading checkpoint: {e}")
                print("Please check your internet connection or provide a valid --ckpt_path")
                return

    # --- 2. Initialize Model ---
    print(f"Initializing model on {args.device}...")
    dtype = torch.bfloat16 if args.device.startswith("cuda") and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    try:
        model = VGGT_OriAny_Ref(out_dim=900, dtype=dtype, nopretrain=True)
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        model.eval()
        model = model.to(args.device)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model architecture or weights: {e}")
        traceback.print_exc()
        return

    # --- 3. Load and Preprocess Images ---
    if not os.path.exists(args.ref_image):
        print(f"Error: Reference image file not found: {args.ref_image}")
        return
    
    print(f"Loading reference image: {args.ref_image}")
    try:
        pil_ref = Image.open(args.ref_image).convert("RGB")
    except Exception as e:
        print(f"Error opening reference image: {e}")
        return
    
    pil_tgt = None
    if args.tgt_image:
        if not os.path.exists(args.tgt_image):
            print(f"Error: Target image file not found: {args.tgt_image}")
            return
        print(f"Loading target image: {args.tgt_image}")
        try:
            pil_tgt = Image.open(args.tgt_image).convert("RGB")
        except Exception as e:
            print(f"Error opening target image: {e}")
            return

    if args.remove_bg:
        print("Removing background from images...")
        try:
            pil_ref = background_preprocess(pil_ref, True)
            if pil_tgt:
                pil_tgt = background_preprocess(pil_tgt, True)
        except Exception as e:
            print(f"Error removing background: {e}")
            print("Ensure 'rembg' is installed properly.")
            return

    # --- 4. Run Inference ---
    print("Running inference...")
    try:
        # inf_single_case returns a dictionary of results
        ans_dict = inf_single_case(model, pil_ref, pil_tgt)
        
        print("\n" + "="*30)
        print("INFERENCE RESULTS")
        print("="*30)
        
        # Display results nicely
        def safe_float(val, default=0.0):
            try:
                if hasattr(val, 'item'):
                    return val.item()
                return float(val)
            except:
                return float(default)

        # Get values
        az = safe_float(ans_dict.get('ref_az_pred', 0))
        el = safe_float(ans_dict.get('ref_el_pred', 0))
        ro = safe_float(ans_dict.get('ref_ro_pred', 0))
        alpha_val = int(safe_float(ans_dict.get('ref_alpha_pred', 1)))

        print(f"Reference Azimuth (Az):   {az:.2f}")
        print(f"Reference Elevation (El): {el:.2f}")
        print(f"Reference Rotation (Ro):  {ro:.2f}")
        print(f"Prediction Alpha:         {alpha_val}")
        
        if pil_tgt:
            rel_az = safe_float(ans_dict.get('rel_az_pred', 0))
            rel_el = safe_float(ans_dict.get('rel_el_pred', 0))
            rel_ro = safe_float(ans_dict.get('rel_ro_pred', 0))

            print("-" * 20)
            print(f"Relative Azimuth:       {rel_az:.2f}")
            print(f"Relative Elevation:     {rel_el:.2f}")
            print(f"Relative Rotation:      {rel_ro:.2f}")
            
        print("="*30)
        
        # --- Visualization ---
        if HAS_BLEND_RENDERER:
             if os.path.exists(RENDER_FILE):
                 print("\nRunning Blender visualization...")
                 try:
                     # Initialize renderer
                     renderer = BlendRenderer(RENDER_FILE)
                     
                     # Create temp file for output
                     with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                         tmp_path = tmp_file.name
                     
                     print("Rendering axis...")
                     renderer.render_axis(az, el, ro, alpha_val, save_path=tmp_path)
                     
                     if os.path.exists(tmp_path):
                         # Load rendered axis
                         axis_img = Image.open(tmp_path).convert("RGBA")
                         
                         # Resize original image to match axis size (512x512 from blender)
                         # This preserves transparency correctly
                         if axis_img.size != pil_ref.size:
                             pil_ref_resized = pil_ref.resize(axis_img.size, Image.BICUBIC)
                         else:
                             pil_ref_resized = pil_ref
                             
                         # Overlay
                         final_ref_img = Image.alpha_composite(pil_ref_resized.convert("RGBA"), axis_img).convert("RGB")
                         
                         out_name = "output_ref.png"
                         final_ref_img.save(out_name)
                         print(f"Visualization saved to: {os.path.abspath(out_name)}")
                         
                         # Clean up
                         os.remove(tmp_path)
                     else:
                         print("Error: Blender failed to produce output image.")

                     if pil_tgt:
                         # Calculate absolute target rotation (assuming Get_target_azi_ele_rot handles 3 inputs)
                         # The function from app_utils takes (azi, ele, rot, rel_azi, rel_ele, rel_rot)
                         # Note: Ensure these inputs are floats or tensors as expected.
                         # Based on app.py usage: tgt_azi, tgt_ele, tgt_rot = Get_target_azi_ele_rot(az, el, ro, rel_az, rel_el, rel_ro)
                         
                         tgt_azi, tgt_ele, tgt_rot = Get_target_azi_ele_rot(az, el, ro, rel_az, rel_el, rel_ro)
                         
                         try:
                             # Convert if tensor
                             if hasattr(tgt_azi, 'item'): tgt_azi = tgt_azi.item()
                             if hasattr(tgt_ele, 'item'): tgt_ele = tgt_ele.item()
                             if hasattr(tgt_rot, 'item'): tgt_rot = tgt_rot.item()
                         except:
                             pass

                         print(f"Calculated Target Pose: Az={tgt_azi:.2f}, El={tgt_ele:.2f}, Ro={tgt_rot:.2f}")

                         with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                             tmp_path_tgt = tmp_file.name
                             
                         # Render target axis (alpha default to 1)
                         renderer.render_axis(tgt_azi, tgt_ele, tgt_rot, alpha=1, save_path=tmp_path_tgt)

                         if os.path.exists(tmp_path_tgt):
                             axis_img_tgt = Image.open(tmp_path_tgt).convert("RGBA")
                             
                             if axis_img_tgt.size != pil_tgt.size:
                                 pil_tgt_resized = pil_tgt.resize(axis_img_tgt.size, Image.BICUBIC)
                             else:
                                 pil_tgt_resized = pil_tgt
                            
                             final_tgt_img = Image.alpha_composite(pil_tgt_resized.convert("RGBA"), axis_img_tgt).convert("RGB")
                             out_name_tgt = "output_tgt.png"
                             final_tgt_img.save(out_name_tgt)
                             print(f"Target Visualization saved to: {os.path.abspath(out_name_tgt)}")
                             os.remove(tmp_path_tgt)
                         else:
                             print("Error: Blender failed to render target axis.")

                 except Exception as e:
                     print(f"Visualization failed: {e}")
                     traceback.print_exc()
             else:
                 print(f"Warning: Render file not found at {RENDER_FILE}. Skipping visualization.")
        else:
            print("Note: Install 'bpy' to enable 3D axis visualization.")

    except Exception as e:
        print(f"Inference process failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
