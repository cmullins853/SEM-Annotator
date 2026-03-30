import argparse
import os
import numpy as np
import cv2
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import rasterio

# Disable DecompressionBombError for large images
Image.MAX_IMAGE_PIXELS = None

def get_scaling_params(arr):
    """Calculates percentile-based scaling parameters for an array."""
    valid_mask = (arr > 0) & (arr < 65535) 
    if not np.any(valid_mask):
        return np.min(arr), np.max(arr)
    
    p2, p98 = np.percentile(arr[valid_mask], (2, 98))
    return p2, p98

def scale_to_u8(arr, p2, p98):
    """Scales a numpy array to uint8 using provided percentile bounds."""
    if p98 <= p2:
        return np.clip(arr, 0, 255).astype(np.uint8)
    scaled = np.clip((arr.astype(np.float32) - p2) / (p98 - p2) * 255, 0, 255)
    return scaled.astype(np.uint8)

def prepare_mask(mask_arr):
    """Normalizes a mask into a single-channel uint8 array without changing tile geometry."""
    mask = np.asarray(mask_arr)
    mask = np.squeeze(mask)

    if mask.ndim == 3:
        channel_count = mask.shape[2]
        if channel_count == 4:
            rgb = mask[..., :3]
            alpha = mask[..., 3]
            if np.any(alpha) and not np.any(rgb):
                mask = alpha
            elif np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 1], rgb[..., 2]):
                mask = rgb[..., 0]
            else:
                mask = np.max(rgb, axis=-1)
        elif channel_count >= 3:
            rgb = mask[..., :3]
            if np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 1], rgb[..., 2]):
                mask = rgb[..., 0]
            else:
                mask = np.max(rgb, axis=-1)
        else:
            mask = mask[..., 0]

    if mask.ndim != 2:
        raise ValueError(f"Unsupported mask shape: {mask_arr.shape}")

    if mask.dtype == np.bool_:
        return mask.astype(np.uint8) * 255

    if np.issubdtype(mask.dtype, np.floating):
        mask = np.nan_to_num(mask, nan=0.0)

    max_val = float(np.max(mask)) if mask.size else 0.0
    unique_vals = np.unique(mask)

    if max_val <= 1.0:
        return (mask > 0).astype(np.uint8) * 255

    if len(unique_vals) <= 2:
        return (mask > 0).astype(np.uint8) * 255

    if max_val > 255:
        return np.where(mask > 0, 255, 0).astype(np.uint8)

    return np.clip(mask, 0, 255).astype(np.uint8)

def load_image_info(img_path):
    """Loads image robustly, preserving actual depth and channels."""
    if img_path.lower().endswith(('.tif', '.tiff')):
        with rasterio.open(img_path) as src:
            if src.count >= 3:
                arr = src.read([1, 2, 3])
                arr = np.transpose(arr, (1, 2, 0))
                return arr, True
            else:
                arr = src.read(1)
                return arr, False
    else:
        # Use cv2 to load PNG/JPG safely preserving 16-bit depth
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            # Fallback
            img = np.array(Image.open(img_path))
            
        if len(img.shape) == 3:
            # If loaded as BGR with cv2, convert to RGB
            if img.shape[2] == 3:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), True
            if img.shape[2] == 4:
                return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA), True
        return img, False

def process_images_aligned(img_path, mask_path=None, tile_size=1024):
    if not os.path.exists(img_path):
        print(f"Error: Image not found: {img_path}")
        return False

    print(f"--- Starting Final Aligned Tiling ---")
    print(f"Loading image: {os.path.basename(img_path)}")
    img_np, is_rgb = load_image_info(img_path)
    h, w = img_np.shape[:2]
    print(f"Image Resolution: {w}x{h}, Channels: {img_np.shape[2] if is_rgb else 1}, Dtype: {img_np.dtype}")

    print("Calculating scaling parameters for visualization...")
    if is_rgb:
        scaling_params = [get_scaling_params(img_np[..., i]) for i in range(3)]
    else:
        scaling_params = [get_scaling_params(img_np)]

    mask_np = None
    if mask_path:
        print(f"Loading mask: {os.path.basename(mask_path)}")
        mask_np, _ = load_image_info(mask_path)
        if mask_np.shape[:2] != (h, w):
            print(f"Warning: Mask shape differs from image shape. Resizing.")
            mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_np = prepare_mask(mask_np)

    base_dir = os.path.dirname(os.path.abspath(img_path))
    output_root = os.path.join(base_dir, "tiles_output")
    img_out_dir = os.path.join(output_root, "images")
    mask_out_dir = os.path.join(output_root, "masks")
    
    os.makedirs(img_out_dir, exist_ok=True)
    if mask_np is not None:
        os.makedirs(mask_out_dir, exist_ok=True)
    
    img_base_name = os.path.splitext(os.path.basename(img_path))[0]

    num_cols = (w + tile_size - 1) // tile_size
    num_rows = (h + tile_size - 1) // tile_size
    total_tiles = num_rows * num_cols
    
    print(f"Tiling into {total_tiles} patches...")

    count = 0
    saved_count = 0
    for r in range(num_rows):
        for c in range(num_cols):
            count += 1
            left = c * tile_size
            top = r * tile_size
            right = min(left + tile_size, w)
            bottom = min(top + tile_size, h)
            
            img_tile_raw = img_np[top:bottom, left:right].copy()
            
            # nodata mask ensures perfectly black areas from stitch padding remain 0
            if is_rgb:
                nodata_mask = np.all(img_tile_raw == 0, axis=-1)
            else:
                nodata_mask = (img_tile_raw == 0)

            # Scale to 8-bit output
            if is_rgb:
                scaled_bands = []
                for i in range(3):
                    p2, p98 = scaling_params[i]
                    scaled_bands.append(scale_to_u8(img_tile_raw[..., i], p2, p98))
                img_tile_u8 = np.stack(scaled_bands, axis=-1)
            else:
                p2, p98 = scaling_params[0]
                img_tile_u8 = scale_to_u8(img_tile_raw, p2, p98)

            img_tile_u8[nodata_mask] = 0

            tile_filename = f"{img_base_name}_tile_{r}_{c}.png"
            img_out_path = os.path.join(img_out_dir, tile_filename)
            Image.fromarray(img_tile_u8).save(img_out_path)

            if mask_np is not None:
                mask_tile_u8 = mask_np[top:bottom, left:right].copy()
                mask_out_path = os.path.join(mask_out_dir, tile_filename)
                Image.fromarray(mask_tile_u8).save(mask_out_path)
            
            saved_count += 1
            if count % 10 == 0 or count == total_tiles:
                print(f" Progress: {count}/{total_tiles} tiles processed.", end='\r')

    print(f"\nProcessing Complete. Saved {saved_count} pairs to: {output_root}")
    return True

def process_directory(img_dir, mask_dir=None, tile_size=1024):
    if not os.path.exists(img_dir):
        print(f"Error: Directory not found: {img_dir}")
        return
    
    valid_exts = {'.tif', '.tiff', '.png', '.jpg', '.jpeg'}
    img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) 
                 if os.path.isfile(os.path.join(img_dir, f)) and os.path.splitext(f)[1].lower() in valid_exts]
                 
    if not img_paths:
        print(f"No valid images found in {img_dir}")
        return
        
    print(f"Found {len(img_paths)} images to process in {img_dir}\n")
    for img_path in sorted(img_paths):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = None
        
        if mask_dir and os.path.exists(mask_dir):
            potential_masks = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) 
                               if os.path.splitext(f)[0] == base_name and os.path.isfile(os.path.join(mask_dir, f))]
            if potential_masks:
                mask_path = potential_masks[0]
                
        process_images_aligned(img_path, mask_path, tile_size)
        print("-" * 40)

def run_gui():
    root = tk.Tk()
    root.withdraw()
    img_dir = filedialog.askdirectory(title="Select Image Directory")
    if not img_dir: return
    mask_dir = filedialog.askdirectory(title="Select Mask Directory (Cancel if none, or match names in folder)")
    if not mask_dir: mask_dir = None
    process_directory(img_dir, mask_dir)
    root.destroy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Directory-Based Robust Aligned Tiling")
    parser.add_argument("--img_dir", type=str, help="Path to directory containing images")
    parser.add_argument("--mask_dir", type=str, help="Path to directory containing masks")
    parser.add_argument("--tile_size", type=int, default=1024)
    parser.add_argument("--no-gui", action="store_true")
    args = parser.parse_args()
    
    if args.no_gui:
        if not args.img_dir:
            print("Error: --img_dir is required in no-gui mode")
        else:
            process_directory(args.img_dir, args.mask_dir, args.tile_size)
    else:
        if args.img_dir: 
            process_directory(args.img_dir, args.mask_dir, args.tile_size)
        else: 
            run_gui()
