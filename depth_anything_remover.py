import cv2
import numpy as np
import os
import argparse
import glob
import torch
from tqdm import tqdm

try:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

def nothing(x):
    pass

def resize_to_fit(img, target_width=800, target_height=800):
    h, w = img.shape[:2]
    scale = min(target_width / w, target_height / h)
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA), scale
    return img, 1.0

class DepthAnythingPredictor:
    def __init__(self, model_id="depth-anything/Depth-Anything-V2-Small-hf"):
        if not HAS_TRANSFORMERS:
            raise ImportError("Please install 'transformers' and 'accelerate' to use Depth Anything V2.")
        
        print(f"Loading {model_id}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id).to(self.device).eval()
        print(f"Model loaded on {self.device}")

    @torch.no_grad()
    def predict(self, img_bgr):
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=img_rgb, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        
        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            outputs.predicted_depth.unsqueeze(1),
            size=img_bgr.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        
        depth = prediction.cpu().numpy()
        # Normalize to 0-255 for thresholding
        depth_min, depth_max = depth.min(), depth.max()
        if depth_max - depth_min > 1e-5:
            depth_norm = (depth - depth_min) / (depth_max - depth_min) * 255.0
        else:
            depth_norm = depth * 0
            
        return depth_norm.astype(np.uint8)

def main():
    parser = argparse.ArgumentParser(description="Interactive Depth-Based Background Remover (Depth Anything V2)")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--ext", type=str, default="*.png", help="Image extension (e.g., *.png, *.tif, *.jpg)")
    parser.add_argument("--save_dir", type=str, default="output_depth", help="Directory to save processed images")
    parser.add_argument("--width", type=int, default=800, help="Target width for display")
    parser.add_argument("--model", type=str, default="depth-anything/Depth-Anything-V2-Small-hf", help="HF Model ID")
    args = parser.parse_args()

    if not HAS_TRANSFORMERS:
        print("Error: 'transformers' library not found. Run: pip install transformers accelerate")
        return

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    image_paths = sorted(glob.glob(os.path.join(args.dir, args.ext)))
    if not image_paths:
        print(f"No images found in {args.dir} with extension {args.ext}")
        return

    # Initialize model
    predictor = DepthAnythingPredictor(args.model)
    
    current_idx = 0
    cv2.namedWindow("Depth Background Remover", cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar("Depth Cutoff", "Depth Background Remover", 127, 255, nothing)
    cv2.createTrackbar("Label %", "Depth Background Remover", 12, 50, nothing)
    cv2.createTrackbar("Invert", "Depth Background Remover", 0, 1, nothing)
    
    print("\n--- Controls ---")
    print("'n': Next | 'p': Prev | 's': Save | 'a': Process ALL | 'q': Quit")
    
    # Cache depth maps: {(img_path, crop_pct): depth_map}
    depth_cache = {}

    def get_processed(original_img, img_path, cutoff_val, inv_val, crop_pct):
        h, w = original_img.shape[:2]
        crop_y = int(h * (1 - crop_pct / 100.0))
        
        # Cache based on path AND crop since crop changes the model input
        cache_key = (img_path, crop_pct)

        if cache_key not in depth_cache:
            # Prepare image slice for model
            img_slice = original_img[:crop_y, :]
            if len(img_slice.shape) == 2:
                img_slice = cv2.cvtColor(img_slice, cv2.COLOR_GRAY2BGR)
            if img_slice.dtype == np.uint16:
                img_slice = (img_slice / 256).astype(np.uint8)
            
            # Predict only on the non-label portion
            d_slice = predictor.predict(img_slice)
            
            # Pad back to original size
            d_full = np.zeros((h, w), dtype=np.uint8)
            d_full[:crop_y, :] = d_slice
            depth_cache[cache_key] = d_full
        
        d_map = depth_cache[cache_key]
        
        if inv_val:
            mask = (d_map > cutoff_val).astype(np.uint8) * 255
        else:
            # We must also ensure the bottom part is masked out as background
            mask = (d_map < cutoff_val).astype(np.uint8) * 255
            mask[crop_y:, :] = 0
            
        return cv2.bitwise_and(original_img, original_img, mask=mask), d_map

    while True:
        img_path = image_paths[current_idx]
        original_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        
        if original_img is None:
            print(f"Failed to load {img_path}")
            current_idx = (current_idx + 1) % len(image_paths)
            continue
            
        display_base = (original_img / 256).astype(np.uint8) if original_img.dtype == np.uint16 else original_img
        if len(display_base.shape) == 2: display_base = cv2.cvtColor(display_base, cv2.COLOR_GRAY2BGR)
        
        display_img_small, scale = resize_to_fit(display_base, target_width=args.width)
        h_small, w_small = display_img_small.shape[:2]

        while True:
            cutoff = cv2.getTrackbarPos("Depth Cutoff", "Depth Background Remover")
            invert = cv2.getTrackbarPos("Invert", "Depth Background Remover")
            crop = cv2.getTrackbarPos("Label %", "Depth Background Remover")
            
            processed_img, depth_map = get_processed(original_img, img_path, cutoff, invert, crop)
            
            # Display logic
            proc_display = (processed_img / 256).astype(np.uint8) if original_img.dtype == np.uint16 else processed_img
            if len(proc_display.shape) == 2: proc_display = cv2.cvtColor(proc_display, cv2.COLOR_GRAY2BGR)
            
            proc_small = cv2.resize(proc_display, (w_small, h_small), interpolation=cv2.INTER_AREA)
            
            depth_map_small = cv2.resize(depth_map, (w_small, h_small), interpolation=cv2.INTER_AREA)
            depth_vis = cv2.applyColorMap(depth_map_small, cv2.COLORMAP_MAGMA)
            
            canvas = np.zeros((h_small, w_small * 3, 3), dtype=np.uint8)
            canvas[:, :w_small] = display_img_small
            canvas[:, w_small:w_small*2] = depth_vis
            canvas[:, w_small*2:] = proc_small
            
            cv2.putText(canvas, "ORIGINAL", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(canvas, "TOPOGRAPHY (DEPTH)", (w_small + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(canvas, f"PROCESSED (Cut:{cutoff})", (w_small*2 + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            cv2.imshow("Depth Background Remover", canvas)
            
            key = cv2.waitKey(30) & 0xFF
            if key in [ord('q'), ord('n'), ord('p'), ord('s'), ord('a')]:
                break
        
        if key == ord('q'):
            break
        elif key == ord('n'):
            current_idx = (current_idx + 1) % len(image_paths)
        elif key == ord('p'):
            current_idx = (current_idx - 1) % len(image_paths)
        elif key == ord('s'):
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            save_path = os.path.join(args.save_dir, "depth_" + base_name + ".png")
            cv2.imwrite(save_path, processed_img)
            print(f"Saved: {save_path}")
        elif key == ord('a'):
            print(f"Processing all {len(image_paths)} images...")
            for p in tqdm(image_paths):
                img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
                if img is None: continue
                res, _ = get_processed(img, p, cutoff, invert, crop)
                
                base_name = os.path.splitext(os.path.basename(p))[0]
                s_path = os.path.join(args.save_dir, "depth_" + base_name + ".png")
                cv2.imwrite(s_path, res)
            print("Batch processing complete.")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
