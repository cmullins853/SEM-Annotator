import os
import cv2
import numpy as np
import argparse
from glob import glob
from tqdm import tqdm
from PIL import Image

def create_base_mask(image_shape, label_path):
    """
    Creates a binary mask from YOLO segmentation labels.
    
    Args:
        image_shape: (height, width)
        label_path: Path to the YOLO .txt file.
    Returns:
        numpy array: Binary mask (0 and 255).
    """
    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    if os.path.exists(label_path):
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            
            coords = list(map(float, parts[1:]))
            points = np.array(coords).reshape(-1, 2)
            points[:, 0] *= width
            points[:, 1] *= height
            points = points.astype(np.int32)
            
            cv2.fillPoly(mask, [points], 255)
    return mask

def main():
    parser = argparse.ArgumentParser(
        description="Convert YOLO segmentation labels to binary masks."
    )
    parser.add_argument("--image-dir", required=True, 
                        help="Path to directory containing source images.")
    parser.add_argument("--label-dir", required=True, 
                        help="Path to directory containing YOLO segmentation .txt labels.")
    parser.add_argument("--output-dir", required=True, 
                        help="Path to directory where masks will be saved.")

    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    image_paths = sorted(
        glob(os.path.join(args.image_dir, "*.png")) +
        glob(os.path.join(args.image_dir, "*.jpg")) +
        glob(os.path.join(args.image_dir, "*.jpeg")) +
        glob(os.path.join(args.image_dir, "*.tif")) +
        glob(os.path.join(args.image_dir, "*.tiff"))
    )
    
    if not image_paths:
        print(f"No images found in {args.image_dir}")
        return
        
    print(f"Found {len(image_paths)} images. Processing...")
    
    for img_path in tqdm(image_paths):
        basename = os.path.basename(img_path)
        name_no_ext, ext = os.path.splitext(basename)
        
        label_path = os.path.join(args.label_dir, f"{name_no_ext}.txt")
        
        try:
            # Load original image to get shape
            img_pil = Image.open(img_path)
            # Efficiently get shape without loading full pixels if needed, 
            # but create_base_mask needs it. 
            width, height = img_pil.size
            
            # Create mask
            mask = create_base_mask((height, width), label_path)
            
            # Save mask
            mask_out_path = os.path.join(args.output_dir, basename)
            Image.fromarray(mask).save(mask_out_path)
            
        except Exception as e:
            print(f"\nError processing {basename}: {e}")

    print(f"\nDone! Masks saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
