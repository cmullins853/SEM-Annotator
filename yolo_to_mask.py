import os
import cv2
import numpy as np
import argparse
from glob import glob
from tqdm import tqdm
from PIL import Image
import albumentations as A

def get_aug_pipeline(aug_count=5, flip_h=True, flip_v=True, blur=True, color=True, 
                     blur_limit=(3, 7), sat_limit=30, hue_limit=20, val_limit=20):
    """
    Returns an albumentations composition based on user-selected toggles and intensity limits.
    Probabilities are dynamic based on aug_count: p = min(0.8, 2.0 / max(1, aug_count)).
    """
    # Proportional probability (e.g. if aug_count=2, p=1.0; if aug_count=5, p=0.4)
    # Clamp between 0.1 and 0.8 to ensure some variety but not total chaos.
    p = max(0.1, min(0.8, 2.0 / max(1, aug_count)))
    
    transforms = []
    if flip_h:
        transforms.append(A.HorizontalFlip(p=p))
    if flip_v:
        transforms.append(A.VerticalFlip(p=p))
    if blur:
        # Use slightly lower p for blur as it's more destructive
        transforms.append(A.GaussianBlur(blur_limit=blur_limit, p=p * 0.75))
    if color:
        # Split color into and contrast/brightness vs hue/sat/val
        transforms.append(A.RandomBrightnessContrast(brightness_limit=val_limit/255.0, contrast_limit=val_limit/255.0, p=p))
        transforms.append(A.HueSaturationValue(hue_shift_limit=hue_limit, sat_shift_limit=sat_limit, val_shift_limit=val_limit, p=p))
    
    return A.Compose(transforms)

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
        description="Convert YOLO segmentation labels to binary masks with configurable offline augmentation."
    )
    parser.add_argument("--image-dir", required=True, 
                        help="Path to directory containing source images.")
    parser.add_argument("--label-dir", required=True, 
                        help="Path to directory containing YOLO segmentation .txt labels.")
    parser.add_argument("--output-dir", required=True, 
                        help="Path to directory where masks will be saved.")
    
    # Global Augmentation toggle
    parser.add_argument("--augment", action="store_true",
                        help="Enable offline data augmentation (generates augmented image-mask pairs).")
    parser.add_argument("--aug-count", type=int, default=5,
                        help="Number of augmented variations to generate per image.")
    parser.add_argument("--aug-img-out", default=None,
                        help="Directory to save augmented images. If None, saves to a subdirectory in image-dir.")

    # Individual Augmentation Toggles
    parser.add_argument("--aug-flip-h", action="store_true", default=False, help="Enable Horizontal Flip.")
    parser.add_argument("--aug-flip-v", action="store_true", default=False, help="Enable Vertical Flip.")
    parser.add_argument("--aug-blur",   action="store_true", default=False, help="Enable Gaussian Blur.")
    parser.add_argument("--aug-color",  action="store_true", default=False, help="Enable Brightness/Contrast/Hue/Saturation.")
    parser.add_argument("--aug-all",    action="store_true", default=False, help="Enable all augmentations.")

    # Intensity Limits
    parser.add_argument("--aug-blur-limit", type=int, default=3, help="Maximum kernel size for Gaussian blur (must be odd, e.g., 3, 5, 7).")
    parser.add_argument("--aug-sat-limit",  type=int, default=10, help="Maximum saturation shift limit.")
    parser.add_argument("--aug-hue-limit",  type=int, default=0, help="Maximum hue shift limit.")
    parser.add_argument("--aug-val-limit",  type=int, default=0, help="Maximum value (brightness) shift limit.")

    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    flip_h = args.aug_flip_h or args.aug_all
    flip_v = args.aug_flip_v or args.aug_all
    blur   = args.aug_blur   or args.aug_all
    color  = args.aug_color  or args.aug_all

    if args.augment and not (flip_h or flip_v or blur or color):
        print("Warning: --augment set but no specific augmentation flags enabled. Enabling all by default.")
        flip_h = flip_v = blur = color = True

    # Ensure blur_limit is odd and >= 3
    blur_max = max(3, args.aug_blur_limit)
    if blur_max % 2 == 0:
        blur_max += 1
    blur_limit = (3, blur_max)

    aug_img_dir = args.aug_img_out
    if args.augment and aug_img_dir is None:
        aug_img_dir = os.path.join(args.image_dir, "augmented")
    
    if args.augment:
        os.makedirs(aug_img_dir, exist_ok=True)
        aug_pipeline = get_aug_pipeline(
            aug_count=args.aug_count,
            flip_h=flip_h, flip_v=flip_v, blur=blur, color=color,
            blur_limit=blur_limit, sat_limit=args.aug_sat_limit,
            hue_limit=args.aug_hue_limit, val_limit=args.aug_val_limit
        )
        print(f"Augmentation enabled: Generating {args.aug_count} samples per image.")
        print(f"Base probability (p) = {max(0.1, min(0.8, 2.0 / max(1, args.aug_count))):.2f}")
        print(f"Enabled transforms: FlipH={flip_h}, FlipV={flip_v}, Blur={blur} (limit={blur_limit}), Color={color}")
        print(f"Color limits: Saturation={args.aug_sat_limit}, Hue={args.aug_hue_limit}, Value={args.aug_val_limit}")
        print(f"Augmented images will be saved to: {aug_img_dir}")

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
            # Load original image
            img_pil = Image.open(img_path).convert("RGB")
            img = np.array(img_pil)
            
            # Create base mask
            mask = create_base_mask(img.shape, label_path)
            
            # Save original mask
            mask_out_path = os.path.join(args.output_dir, basename)
            Image.fromarray(mask).save(mask_out_path)
            
            if args.augment:
                # Also save original image to augmented output dir
                Image.fromarray(img).save(os.path.join(aug_img_dir, basename))
                
                for i in range(1, args.aug_count + 1):
                    augmented = aug_pipeline(image=img, mask=mask)
                    aug_img = augmented['image']
                    aug_mask = augmented['mask']
                    
                    aug_name = f"{name_no_ext}_aug{i}{ext}"
                    
                    # Save augmented image
                    Image.fromarray(aug_img).save(os.path.join(aug_img_dir, aug_name))
                    # Save augmented mask
                    Image.fromarray(aug_mask).save(os.path.join(args.output_dir, aug_name))
                    
        except Exception as e:
            print(f"\nError processing {basename}: {e}")

    print(f"\nDone! Masks saved to: {args.output_dir}")
    if args.augment:
        print(f"Augmented images saved to: {aug_img_dir}")

if __name__ == "__main__":
    main()
