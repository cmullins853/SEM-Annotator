import cv2
import numpy as np
import os
import argparse
import glob

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

def main():
    parser = argparse.ArgumentParser(description="Interactive Background Remover for Grayscale Images")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--ext", type=str, default="*.png", help="Image extension (e.g., *.png, *.tif, *.jpg)")
    parser.add_argument("--save_dir", type=str, default="output", help="Directory to save processed images")
    parser.add_argument("--width", type=int, default=800, help="Target width for each image window side")
    # Added method argument as requested
    parser.add_argument("--method", type=str, choices=["inrange", "binary", "tozero", "otsu"], 
                        default="inrange", help="Thresholding method to start with")
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    image_paths = glob.glob(os.path.join(args.dir, args.ext))
    if not image_paths:
        print(f"No images found in {args.dir} with extension {args.ext}")
        return

    image_paths.sort()
    current_idx = 0
    
    cv2.namedWindow("Background Remover", cv2.WINDOW_AUTOSIZE)
    
    # Method mapping for trackbar
    methods = ["inrange", "binary", "tozero", "otsu"]
    initial_mode = methods.index(args.method)
    
    cv2.createTrackbar("Lower", "Background Remover", 10, 255, nothing)
    cv2.createTrackbar("Upper", "Background Remover", 255, 255, nothing)
    cv2.createTrackbar("Bottom %", "Background Remover", 10, 50, nothing)
    cv2.createTrackbar("Mode", "Background Remover", initial_mode, 3, nothing)
    
    print("\n--- Controls ---")
    print("'n': Next | 'p': Prev | 's': Save | 'a': ALL | 'q': Quit")
    
    while True:
        img_path = image_paths[current_idx]
        original_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        
        if original_img is None:
            print(f"Failed to load {img_path}")
            current_idx = (current_idx + 1) % len(image_paths)
            continue
            
        if len(original_img.shape) == 3:
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
            
        is_16bit = (original_img.dtype == np.uint16)
        max_val = 65535 if is_16bit else 255
        
        display_base = (original_img / 256).astype(np.uint8) if is_16bit else original_img.copy()
        display_base_small, scale = resize_to_fit(display_base, target_width=args.width)
        h_small, w_small = display_base_small.shape
        
        print(f"Current: {os.path.basename(img_path)} ({original_img.dtype}, {original_img.shape})")

        while True:
            # Get settings
            lower_v = cv2.getTrackbarPos("Lower", "Background Remover")
            upper_v = cv2.getTrackbarPos("Upper", "Background Remover")
            mode_v = cv2.getTrackbarPos("Mode", "Background Remover")
            crop_v = cv2.getTrackbarPos("Bottom %", "Background Remover")
            
            l_actual = int(lower_v * (max_val / 255.0))
            u_actual = int(upper_v * (max_val / 255.0))
            method_name = methods[mode_v]
            
            def apply_threshold(img, method, low, high, mx_v, crop_pct):
                h, w = img.shape[:2]
                crop_y = int(h * (1 - crop_pct / 100.0))
                
                # Apply threshold to the whole image first
                if method == "inrange":
                    mask = cv2.inRange(img, low, high)
                    res = cv2.bitwise_and(img, img, mask=mask)
                elif method == "binary":
                    mask = cv2.inRange(img, low, high)
                    res = mask
                elif method == "tozero":
                    _, res = cv2.threshold(img, low, mx_v, cv2.THRESH_TOZERO)
                    if low < high < 255:
                        mask = cv2.inRange(img, 0, high)
                        res = cv2.bitwise_and(res, res, mask=mask)
                elif method == "otsu":
                    # Use only top portion for Otsu calculation
                    top = img[:crop_y, :]
                    if img.dtype == np.uint16:
                        top_8 = (top / 256).astype(np.uint8)
                        val, _ = cv2.threshold(top_8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    else:
                        val, _ = cv2.threshold(top, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    mask = cv2.inRange(img, int(val * (mx_v/255.0)), mx_v)
                    res = cv2.bitwise_and(img, img, mask=mask)
                
                # Black out bottom part
                res[crop_y:, :] = 0
                return res

            processed_img = apply_threshold(original_img, method_name, l_actual, u_actual, max_val, crop_v)

            if is_16bit:
                proc_display = (processed_img / (max_val/255.0)).astype(np.uint8)
            else:
                proc_display = processed_img
            
            proc_display_small = cv2.resize(proc_display, (w_small, h_small), interpolation=cv2.INTER_AREA)
            
            canvas = np.zeros((h_small, w_small * 2), dtype=np.uint8)
            canvas[:, :w_small] = display_base_small
            canvas[:, w_small:] = proc_display_small
            
            cv2.putText(canvas, f"METHOD: {method_name.upper()}", (10, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255, 1)
            cv2.putText(canvas, f"PARAMS: {l_actual}-{u_actual}", (w_small + 10, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255, 1)
            
            cv2.imshow("Background Remover", canvas)
            
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
            save_path = os.path.join(args.save_dir, "proc_" + base_name + ".png")
            cv2.imwrite(save_path, processed_img)
            print(f"Saved: {save_path}")
        elif key == ord('a'):
            print(f"Processing all {len(image_paths)} images...")
            for p in tqdm(image_paths):
                img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
                if img is None: continue
                if len(img.shape) == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                mx = 65535 if img.dtype == np.uint16 else 255
                res = apply_threshold(img, method_name, l_actual, u_actual, mx, crop_v)
                
                base_name = os.path.splitext(os.path.basename(p))[0]
                s_path = os.path.join(args.save_dir, "proc_" + base_name + ".png")
                cv2.imwrite(s_path, res)
            print("Batch processing complete.")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
