import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from model.sam import SHFSAMEncoder
from SHF.shf import FixedQuadTree

try:
    from depth_anything_remover import DepthAnythingPredictor
    HAS_DEPTH_ANYTHING = True
except ImportError:
    HAS_DEPTH_ANYTHING = False
from sam_polygon_diameters import (
    compute_metrics,
    draw_measurements,
    extract_contours,
    load_mask,
    resolve_methods,
    serialize_instance_summaries,
    serialize_measurement_instances,
    serialize_measurement_rows,
    serialize_rows,
    threshold_mask,
)


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def get_scaling_params(arr):
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.nan_to_num(arr, nan=0.0)
    valid = arr > 0
    if np.any(valid):
        return np.percentile(arr[valid], (2, 98))
    return float(np.min(arr)), float(np.max(arr))


def scale_to_u8(arr, p2, p98):
    if p98 <= p2:
        return np.clip(arr, 0, 255).astype(np.uint8)
    scaled = np.clip((arr.astype(np.float32) - p2) / (p98 - p2) * 255.0, 0, 255)
    return scaled.astype(np.uint8)


def load_image_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        img = np.array(Image.open(path))
    if img is None:
        raise ValueError(f"Could not read image: {path}")

    if img.ndim == 2:
        p2, p98 = get_scaling_params(img)
        mono = scale_to_u8(img, p2, p98) if img.dtype != np.uint8 else img.astype(np.uint8)
        return np.stack([mono, mono, mono], axis=-1)

    if img.ndim != 3:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        rgb = img[..., :3]
    else:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if rgb.dtype == np.uint8:
        return rgb

    bands = []
    for i in range(3):
        p2, p98 = get_scaling_params(rgb[..., i])
        bands.append(scale_to_u8(rgb[..., i], p2, p98))
    return np.stack(bands, axis=-1)


def enforce_len(arr, target_len):
    curr = arr.shape[0]
    if curr > target_len:
        return arr[:target_len]
    if curr < target_len:
        pad_shape = list(arr.shape)
        pad_shape[0] = target_len - curr
        return np.concatenate([arr, np.zeros(pad_shape, dtype=arr.dtype)], axis=0)
    return arr


def build_edge_map(image_rgb, method, blur_ksize, canny_t1):
    if blur_ksize % 2 == 0:
        blur_ksize += 1

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    if method == "canny":
        edges = cv2.Canny(blurred, canny_t1, canny_t1 * 2)
    else:
        raise ValueError(f"Unsupported method: {method}")

    return edges


def build_inference_batch(image_rgb, fixed_length, patch_size, method, blur_ksize, canny_t1):
    edges = build_edge_map(image_rgb, method, blur_ksize, canny_t1)
    qdt = FixedQuadTree(edges, fixed_length=fixed_length)
    seq_patches, seq_sizes, seq_pos = qdt.serialize(image_rgb, patch_size, 3)
    coords = np.array([node[0].get_coord() for node in qdt.nodes], dtype=np.float32)

    patches = np.stack(seq_patches, axis=0).astype(np.float32)
    seq_sizes = np.array(seq_sizes, dtype=np.float32)
    seq_pos = np.array(seq_pos, dtype=np.float32)

    patches = enforce_len(patches, fixed_length)
    coords = enforce_len(coords, fixed_length)
    seq_sizes = enforce_len(seq_sizes, fixed_length)
    seq_pos = enforce_len(seq_pos, fixed_length)

    patches_t = torch.from_numpy(patches).permute(0, 3, 1, 2).float()
    mean = torch.tensor(IMAGENET_DEFAULT_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_DEFAULT_STD).view(1, 3, 1, 1)
    patches_t = (patches_t / 255.0 - mean) / std

    return {
        "patches": patches_t.unsqueeze(0),
        "positions": torch.from_numpy(seq_pos).unsqueeze(0).float(),
        "sizes": torch.from_numpy(seq_sizes).unsqueeze(0).float(),
        "coords": torch.from_numpy(coords).unsqueeze(0).float(),
    }


class GDTModelWrapper(nn.Module):
    def __init__(self, encoder, patch_size=16):
        super().__init__()
        self.encoder = encoder
        self.embed_dim = encoder.embed_dim
        self.patch_size = patch_size
        self.mask_decoder = nn.Linear(self.embed_dim, patch_size * patch_size)

    def forward(self, batch_dict):
        x = self.encoder.forward_features(batch_dict)
        patch_tokens = x[:, 1:, :]
        bsz, count, _ = patch_tokens.shape
        patch = self.patch_size
        logits = self.mask_decoder(patch_tokens)
        return {"logits": logits.view(bsz, count, 1, patch, patch)}


def reconstruct_batch_preds(probs, coords, target_size):
    batch, count, _, patch, _ = probs.shape
    preds = torch.zeros((batch, target_size, target_size), device=probs.device)
    for b in range(batch):
        b_coords = coords[b].long()
        b_probs = probs[b, :, 0]
        valid = (b_coords[:, 1] > b_coords[:, 0]) & (b_coords[:, 3] > b_coords[:, 2])
        b_coords = b_coords[valid]
        b_probs = b_probs[valid]
        for i in range(b_coords.shape[0]):
            x1, x2, y1, y2 = b_coords[i]
            p = b_probs[i]
            if (x2 - x1) == patch and (y2 - y1) == patch:
                preds[b, y1:y2, x1:x2] = p
            else:
                resized = F.interpolate(
                    p.unsqueeze(0).unsqueeze(0),
                    size=(int(y2 - y1), int(x2 - x1)),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).squeeze(0)
                preds[b, y1:y2, x1:x2] = resized
    return preds


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def find_default_weights():
    candidates = sorted(Path("SEM_comp").glob("**/best_model_sam.pth"))
    if not candidates:
        candidates = sorted(Path("SEM_comp").glob("**/best_model.pth"))
    return candidates[0] if candidates else None


def infer_params_from_weights(weights_path):
    method = None
    coverage = None
    for part in weights_path.parts[::-1]:
        lowered = part.lower()
        if method is None:
            if "canny" in lowered:
                method = "canny"
        if coverage is None:
            # Match cov0.3 or cov_0.3
            match = re.search(r"cov_?(\d+\.?\d*)", lowered)
            if match:
                coverage = float(match.group(1))
    return method, coverage


def derive_fixed_length(target_size, patch_size, coverage, method):
    if method == "base":
        return (target_size // patch_size) ** 2
    raw_length = int((target_size * target_size * coverage) // (patch_size ** 2))
    fixed_length = max((raw_length // 16) * 16, 1)
    while (fixed_length - 1) % 3 != 0:
        fixed_length += 1
    return fixed_length


def load_checkpoint_bundle(weights_path):
    bundle = torch.load(weights_path, map_location="cpu")
    ckpt_args = {}

    if isinstance(bundle, dict) and "model_state" in bundle:
        state_dict = bundle["model_state"]
        ckpt_args = bundle.get("args", {}) or {}
    else:
        state_dict = bundle

    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint format in {weights_path}")

    normalized_state = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        normalized_state[key] = value

    return normalized_state, ckpt_args


def load_model(state_dict, target_size, patch_size, device):
    encoder = SHFSAMEncoder(
        img_size=target_size,
        patch_size=patch_size,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
    )
    model = GDTModelWrapper(encoder, patch_size=patch_size)
    msg = model.load_state_dict(state_dict, strict=False)
    if msg.missing_keys or msg.unexpected_keys:
        raise RuntimeError(
            "Checkpoint architecture mismatch.\n"
            f"Missing keys: {msg.missing_keys[:8]}\n"
            f"Unexpected keys: {msg.unexpected_keys[:8]}"
        )
    model.to(device)
    model.eval()
    return model


def iter_image_paths(input_path):
    path = Path(input_path)
    if path.is_file():
        return [path]
    if path.is_dir():
        return [p for p in sorted(path.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def clean_binary_mask(binary_mask, min_area):
    contours = extract_contours(binary_mask.astype(np.uint8), min_area)
    clean = np.zeros_like(binary_mask, dtype=np.uint8)
    for contour in contours:
        cv2.drawContours(clean, [contour.astype(np.int32).reshape(-1, 1, 2)], -1, 1, thickness=-1)
    return clean, contours


def save_polygon_json(output_path, image_path, rows, method, threshold):
    payload = {
        "source_file": str(image_path),
        "method": method,
        "threshold": threshold,
        "polygon_count": len(rows),
        "polygons": [],
    }
    for row in rows:
        item = {k: v for k, v in row.items() if k != "contour"}
        item["points"] = [[float(x), float(y)] for x, y in row["contour"]]
        payload["polygons"].append(item)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def overlay_mask_fill(canvas, contours):
    if not contours:
        return canvas
    fill = canvas.copy()
    cnts = [np.round(c).astype(np.int32).reshape(-1, 1, 2) for c in contours]
    cv2.drawContours(fill, cnts, -1, (0, 180, 0), thickness=-1)
    return cv2.addWeighted(fill, 0.22, canvas, 0.78, 0.0)


def build_rows(contours, source_path, diameter_method, aggregate_rows):
    rows = []
    for idx, contour in enumerate(contours, start=1):
        row = compute_metrics(contour, diameter_method)
        row["source_file"] = str(source_path)
        row["polygon_id"] = idx
        rows.append(row)
        aggregate_rows.append(row)
    return rows


def save_preview(preview, rows, methods, output_path, show_values=True, max_instances_per_method=3, mic_spacing_factor=1.2):
    for row in rows:
        draw_measurements(
            preview,
            row,
            methods,
            show_values=show_values,
            max_instances_per_method=max_instances_per_method,
            mic_spacing_factor=mic_spacing_factor,
        )
    Image.fromarray(preview).save(output_path)


def apply_depth_mask(image_rgb, depth_predictor, threshold, invert, label_pct=0.0):
    """Run Depth Anything on an RGB image and return a masked copy.

    label_pct: percentage of the image bottom to exclude from the depth model
               (mirrors the interactive 'Label %' trackbar in depth_anything_remover.py),
               so SEM scale bars don't pollute the depth map.
    """
    h, w = image_rgb.shape[:2]
    crop_y = int(h * (1.0 - label_pct / 100.0)) if label_pct > 0 else h

    # Only feed the non-label portion into the depth model
    img_slice = image_rgb[:crop_y, :]
    img_bgr = cv2.cvtColor(img_slice, cv2.COLOR_RGB2BGR)
    d_slice = depth_predictor.predict(img_bgr)

    # Pad depth map back to full image height
    depth_map = np.zeros((h, w), dtype=np.uint8)
    depth_map[:crop_y, :] = d_slice

    if invert:
        mask = (depth_map > threshold).astype(np.uint8) * 255
    else:
        mask = (depth_map < threshold).astype(np.uint8) * 255

    # Always blank out the label strip regardless of threshold
    if crop_y < h:
        mask[crop_y:, :] = 0

    masked = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)
    return masked


def tile_image(image_rgb, tile_size):
    """Yield (row, col, tile_rgb, (top, left, bottom, right)) for each tile."""
    h, w = image_rgb.shape[:2]
    num_rows = (h + tile_size - 1) // tile_size
    num_cols = (w + tile_size - 1) // tile_size
    for r in range(num_rows):
        for c in range(num_cols):
            top = r * tile_size
            left = c * tile_size
            bottom = min(top + tile_size, h)
            right = min(left + tile_size, w)
            yield r, c, image_rgb[top:bottom, left:right], (top, left, bottom, right)


def _infer_tile(tile_rgb, model, device, args):
    """Run SAM inference on a single tile; returns prob map in tile coords."""
    th, tw = tile_rgb.shape[:2]
    resized = cv2.resize(tile_rgb, (args.target_size, args.target_size), interpolation=cv2.INTER_CUBIC)
    batch = build_inference_batch(
        resized,
        fixed_length=args.fixed_length,
        patch_size=args.patch_size,
        method=args.method,
        blur_ksize=args.blur_ksize,
        canny_t1=args.canny_t1,
    )
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        probs = torch.sigmoid(model(batch)["logits"])
        pred = reconstruct_batch_preds(probs, batch["coords"], args.target_size)[0].cpu().numpy()
    return cv2.resize(pred, (tw, th), interpolation=cv2.INTER_LINEAR)


def process_image(
    image_path,
    model,
    device,
    output_dirs,
    args,
    aggregate_rows,
    depth_predictor=None,
):
    original_rgb = load_image_rgb(image_path)
    orig_h, orig_w = original_rgb.shape[:2]

    # Optional depth-based background removal
    if depth_predictor is not None:
        working_rgb = apply_depth_mask(
            original_rgb, depth_predictor,
            args.depth_threshold, args.depth_invert,
            label_pct=args.depth_label_pct,
        )
    else:
        working_rgb = original_rgb

    # Tiling branch — save each tile individually, no full-image reconstruction
    if args.tile_size is not None and (orig_h > args.tile_size or orig_w > args.tile_size):
        total_polygons = 0
        for r, c, tile_rgb, (top, left, bottom, right) in tile_image(working_rgb, args.tile_size):
            tile_prob = _infer_tile(tile_rgb, model, device, args)
            tile_binary = (tile_prob >= args.threshold).astype(np.uint8)
            tile_clean, tile_contours = clean_binary_mask(tile_binary, args.min_area)
            tile_stem = f"{image_path.stem}_tile_{r}_{c}"

            tile_rows = build_rows(tile_contours, image_path, args.diameter_method, aggregate_rows)
            total_polygons += len(tile_rows)

            Image.fromarray((tile_clean * 255).astype(np.uint8)).save(
                output_dirs["masks"] / f"{tile_stem}_mask.png"
            )

            if args.save_probability:
                Image.fromarray(np.clip(tile_prob * 255.0, 0, 255).astype(np.uint8)).save(
                    output_dirs["probabilities"] / f"{tile_stem}_prob.png"
                )

            save_polygon_json(
                output_dirs["polygons"] / f"{tile_stem}_polygons.json",
                image_path,
                tile_rows,
                args.diameter_method,
                args.threshold,
            )

            tile_preview = overlay_mask_fill(tile_rgb.copy(), tile_contours)
            save_preview(
                tile_preview,
                tile_rows,
                args.overlay_methods,
                output_dirs["overlays"] / f"{tile_stem}_overlay.png",
                show_values=(args.overlay_labels == "values"),
                max_instances_per_method=args.max_instances_per_method,
                mic_spacing_factor=args.mic_spacing_factor,
            )

        print(f"{image_path.name}: {total_polygons} polygon(s) across tiles")
        return

    # Single-pass (no tiling) — save one mask/overlay per image
    resized_rgb = cv2.resize(working_rgb, (args.target_size, args.target_size), interpolation=cv2.INTER_CUBIC)
    batch = build_inference_batch(
        resized_rgb,
        fixed_length=args.fixed_length,
        patch_size=args.patch_size,
        method=args.method,
        blur_ksize=args.blur_ksize,
        canny_t1=args.canny_t1,
    )
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        probs = torch.sigmoid(model(batch)["logits"])
        pred_resized = reconstruct_batch_preds(probs, batch["coords"], args.target_size)[0].cpu().numpy()
    prob_orig = cv2.resize(pred_resized, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    binary_orig = (prob_orig >= args.threshold).astype(np.uint8)
    clean_mask, contours = clean_binary_mask(binary_orig, args.min_area)

    rows = build_rows(contours, image_path, args.diameter_method, aggregate_rows)

    Image.fromarray((clean_mask * 255).astype(np.uint8)).save(
        output_dirs["masks"] / f"{image_path.stem}_mask.png"
    )

    if args.save_probability:
        Image.fromarray(np.clip(prob_orig * 255.0, 0, 255).astype(np.uint8)).save(
            output_dirs["probabilities"] / f"{image_path.stem}_prob.png"
        )

    save_polygon_json(
        output_dirs["polygons"] / f"{image_path.stem}_polygons.json",
        image_path,
        rows,
        args.diameter_method,
        args.threshold,
    )

    preview = overlay_mask_fill(original_rgb.copy(), contours)
    save_preview(
        preview,
        rows,
        args.overlay_methods,
        output_dirs["overlays"] / f"{image_path.stem}_overlay.png",
        show_values=(args.overlay_labels == "values"),
        max_instances_per_method=args.max_instances_per_method,
        mic_spacing_factor=args.mic_spacing_factor,
    )

    print(f"{image_path.name}: {len(rows)} polygon(s)")


def process_mask_image(mask_path, output_dirs, args, aggregate_rows):
    mask = load_mask(mask_path)
    binary = threshold_mask(mask, args.mask_threshold)
    clean_mask, contours = clean_binary_mask(binary, args.min_area)
    rows = build_rows(contours, mask_path, args.diameter_method, aggregate_rows)

    Image.fromarray((clean_mask * 255).astype(np.uint8)).save(output_dirs["masks"] / f"{mask_path.stem}_mask.png")
    save_polygon_json(
        output_dirs["polygons"] / f"{mask_path.stem}_polygons.json",
        mask_path,
        rows,
        args.diameter_method,
        args.mask_threshold,
    )

    preview = cv2.cvtColor((clean_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    preview = overlay_mask_fill(preview, contours)
    save_preview(
        preview,
        rows,
        args.overlay_methods,
        output_dirs["overlays"] / f"{mask_path.stem}_overlay.png",
        show_values=(args.overlay_labels == "values"),
        max_instances_per_method=args.max_instances_per_method,
        mic_spacing_factor=args.mic_spacing_factor,
    )
    print(f"{mask_path.name}: {len(rows)} polygon(s)")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run SAM-SHF inference or consume preprocessed masks, then extract polygons, measure diameters, and save preview overlays."
    )
    parser.add_argument("--input", required=True, help="Input image or directory of images.")
    parser.add_argument("--input_mode", choices=["image", "mask"], default="image", help="Use raw images with SAM inference or consume preprocessed masks directly.")
    parser.add_argument("--output_dir", default="sam_inference_output", help="Directory for masks, polygons, overlays, and CSV output.")
    parser.add_argument("--weights", help="Path to best_model_sam.pth. If omitted, the script tries to find one under SEM_comp.")
    parser.add_argument("--method", choices=["canny"], help="Patchification method used during training. If omitted, inferred from checkpoint path when possible.")
    parser.add_argument("--device", default="auto", help="Torch device, e.g. auto, cpu, cuda:0.")
    parser.add_argument("--target_size", type=int, help="Inference size expected by the checkpoint.")
    parser.add_argument("--patch_size", type=int, help="Patch size used during training.")
    parser.add_argument("--coverage", type=float, help="Coverage ratio used during training (e.g. 0.3).")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for the predicted mask.")
    parser.add_argument("--min_area", type=float, default=16.0, help="Discard polygons smaller than this area in pixels.")
    parser.add_argument("--pixel_size", type=float, default=1.0, help="Physical size of one pixel.")
    parser.add_argument("--unit", default="px", help="Unit label for physical measurement columns.")
    parser.add_argument("--diameter_method", choices=["mic", "min_feret"], default="mic")
    parser.add_argument("--methods", default=None, help="Comma-separated overlay/export measurement methods or 'all'. Example: mic,min_feret")
    parser.add_argument("--overlay_labels", choices=["values", "id"], default="values", help="Overlay labels as values or polygon id only.")
    parser.add_argument("--measurements_csv", default=None, help="Optional long-format CSV with one row per polygon per selected measurement method.")
    parser.add_argument("--instances_csv", default=None, help="Optional instance-level CSV with repeated measurements per polygon and method.")
    parser.add_argument("--per_polygon_stats_csv", default=None, help="Optional summary CSV with mean/std per polygon and method.")
    parser.add_argument("--per_image_stats_csv", default=None, help="Optional summary CSV with mean/std per image and method.")
    parser.add_argument("--max_instances_per_method", type=int, default=3, help="How many measurement geometries to draw and sample per method per polygon.")
    parser.add_argument("--mic_spacing_factor", type=float, default=1.2, help="How aggressively MIC circles suppress nearby candidates. Larger means more spacing.")
    parser.add_argument("--blur_ksize", type=int, default=3, help="Gaussian blur kernel size before edge extraction.")
    parser.add_argument("--canny_t1", type=int, default=100, help="Lower Canny threshold when method=canny.")
    parser.add_argument("--save_probability", action="store_true", help="Save grayscale probability maps alongside binary masks.")
    parser.add_argument("--mask_threshold", type=float, default=127.0, help="Threshold used when input_mode=mask.")
    # Depth Anything preprocessing
    parser.add_argument("--depth_preprocess", action="store_true", help="Apply Depth Anything V2 background removal before inference.")
    parser.add_argument("--depth_threshold", type=int, default=80, help="Depth cutoff value (0-255) for background masking.")
    parser.add_argument("--depth_invert", action="store_true", default=True, help="Invert depth mask: keep pixels ABOVE threshold (default True).")
    parser.add_argument("--no-depth_invert", dest="depth_invert", action="store_false", help="Keep pixels BELOW depth threshold instead.")
    parser.add_argument("--depth_label_pct", type=float, default=12.0, help="Exclude the bottom N%% of the image from depth estimation (e.g. 12 for a 12%% scale-bar strip).")
    parser.add_argument("--depth_model", type=str, default="depth-anything/Depth-Anything-V2-Small-hf", help="HuggingFace model ID for Depth Anything V2.")
    # Tiling
    parser.add_argument("--tile_size", type=int, default=None, help="If set, split each image into tiles of this pixel size before inference.")
    return parser


def main():
    args = build_arg_parser().parse_args()
    args.overlay_methods = resolve_methods(args.methods, default_method=args.diameter_method)

    image_paths = iter_image_paths(args.input)
    if not image_paths:
        raise ValueError(f"No supported images found in {args.input}")

    output_root = Path(args.output_dir)
    output_dirs = {
        "root": output_root,
        "masks": output_root / "masks",
        "polygons": output_root / "polygons",
        "overlays": output_root / "overlays",
        "probabilities": output_root / "probabilities",
    }
    for path in output_dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    model = None
    device = None
    if args.input_mode == "image":
        weights = Path(args.weights) if args.weights else find_default_weights()
        if weights is None or not weights.exists():
            raise FileNotFoundError("Could not find a checkpoint. Pass --weights path/to/best_model_sam.pth or best_model.pth")

        state_dict, ckpt_args = load_checkpoint_bundle(str(weights))

        inferred_method, inferred_coverage = infer_params_from_weights(weights)

        if args.method is None:
            args.method = ckpt_args.get("method") or inferred_method or "canny"

        if args.target_size is None:
            args.target_size = int(ckpt_args.get("target_size", 1024))

        if args.patch_size is None:
            args.patch_size = int(ckpt_args.get("patch_size", 16))

        if args.coverage is None:
            # Priority: ckpt_args['coverage'] > inferred from path > default 0.3
            args.coverage = ckpt_args.get("coverage") or inferred_coverage

        # Finally derive fixed_length from coverage
        if "fixed_length" in ckpt_args and args.coverage is None:
            args.fixed_length = int(ckpt_args["fixed_length"])
        else:
            coverage_val = args.coverage if args.coverage is not None else 0.3
            args.fixed_length = derive_fixed_length(
                target_size=args.target_size,
                patch_size=args.patch_size,
                coverage=float(coverage_val),
                method=args.method,
            )



        device = resolve_device(args.device)
        print(f"Using checkpoint: {weights}")
        print(f"Using method: {args.method}")
        print(f"Using device: {device}")
        print(f"Using target_size={args.target_size}, patch_size={args.patch_size}, fixed_length={args.fixed_length}")
        model = load_model(state_dict, args.target_size, args.patch_size, device)
    else:
        print("Using preprocessed mask mode")

    # Lazy-init depth predictor if requested
    depth_predictor = None
    if args.input_mode == "image" and args.depth_preprocess:
        if not HAS_DEPTH_ANYTHING:
            raise ImportError("depth_anything_remover.py not found or 'transformers' not installed.")
        print(f"Loading Depth Anything model: {args.depth_model}")
        print(f"Depth threshold={args.depth_threshold}, invert={args.depth_invert}, label_pct={args.depth_label_pct}")
        depth_predictor = DepthAnythingPredictor(args.depth_model)

    if args.tile_size is not None:
        print(f"Tiling enabled: tile_size={args.tile_size}")

    all_rows = []
    for image_path in image_paths:
        if args.input_mode == "image":
            process_image(image_path, model, device, output_dirs, args, all_rows, depth_predictor=depth_predictor)
        else:
            process_mask_image(image_path, output_dirs, args, all_rows)

    serialize_rows(
        all_rows,
        output_root / "diameters.csv",
        pixel_size=args.pixel_size,
        unit_name=args.unit,
        method=args.diameter_method,
    )
    long_csv = Path(args.measurements_csv) if args.measurements_csv else (output_root / "diameters_long.csv")
    serialize_measurement_rows(
        all_rows,
        long_csv,
        pixel_size=args.pixel_size,
        unit_name=args.unit,
        methods=args.overlay_methods,
    )
    instances_csv = Path(args.instances_csv) if args.instances_csv else (output_root / "diameter_instances.csv")
    instances = serialize_measurement_instances(
        all_rows,
        instances_csv,
        pixel_size=args.pixel_size,
        unit_name=args.unit,
        methods=args.overlay_methods,
        max_instances_per_method=args.max_instances_per_method,
        mic_spacing_factor=args.mic_spacing_factor,
    )
    serialize_instance_summaries(
        instances,
        Path(args.per_polygon_stats_csv) if args.per_polygon_stats_csv else (output_root / "diameter_stats_per_polygon.csv"),
        Path(args.per_image_stats_csv) if args.per_image_stats_csv else (output_root / "diameter_stats_per_image.csv"),
        pixel_size=args.pixel_size,
        unit_name=args.unit,
    )
    print(f"Wrote {len(all_rows)} measurements to {output_root / 'diameters.csv'}")


if __name__ == "__main__":
    main()
