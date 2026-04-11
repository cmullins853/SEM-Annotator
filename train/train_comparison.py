import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import random
import gc
import math
from contextlib import nullcontext
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from glob import glob
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import re
import traceback

try:
    import accelerate
except ImportError:
    accelerate = None


# Ensure project root is in path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from model.vit import SHFVisionTransformer
from model.sam import SHFSAMEncoder, checkpoint_filter_fn, default_cfgs
from torch.hub import load_state_dict_from_url

try:
    from train.energy_monitor import EnergyMonitor
except ImportError:
    EnergyMonitor = None

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torchvision.transforms.functional as TF
from PIL import Image
import cv2
from SHF.shf import ImagePatchify


class _IdentityAccelerator:
    def __init__(self, cpu=False):
        self.device = torch.device("cpu" if cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.num_processes = 1
        self.process_index = 0
        self.is_main_process = True
        self.is_local_main_process = True
        self.sync_gradients = True

    def prepare(self, *objs):
        return objs if len(objs) != 1 else objs[0]

    def accumulate(self, _model):
        return nullcontext()

    def backward(self, loss):
        loss.backward()

    def unwrap_model(self, model):
        return model

    def gather_for_metrics(self, data):
        return data

    def wait_for_everyone(self):
        return None


def create_accelerator(args):
    cpu_mode = args.gpu.lower() == 'cpu'
    if accelerate is None:
        return _IdentityAccelerator(cpu=cpu_mode)
    ddp_kwargs = None
    try:
        from accelerate.utils import DistributedDataParallelKwargs
        ddp_kwargs = DistributedDataParallelKwargs(
            gradient_as_bucket_view=False,
            find_unused_parameters=True,
        )
    except ImportError:
        ddp_kwargs = None
    return accelerate.Accelerator(
        cpu=cpu_mode,
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.grad_accumulation_steps,
        kwargs_handlers=[ddp_kwargs] if ddp_kwargs is not None else None,
    )


def _sync_stop_flag(flag, device):
    tensor = torch.tensor([1 if flag else 0], device=device, dtype=torch.int32)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return bool(tensor.item())


def extract_uniform_patches(image_rgb, mask_gray, patch_size=16):
    """Extract a regular non-overlapping grid of patches in raster order."""
    h, w = image_rgb.shape[:2]
    img_patches, mask_patches, sizes, centers, coords = [], [], [], [], []
    for row in range(0, h - patch_size + 1, patch_size):
        for col in range(0, w - patch_size + 1, patch_size):
            img_patch = image_rgb[row:row + patch_size, col:col + patch_size]
            mask_patch = mask_gray[row:row + patch_size, col:col + patch_size]
            img_patches.append(img_patch)
            mask_patches.append((mask_patch > 127).astype(np.float32))
            sizes.append(patch_size)
            centers.append((col + patch_size / 2.0, row + patch_size / 2.0))
            coords.append([col, col + patch_size, row, row + patch_size])
    return (
        np.stack(img_patches, axis=0),
        np.stack(mask_patches, axis=0),
        np.array(sizes, dtype=np.float32),
        np.array(centers, dtype=np.float32),
        np.array(coords, dtype=np.float32),
    )


def apply_training_augmentations_pil(pil_img, pil_mask, brightness_jitter=0.15):
    if random.random() > 0.5:
        pil_img = TF.hflip(pil_img)
        pil_mask = TF.hflip(pil_mask)
    if random.random() > 0.5:
        pil_img = TF.vflip(pil_img)
        pil_mask = TF.vflip(pil_mask)
    if brightness_jitter > 0:
        factor = 1.0 + random.uniform(-brightness_jitter, brightness_jitter)
        pil_img = TF.adjust_brightness(pil_img, factor)
    return pil_img, pil_mask


def apply_training_augmentations_preprocessed(img_data, mask_data, coords, target_size, brightness_jitter=0.15):
    img_data = img_data.copy()
    mask_data = mask_data.copy()
    coords = coords.copy()

    if random.random() > 0.5:
        img_data = np.flip(img_data, axis=3).copy()
        mask_data = np.flip(mask_data, axis=3).copy()
        x1 = coords[:, 0].copy()
        x2 = coords[:, 1].copy()
        coords[:, 0] = target_size - x2
        coords[:, 1] = target_size - x1

    if random.random() > 0.5:
        img_data = np.flip(img_data, axis=2).copy()
        mask_data = np.flip(mask_data, axis=2).copy()
        y1 = coords[:, 2].copy()
        y2 = coords[:, 3].copy()
        coords[:, 2] = target_size - y2
        coords[:, 3] = target_size - y1

    if brightness_jitter > 0:
        factor = 1.0 + random.uniform(-brightness_jitter, brightness_jitter)
        img_data = np.clip(img_data * factor, 0, 255)

    return img_data, mask_data, coords


def repeat_items(items, repeat_factor):
    repeat_factor = max(1, int(repeat_factor))
    if repeat_factor == 1:
        return list(items)
    return list(items) * repeat_factor

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class OnlineSHFDataset(Dataset):
    """
    Loads raw tiles/masks, optionally applies synchronous augmentations,
    and runs them through a dynamic, stochastic ImagePatchify QuadTree
    on every epoch to create a true 'Hierarchical Forest' during training.
    """
    def __init__(self, image_dir, mask_dir, target_size, fixed_length, patch_size=16, is_train=True, method='canny', invert=None, bgr=False, brightness_jitter=0.15):
        self.image_paths  = sorted(
            glob(os.path.join(image_dir, "*.png"))  +
            glob(os.path.join(image_dir, "*.jpg"))  +
            glob(os.path.join(image_dir, "*.tif"))  +
            glob(os.path.join(image_dir, "*.tiff"))
        )
        self.mask_dir     = mask_dir
        self.target_size  = target_size
        self.fixed_length = fixed_length
        self.patch_size   = patch_size
        self.is_train     = is_train
        self.method       = method
        self.invert       = invert
        self.bgr          = bgr
        self.brightness_jitter = brightness_jitter
        
        self.patchify = ImagePatchify(fixed_length=fixed_length, patch_size=patch_size, num_channels=3, is_train=is_train, method=method, invert=invert)
        
        if len(self.image_paths) == 0:
            print(f"Warning: No images found in {image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def set_fixed_length(self, fixed_length):
        self.fixed_length = int(fixed_length)
        self.patchify = ImagePatchify(
            fixed_length=self.fixed_length,
            patch_size=self.patch_size,
            num_channels=3,
            is_train=self.is_train,
            method=self.method,
            invert=self.invert,
        )

    def _find_mask_path(self, basename_no_ext):
        for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
            candidate = os.path.join(self.mask_dir, basename_no_ext + ext)
            if os.path.exists(candidate):
                return candidate
        return None

    def __getitem__(self, idx):
        target_len = self.fixed_length
        P = self.patch_size
        
        img_path = self.image_paths[idx]
        basename_no_ext = os.path.splitext(os.path.basename(img_path))[0]
        
        # Load full ground truth mask
        msk_path_found = self._find_mask_path(basename_no_ext)
                
        pil_img = Image.open(img_path).convert("RGB")
        if self.bgr:
            # Swap R and B channels
            img_rgb_np = np.array(pil_img)
            img_rgb_np = img_rgb_np[..., ::-1]
            pil_img = Image.fromarray(img_rgb_np)

        if msk_path_found:
            pil_mask = Image.open(msk_path_found).convert("L")
        else:
            pil_mask = Image.new("L", pil_img.size, color=0)
            
        # Resize to target size
        pil_img = pil_img.resize((self.target_size, self.target_size), Image.BICUBIC)
        pil_mask = pil_mask.resize((self.target_size, self.target_size), Image.NEAREST)
        
        if self.is_train:
            pil_img, pil_mask = apply_training_augmentations_pil(
                pil_img, pil_mask, brightness_jitter=self.brightness_jitter
            )
        
        img_np = np.array(pil_img)
        mask_np = np.array(pil_mask)
        mask_gt = (mask_np > 127).astype(np.float32)
        
        if self.method == 'base':
            img_patches, mask_patches, seq_sizes, seq_pos, coords = extract_uniform_patches(
                img_np, mask_np, patch_size=P
            )
        else:
            # 1. Stochastic QuadTree build on the image
            seq_patches, seq_sizes, seq_pos, qdt = self.patchify(img_np)

            # 2. Serialize the mask using the SAME tree
            mask_3ch = cv2.cvtColor(mask_np, cv2.COLOR_GRAY2RGB)
            mask_patches_raw, _, _ = qdt.serialize(mask_3ch, P, 3)

            mask_patches = []
            for mp in mask_patches_raw:
                mp_mono = mp[:, :, 0]
                mp_bin = (mp_mono > 127).astype(np.float32)
                mask_patches.append(mp_bin)

            img_patches = np.stack(seq_patches, axis=0)
            mask_patches = np.stack(mask_patches, axis=0)
            coords = np.array([node[0].get_coord() for node in qdt.nodes], dtype=np.float32)
        
        # Enforce fixed length padding safely for each array
        def enforce_len(arr, tgt_len):
            curr = arr.shape[0]
            if curr > tgt_len:
                return arr[:tgt_len]
            elif curr < tgt_len:
                pad_shape = list(arr.shape)
                pad_shape[0] = tgt_len - curr
                return np.concatenate([arr, np.zeros(pad_shape, dtype=arr.dtype)], axis=0)
            return arr
            
        seq_sizes    = np.array(seq_sizes, dtype=np.float32)
        seq_pos      = np.array(seq_pos, dtype=np.float32)
        
        img_patches  = enforce_len(img_patches, target_len)
        mask_patches = enforce_len(mask_patches, target_len)
        coords       = enforce_len(coords, target_len)
        seq_sizes    = enforce_len(seq_sizes, target_len)
        seq_pos      = enforce_len(seq_pos, target_len)
            
        # Tensors
        patches_t = torch.from_numpy(img_patches).permute(0, 3, 1, 2).float()
        mask_t    = torch.from_numpy(mask_patches).unsqueeze(1).float()
        
        # ImageNet normalization
        mean = torch.tensor(IMAGENET_DEFAULT_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_DEFAULT_STD).view(1, 3, 1, 1)
        patches_t = (patches_t / 255.0 - mean) / std

        return {
            'patches':   patches_t,
            'mask':      mask_t,
            'full_mask': torch.from_numpy(mask_gt).float(),
            'positions': torch.tensor(seq_pos, dtype=torch.float32),
            'sizes':     torch.tensor(seq_sizes, dtype=torch.float32),
            'coords':    torch.tensor(coords, dtype=torch.float32),
        }


class PreprocessedSHFDataset(Dataset):
    """
    Loads preprocessed .npz files (img, mask, coords) generated by preprocess_comparison.py.
    Also loads the original full ground-truth mask for evaluation.
    """
    def __init__(self, preproc_dir, image_paths, mask_dir, target_size, fixed_length, patch_size=16, is_train=True, bgr=False, brightness_jitter=0.15):
        self.preproc_dir = preproc_dir
        self.image_paths = image_paths
        self.mask_dir = mask_dir
        self.target_size = target_size
        self.fixed_length = fixed_length
        self.patch_size = patch_size
        self.is_train = is_train
        self.bgr = bgr
        self.brightness_jitter = brightness_jitter
        
        # Map each image path to its corresponding .npz file
        self.paths = []
        for img_p in image_paths:
            basename_no_ext = os.path.splitext(os.path.basename(img_p))[0]
            npz_path = os.path.join(preproc_dir, basename_no_ext + ".npz")
            if os.path.exists(npz_path):
                self.paths.append(npz_path)
            else:
                self.paths.append(None) 

    def __len__(self):
        return len(self.image_paths)

    def set_fixed_length(self, fixed_length):
        self.fixed_length = int(fixed_length)

    def __getitem__(self, idx):
        target_len = self.fixed_length
        P = self.patch_size
        
        # Load full ground truth mask (robust version from old CompDataset)
        basename_no_ext = os.path.splitext(os.path.basename(self.image_paths[idx]))[0]
        mask_opts = [os.path.join(self.mask_dir, basename_no_ext + ext) for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']]
        mask_gt = np.zeros((self.target_size, self.target_size), dtype=np.float32)
        for m_path in mask_opts:
            if os.path.exists(m_path):
                try:
                    m_arr = np.array(Image.open(m_path).convert("L"))
                    if m_arr.shape != (self.target_size, self.target_size):
                        m_arr = cv2.resize(m_arr, (self.target_size, self.target_size), interpolation=cv2.INTER_NEAREST)
                    mask_gt = (m_arr > 127).astype(np.float32)
                except Exception as e:
                    print(f"Error loading mask {m_path}: {e}")
                break

        npz_path = self.paths[idx]
        if npz_path is None or not os.path.exists(npz_path):
            return {
                'patches':   torch.zeros((target_len, 3, P, P)),
                'mask':      torch.zeros((target_len, 1, P, P)),
                'full_mask': torch.from_numpy(mask_gt).float(),
                'positions': torch.zeros((target_len, 2)),
                'sizes':     torch.zeros(target_len),
                'coords':    torch.zeros((target_len, 4)),
            }

        try:
            with np.load(npz_path) as data:
                img_data  = data['img'].astype(np.float32)    # [N, 3, P, P]
                mask_data = data['mask'].astype(np.float32)   # [N, 1, P, P]
                coords    = data['coords'].astype(np.float32) # [N, 4]

                if self.bgr:
                    # img_data is [N, 3, P, P]
                    img_data = img_data[:, [2, 1, 0], :, :]

                if self.is_train:
                    img_data, mask_data, coords = apply_training_augmentations_preprocessed(
                        img_data,
                        mask_data,
                        coords,
                        target_size=self.target_size,
                        brightness_jitter=self.brightness_jitter,
                    )

                # Enforce fixed length (padding/truncating)
                curr_len = img_data.shape[0]
                if curr_len > target_len:
                    img_data  = img_data[:target_len]
                    mask_data = mask_data[:target_len]
                    coords    = coords[:target_len]
                elif curr_len < target_len:
                    diff = target_len - curr_len
                    img_data  = np.concatenate([img_data,  np.zeros((diff, 3, P, P), dtype=np.float32)], axis=0)
                    mask_data = np.concatenate([mask_data, np.zeros((diff, 1, P, P), dtype=np.float32)], axis=0)
                    coords    = np.concatenate([coords,    np.zeros((diff, 4),       dtype=np.float32)], axis=0)

                # Tensors
                patches_t = torch.from_numpy(img_data).float()
                mask_t    = torch.from_numpy(mask_data).float() / 255.0  # Normalize masks
                
                # ImageNet normalization for patches
                mean = torch.tensor(IMAGENET_DEFAULT_MEAN).view(1, 3, 1, 1)
                std = torch.tensor(IMAGENET_DEFAULT_STD).view(1, 3, 1, 1)
                patches_t = (patches_t / 255.0 - mean) / std

                # Derive positions and sizes from coords [x1, x2, y1, y2]
                x1, x2, y1, y2 = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                seq_sizes = (x2 - x1)
                seq_pos   = np.stack([cx, cy], axis=-1)

                return {
                    'patches':   patches_t,
                    'mask':      mask_t,
                    'full_mask': torch.from_numpy(mask_gt).float(),
                    'positions': torch.from_numpy(seq_pos).float(),
                    'sizes':     torch.from_numpy(seq_sizes).float(),
                    'coords':    torch.from_numpy(coords).float(),
                }
        except Exception as e:
            print(f"Error loading {npz_path}: {e}")
            return {
                'patches':   torch.zeros((target_len, 3, P, P)),
                'mask':      torch.zeros((target_len, 1, P, P)),
                'full_mask': torch.from_numpy(mask_gt).float(),
                'positions': torch.zeros((target_len, 2)),
                'sizes':     torch.zeros(target_len),
                'coords':    torch.zeros((target_len, 4)),
            }


# ---------------------------------------------------------------------------
# Background Analysis Helper
# ---------------------------------------------------------------------------

def is_mask_empty(image_path, mask_dir):
    """
    Checks if an image's corresponding mask is empty (all zeros).
    If no mask file is found, it's treated as empty (all background).
    """
    try:
        basename = os.path.basename(image_path)
        name_no_ext = os.path.splitext(basename)[0]
        
        # Check common mask extensions
        mask_path = None
        for ext in ['.png', '.tif', '.tiff', '.jpg', '.jpeg']:
            test_path = os.path.join(mask_dir, name_no_ext + ext)
            if os.path.exists(test_path):
                mask_path = test_path
                break
        
        # If mask file is missing, it's treated as all-zeros by the dataset loader
        if mask_path is None:
            return True
            
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            # If load fails, treat as empty
            return True
            
        return np.sum(mask) == 0
    except:
        return True

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class GDTModelWrapper(nn.Module):
    """
    Wraps SHFVisionTransformer with a lightweight linear mask decoder.
    Encoder → [B, N+1, D] → strip CLS → linear → [B, N, 1, P, P] logits.
    """
    def __init__(self, encoder, patch_size=16):
        super().__init__()
        self.encoder = encoder
        self.embed_dim = encoder.embed_dim
        self.patch_size = patch_size
        self.mask_decoder = nn.Linear(self.embed_dim, patch_size * patch_size)

    def forward(self, batch_dict):
        x = self.encoder.forward_features(batch_dict)   # [B, N+1, D]
        patch_tokens = x[:, 1:, :]                      # [B, N,   D]
        B, N, D = patch_tokens.shape
        P = self.patch_size
        logits = self.mask_decoder(patch_tokens)         # [B, N, P*P]
        return {'logits': logits.view(B, N, 1, P, P)}


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.bce    = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, inputs, targets, token_valid_mask=None):
        if token_valid_mask is None:
            sig = torch.sigmoid(inputs)
            flat_i = sig.reshape(-1)
            flat_t = targets.reshape(-1)
            inter = (flat_i * flat_t).sum()
            dice = 1 - (2. * inter + self.smooth) / (flat_i.sum() + flat_t.sum() + self.smooth)
            return self.bce(inputs, targets) + dice

        valid = token_valid_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(inputs.dtype)
        valid_sum = valid.sum().clamp_min(1.0)

        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        bce = (bce * valid).sum() / valid_sum

        sig = torch.sigmoid(inputs) * valid
        tgt = targets * valid
        flat_i = sig.reshape(-1)
        flat_t = tgt.reshape(-1)
        inter = (flat_i * flat_t).sum()
        dice = 1 - (2. * inter + self.smooth) / (flat_i.sum() + flat_t.sum() + self.smooth)
        return bce + dice


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def compute_metrics(probs, masks, thresholds=None):
    """
    Compute per-threshold IoU and Dice scores.

    Args:
        probs  : torch.Tensor of sigmoid probabilities, any shape.
        masks  : torch.Tensor of binary ground-truth masks, same shape.
        thresholds : iterable of float thresholds (default: 0.1..0.9 in steps of 0.1)

    Returns:
        dict mapping threshold -> {'iou': float, 'dice': float}
    """
    if thresholds is None:
        thresholds = [round(t, 1) for t in np.linspace(0.1, 0.9, 9)]
    out = {}
    for t in thresholds:
        preds        = (probs > t).float()
        flat_p       = preds.reshape(-1)
        flat_g       = masks.reshape(-1)
        intersection = (flat_p * flat_g).sum().item()
        union        = (flat_p + flat_g).clamp(0, 1).sum().item()
        iou          = intersection / (union + 1e-6)
        dice         = (2. * intersection + 1e-6) / (flat_p.sum().item() + flat_g.sum().item() + 1e-6)
        out[t]       = {'iou': iou, 'dice': dice}
    return out


def reconstruct_batch_preds(probs, coords, target_size):
    B, N, _, P, _ = probs.shape
    device = probs.device
    preds = torch.zeros((B, target_size, target_size), device=device)
    for b in range(B):
        b_coords = coords[b].long()
        b_probs = probs[b, :, 0]
        
        valid_idx = (b_coords[:, 1] > b_coords[:, 0]) & (b_coords[:, 3] > b_coords[:, 2])
        b_coords = b_coords[valid_idx]
        b_probs = b_probs[valid_idx]
        
        for i in range(b_coords.shape[0]):
            x1, x2, y1, y2 = b_coords[i]
            p = b_probs[i]
            
            if (x2 - x1) == P and (y2 - y1) == P:
                preds[b, y1:y2, x1:x2] = p
            else:
                p_res = F.interpolate(p.unsqueeze(0).unsqueeze(0), 
                                      size=(int(y2-y1), int(x2-x1)), 
                                      mode='bilinear', 
                                      align_corners=False).squeeze(0).squeeze(0)
                preds[b, y1:y2, x1:x2] = p_res
    return preds


def compute_patchwise_stats(probs, masks, coords, threshold=0.5):
    preds = (probs > threshold).float()
    valid = ((coords[..., 1] > coords[..., 0]) & (coords[..., 3] > coords[..., 2])).to(probs.dtype)
    valid = valid.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    preds = preds * valid
    masks = masks * valid

    intersection = (preds * masks).sum()
    union = (preds + masks).clamp(0, 1).sum()
    pred_sum = preds.sum()
    mask_sum = masks.sum()
    return intersection, union, pred_sum, mask_sum


# ---------------------------------------------------------------------------
# Validation (runs on val split, returns scalar IoU and Dice at t=0.5)
# ---------------------------------------------------------------------------

def validate(model, loader, device, threshold=0.5, target_size=1024, accelerator=None):
    """
    Quick patch-wise validation pass. Returns global IoU and Dice across
    valid serialized patches at the given threshold.
    """
    accelerator = accelerator or _IdentityAccelerator(cpu=(device.type == "cpu"))
    model.eval()
    intersection_sum = torch.tensor(0.0, device=device)
    union_sum = torch.tensor(0.0, device=device)
    pred_sum = torch.tensor(0.0, device=device)
    mask_sum = torch.tensor(0.0, device=device)
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validating", disable=not accelerator.is_local_main_process)
        for batch in pbar:
            for k, v in batch.items():
                batch[k] = v.to(device)
            out = model(batch)
            probs = torch.sigmoid(out['logits'])
            coords = out.get('coords', batch['coords'])
            inter, union, pred_pixels, mask_pixels = compute_patchwise_stats(
                probs, batch['mask'], coords, threshold=threshold
            )
            intersection_sum += inter
            union_sum += union
            pred_sum += pred_pixels
            mask_sum += mask_pixels

            running_iou = float((intersection_sum / union_sum.clamp_min(1e-6)).item())
            running_dice = float(((2.0 * intersection_sum) / (pred_sum + mask_sum).clamp_min(1e-6)).item())
            pbar.set_postfix(iou=f"{running_iou:.4f}", dice=f"{running_dice:.4f}")

    stats = torch.stack([intersection_sum, union_sum, pred_sum, mask_sum]).to(torch.float64)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    mean_iou = float(stats[0].item() / max(stats[1].item(), 1e-6))
    mean_dice = float((2.0 * stats[0].item()) / max(stats[2].item() + stats[3].item(), 1e-6))
    return mean_iou, mean_dice


# ---------------------------------------------------------------------------
# Full evaluation (multi-threshold, with visualisation)
# ---------------------------------------------------------------------------

def evaluate(model, loader, device, output_dir, target_size, accelerator=None):
    accelerator = accelerator or _IdentityAccelerator(cpu=(device.type == "cpu"))
    if accelerator.is_main_process:
        print("\n--- Starting Evaluation ---")
    model.eval()
    thresholds = [round(t, 1) for t in np.linspace(0.1, 0.9, 9)]
    accum      = {t: {'iou': [], 'dice': []} for t in thresholds}
    vis_data   = None
    best_vis_score = -1.0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", disable=not accelerator.is_local_main_process):
            for k, v in batch.items():
                batch[k] = v.to(device)

            out = model(batch)
            logits = out['logits']
            probs  = torch.sigmoid(logits)
            masks  = batch['mask']

            preds_full = reconstruct_batch_preds(probs, out.get('coords', batch['coords']), target_size)
            full_masks = batch['full_mask'].to(device)

            preds_full = accelerator.gather_for_metrics(preds_full)
            full_masks = accelerator.gather_for_metrics(full_masks)
            if accelerator.is_main_process:
                batch_metrics = compute_metrics(preds_full, full_masks, thresholds)
                for t, scores in batch_metrics.items():
                    accum[t]['iou'].append(scores['iou'])
                    accum[t]['dice'].append(scores['dice'])

            # Pick a well-covered sample for visualization
            if not accelerator.is_main_process:
                continue
            gathered_patches = accelerator.gather_for_metrics(batch['patches'])
            gathered_masks = accelerator.gather_for_metrics(masks)
            gathered_probs = accelerator.gather_for_metrics(probs)
            gathered_coords = accelerator.gather_for_metrics(out.get('coords', batch['coords']))
            gathered_full_masks = accelerator.gather_for_metrics(full_masks)
            mean = torch.tensor(IMAGENET_DEFAULT_MEAN, device=gathered_patches.device).view(1, 1, 3, 1, 1)
            std = torch.tensor(IMAGENET_DEFAULT_STD, device=gathered_patches.device).view(1, 1, 3, 1, 1)
            gathered_patches_vis = (gathered_patches * std + mean).clamp(0.0, 1.0)
            for b_idx in range(gathered_patches.shape[0]):
                patches_sample = gathered_patches_vis[b_idx]
                coords_sample  = gathered_coords[b_idx]
                max_vals       = patches_sample.reshape(patches_sample.shape[0], -1).max(dim=1).values
                non_black      = max_vals > 0.1
                if non_black.any():
                    nb_coords = coords_sample[non_black]
                    dx        = nb_coords[:, 1] - nb_coords[:, 0]
                    dy        = nb_coords[:, 3] - nb_coords[:, 2]
                    cov_ratio = (dx * dy).sum().item() / (target_size ** 2)
                    if cov_ratio >= 0.5:
                        mask_score = float(gathered_full_masks[b_idx].sum().item())
                        if mask_score > best_vis_score:
                            best_vis_score = mask_score
                            vis_data = (
                                gathered_patches_vis.cpu(), gathered_masks.cpu(),
                                gathered_probs.cpu(), gathered_coords.cpu(), b_idx, gathered_full_masks.cpu()
                            )

    if not accelerator.is_main_process:
        return
    final_results = {
        t: {'iou':  float(np.mean(v['iou'])),
            'dice': float(np.mean(v['dice']))}
        for t, v in accum.items()
    }
    print("\nEvaluation Results (Full Reconstruction):")
    print(f"  {'Threshold':>10}  {'IoU':>8}  {'Dice':>8}")
    for t in sorted(final_results.keys()):
        r = final_results[t]
        print(f"  {t:>10.1f}  {r['iou']:>8.4f}  {r['dice']:>8.4f}")

    # --- Visualization ---
    if vis_data is not None:
        p_img, p_mask, p_pred, p_coords, idx, p_full_mask = vis_data
        img_patches  = p_img[idx].numpy()
        pred_patches = p_pred[idx].numpy()
        coords       = p_coords[idx].numpy()
        full_mask    = p_full_mask[idx].numpy()

        S = target_size
        canvas_img   = np.zeros((S, S, 3))
        canvas_pred  = np.zeros((S, S))

        for j in range(len(img_patches)):
            x1, x2, y1, y2 = coords[j].astype(int)
            if x2 <= x1 or y2 <= y1:
                continue
            p_rgb   = img_patches[j].transpose(1, 2, 0)
            p_p     = pred_patches[j, 0]
            p_p_bin = (p_p > 0.5).astype(np.float32)

            sz = (x2 - x1, y2 - y1)
            p_rgb   = cv2.resize(p_rgb,   sz, interpolation=cv2.INTER_NEAREST)
            p_p_bin = cv2.resize(p_p_bin, sz, interpolation=cv2.INTER_NEAREST)

            canvas_img  [y1:y2, x1:x2] = p_rgb
            canvas_pred [y1:y2, x1:x2] = p_p_bin
            
        canvas_gt = full_mask
        canvas_error = np.zeros((S, S, 3))
        p_g_bin = canvas_gt > 0.5
        p_p_bin = canvas_pred > 0.5

        canvas_error[ p_g_bin &  p_p_bin] = [1, 1, 1]      # TP White
        canvas_error[~p_g_bin &  p_p_bin] = [1, 0, 0]      # FP Red
        canvas_error[ p_g_bin & ~p_p_bin] = [1, 0.65, 0]   # FN Orange

        plt.figure(figsize=(20, 5))
        plt.subplot(141); plt.imshow(canvas_img.astype(np.float32));  plt.title("Reconstructed Input", pad=12);     plt.axis('off')
        plt.subplot(142); plt.imshow(canvas_gt,   cmap='gray');        plt.title("GT", pad=12);                     plt.axis('off')
        plt.subplot(143); plt.imshow(canvas_pred, cmap='gray');        plt.title("Prediction (Binary)", pad=12);    plt.axis('off')
        plt.subplot(144); plt.imshow(canvas_error);                    plt.title("Error (FP:Red FN:Orange)", pad=12); plt.axis('off')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        vis_path = os.path.join(output_dir, "vis_reconstruct.png")
        plt.savefig(vis_path, bbox_inches='tight', pad_inches=0.25)
        print(f"Visualization saved to {vis_path}")
        plt.close()

    return final_results

def save_metrics(args, monitor, epoch, val_iou=0.0, val_dice=0.0, out_dir='./logs'):
    stats = monitor.get_stats()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{args.method}_{args.coverage}_comparison_metrics.txt"
    file_path = out_dir / filename
    
    file_exists = file_path.exists()
    
    with open(file_path, 'a') as f:
        if not file_exists:
            f.write("Epoch\tValIoU\tValDice\tEnergy(J)\tTime(s)\tkWh\n")
        f.write(f"{epoch+1}\t{val_iou:.4f}\t{val_dice:.4f}\t{stats['joules']:.2f}\t{stats['seconds']:.1f}\t{stats['kwh']:.6f}\n")
    return file_path
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_training_session(args, fixed_length, seed, output_dir, device):
    """
    Executes a single training/validation/evaluation run with the specified parameters.
    """
    accelerator = create_accelerator(args)
    device = accelerator.device
    os.makedirs(output_dir, exist_ok=True)
    
    # Set seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if accelerator.is_main_process:
        print(f"\n" + "="*80)
        print(f" STARTING RUN: Method={args.method}, Length={fixed_length}, Seed={seed}")
        print(f" Output Dir : {output_dir}")
        print("="*80)

    patch_size  = args.patch_size
    target_size = args.target_size

    # ---- Train / validation split ----------------------------------------
    all_image_paths = sorted(
        glob(os.path.join(args.image_dir, "*.png"))  +
        glob(os.path.join(args.image_dir, "*.jpg"))  +
        glob(os.path.join(args.image_dir, "*.tif"))  +
        glob(os.path.join(args.image_dir, "*.tiff"))
    )
    n_total = len(all_image_paths)
    if n_total == 0:
        if accelerator.is_main_process:
            print("Dataset is empty. Skipping run.")
        return
        
    val_split = max(0.0, min(args.val_split, 0.5))
    test_split = max(0.0, min(args.test_split, 0.5))
    if val_split + test_split >= 1.0:
        if accelerator.is_main_process:
            print(
                f"Invalid split configuration: val_split ({val_split}) + "
                f"test_split ({test_split}) must be less than 1.0."
            )
        return
    
    n_test  = int(math.floor(n_total * test_split)) if test_split > 0 else 0
    n_val   = int(math.floor(n_total * val_split))  if val_split > 0 else 0
    n_train = max(0, n_total - n_val - n_test)
    if n_train == 0:
        if accelerator.is_main_process:
            print(
                f"Invalid split configuration leaves no training samples "
                f"(total={n_total}, train={n_train}, val={n_val}, test={n_test})."
            )
        return

    indices = list(range(n_total))
    random.Random(seed).shuffle(indices) # Important to use the run's seed here
    
    train_idx = indices[:n_train]
    val_idx   = indices[n_train : n_train + n_val]
    test_idx  = indices[n_train + n_val : n_train + n_val + n_test]

    # Ensure directories are absolute paths
    args.image_dir = os.path.abspath(args.image_dir)
    args.mask_dir = os.path.abspath(args.mask_dir)
    if args.preprocessed_dir:
        args.preprocessed_dir = os.path.abspath(args.preprocessed_dir)

    # Check if directories exist
    if not os.path.exists(args.mask_dir):
        print(f"\n[!] ERROR: Mask directory not found: {args.mask_dir}")
        print(f"    Check your --mask-dir argument. Aborting training.\n")
        sys.exit(1)
    
    # Check for recognized mask files
    has_masks = any(os.path.isfile(os.path.join(args.mask_dir, f)) for f in os.listdir(args.mask_dir) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')))
    if not has_masks:
        print(f"\n[!] ERROR: Mask directory exists but contains no recognized image files: {args.mask_dir}")
        print(f"    Aborting training to prevent incorrect background labels.\n")
        sys.exit(1)

    # ---- [NEW] Background Analysis ----
    if accelerator.is_main_process:
        print(f"Analyzing dataset split for images with no labels...")
    def get_bg_stats(idx_list, paths, m_dir):
        bg_count = 0
        for i in idx_list:
            if is_mask_empty(paths[i], m_dir):
                bg_count += 1
        return bg_count, len(idx_list)

    train_bg, train_total = get_bg_stats(train_idx, all_image_paths, args.mask_dir)
    val_bg,   val_total   = get_bg_stats(val_idx,   all_image_paths, args.mask_dir)
    test_bg,  test_total  = get_bg_stats(test_idx,  all_image_paths, args.mask_dir)
    
    if accelerator.is_main_process:
        print(f" Dataset Stats:")
        print(f"   - Train Split : {train_bg}/{train_total} images with no labels ({ (train_bg/max(1,train_total))*100:.1f}%)")
        print(f"   - Val Split   : {val_bg}/{val_total} images with no labels ({ (val_bg/max(1,val_total))*100:.1f}%)")
        print(f"   - Test Split  : {test_bg}/{test_total} images with no labels ({ (test_bg/max(1,test_total))*100:.1f}%)")
    # ----------------------------------
    # ----------------------------------

    # Online SHF (Canny, Base, etc.)
    full_dataset_train = OnlineSHFDataset(args.image_dir, args.mask_dir, target_size, fixed_length, patch_size, is_train=True,  method=args.method, invert=False, bgr=args.bgr, brightness_jitter=args.brightness_jitter)
    full_dataset_val   = OnlineSHFDataset(args.image_dir, args.mask_dir, target_size, fixed_length, patch_size, is_train=False, method=args.method, invert=False, bgr=args.bgr, brightness_jitter=args.brightness_jitter)
    full_dataset_test  = OnlineSHFDataset(args.image_dir, args.mask_dir, target_size, fixed_length, patch_size, is_train=False, method=args.method, invert=False, bgr=args.bgr, brightness_jitter=args.brightness_jitter)

    train_dataset = Subset(full_dataset_train, repeat_items(train_idx, args.train_repeat_factor))
    val_dataset   = Subset(full_dataset_val,   val_idx)  if n_val > 0 else None
    test_dataset  = Subset(full_dataset_test,  test_idx) if n_test > 0 else None

    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=False,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=False,
        )
        if val_dataset is not None else None
    )
    test_loader = (
        DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=False,
        )
        if test_dataset is not None else None
    )
    if val_loader is not None and test_loader is not None:
        loader, val_loader, test_loader = accelerator.prepare(loader, val_loader, test_loader)
    elif val_loader is not None:
        loader, val_loader = accelerator.prepare(loader, val_loader)
    elif test_loader is not None:
        loader, test_loader = accelerator.prepare(loader, test_loader)
    else:
        loader = accelerator.prepare(loader)
    if accelerator.is_main_process:
        print(f"Train samples : {len(train_dataset)}  |  Val samples : {len(val_dataset) if val_dataset else 0}  |  Test samples : {len(test_dataset) if test_dataset else 0}")
        print(f"Base train split : {len(train_idx)}  |  Train repeat factor : {args.train_repeat_factor}")

    # Model
    if args.method == 'base':
        if accelerator.is_main_process:
            print("Using SHFVisionTransformer (ViT) for 'base' method.")
        encoder = SHFVisionTransformer(
            img_size=target_size,
            patch_size=patch_size,
            in_channels=3,
            num_classes=0,
            embed_dim=768,
            depth=12,
            num_heads=12,
        )
    else:
        if accelerator.is_main_process:
            print(f"Using SHFSAMEncoder (SAM-B) for method '{args.method}'.")
        encoder = SHFSAMEncoder(
            img_size=target_size,
            patch_size=patch_size,
            in_channels=3,
            num_classes=0,
            embed_dim=768,
            depth=12,
            num_heads=12,
            # SAM specific windowing is disabled inside SHFSAMEncoder by default for SHF
        )
    
    # Weight Loading
    loaded_any = False
    if args.pretrained:
        if os.path.exists(args.pretrained):
            if accelerator.is_main_process:
                print(f"Loading pretrained encoder weights from: {args.pretrained}")
            ckpt = torch.load(args.pretrained, map_location=device)
            state_dict = ckpt['model'] if 'model' in ckpt else ckpt
            
            if isinstance(encoder, SHFSAMEncoder):
                # Filter/remap for SAM if needed (though if it's a saved SHFSAMEncoder ckpt it might be direct)
                # But if it's a standard SAM-B .pth, we use checkpoint_filter_fn
                if any(k.startswith('image_encoder.') for k in state_dict.keys()):
                    state_dict = checkpoint_filter_fn(state_dict, encoder)
            
            encoder_state_dict = {k: v for k, v in state_dict.items() if k in encoder.state_dict()}
            encoder.load_state_dict(encoder_state_dict, strict=False)
            if accelerator.is_main_process:
                print(f"  Loaded {len(encoder_state_dict)} component tensors into encoder.")
            loaded_any = True
        else:
             if accelerator.is_main_process:
                print(f"Pretrained path not found: {args.pretrained}. Skipping manual load.")

    # Auto-load SAM weights if using SAM and no manual weights loaded
    if isinstance(encoder, SHFSAMEncoder) and not loaded_any:
        if accelerator.is_main_process:
            print("Loading default SAM-B weights for SHFSAMEncoder...")
        sam_cfg = default_cfgs.get('samvit_base_patch16.sa1b') or default_cfgs.get('sam_vit_b')
        if sam_cfg:
            state_dict = load_state_dict_from_url(sam_cfg['url'], map_location=device)
            state_dict = checkpoint_filter_fn(state_dict, encoder)
            encoder.load_state_dict(state_dict, strict=False)
            if accelerator.is_main_process:
                print("  SAM-B weights loaded from URL source.")

    model     = GDTModelWrapper(encoder, patch_size=patch_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model, optimizer = accelerator.prepare(model, optimizer)
    criterion = DiceBCELoss()
    monitor   = EnergyMonitor(interval=2.0) if EnergyMonitor and accelerator.is_main_process else None
    best_ckpt = os.path.join(output_dir, "best_model.pth")

    def _get_model_state():
        return accelerator.unwrap_model(model).state_dict()

    def _load_model_state(state_dict):
        target_state = accelerator.unwrap_model(model).state_dict()
        if state_dict and next(iter(state_dict)).startswith('module.') and not next(iter(target_state)).startswith('module.'):
            state_dict = {k.removeprefix('module.'): v for k, v in state_dict.items()}
        accelerator.unwrap_model(model).load_state_dict(state_dict)

    losses         = []
    val_ious       = []
    val_dices      = []
    best_val_iou   = -1.0
    best_val_dice  = -1.0
    best_ckpt_score = -1.0
    epochs_no_imp  = 0
    start_epoch    = 0

    if args.resume:
        if os.path.exists(args.resume):
            if accelerator.is_main_process:
                print(f"Resuming from checkpoint: {args.resume}")
            ckpt_data = torch.load(args.resume, map_location=device)
            if isinstance(ckpt_data, dict) and 'model_state' in ckpt_data:
                _load_model_state(ckpt_data['model_state'])
                optimizer.load_state_dict(ckpt_data['optimizer_state'])
                start_epoch    = ckpt_data.get('epoch', 0) + 1
                best_val_iou   = ckpt_data.get('best_val_iou', -1.0)
                best_val_dice  = ckpt_data.get('best_val_dice', -1.0)
                best_ckpt_score = ckpt_data.get('best_ckpt_score', -1.0)
                losses         = ckpt_data.get('losses', [])
                val_ious       = ckpt_data.get('val_ious', [])
                val_dices      = ckpt_data.get('val_dices', [])
                epochs_no_imp  = ckpt_data.get('epochs_no_imp', 0)
                if accelerator.is_main_process:
                    print(
                        f"  Resumed at epoch {start_epoch}  |  best_val_iou={best_val_iou:.4f}  "
                        f"|  best_val_dice={best_val_dice:.4f}"
                    )
            else:
                model.load_state_dict(ckpt_data)
                if accelerator.is_main_process:
                    print("  Loaded model weights only.")

    if accelerator.is_main_process and args.target_iou is not None:
        print(f"  Early-stop target     : IoU ≥ {args.target_iou}")
    if accelerator.is_main_process and args.target_dice is not None:
        print(f"  Early-stop target     : Dice ≥ {args.target_dice}")
    if accelerator.is_main_process and args.target_compute is not None:
        if monitor is None:
            print("  WARNING: --target-compute set but EnergyMonitor is unavailable; limit ignored.")
        else:
            print(f"  Early-stop target     : {args.target_compute:.1f} J cumulative")
    if accelerator.is_main_process and args.patience is not None:
        print(f"  Patience              : {args.patience} epochs")
    if accelerator.is_main_process and val_loader is not None:
        print(f"  Best checkpoint metric: {args.best_checkpoint_metric.upper()}")

    if monitor:
        monitor.start()

    stop_reason = None
    try:
        for epoch in range(start_epoch, args.epochs):
            if monitor: monitor.reset()
            model.train()
            ep_loss, count = 0.0, 0
            val_iou, val_dice = 0.0, 0.0
            should_stop = False
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}", disable=not accelerator.is_local_main_process)

            for batch in pbar:
                for k, v in batch.items(): batch[k] = v.to(device)
                masks = batch['mask']
                optimizer.zero_grad()
                with accelerator.accumulate(model):
                    out = model(batch)
                    logits = out['logits']
                    loss = criterion(logits, masks, out.get('token_valid_mask'))
                    accelerator.backward(loss)
                    optimizer.step()
                ep_loss += loss.item()
                count   += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            avg = ep_loss / max(1, count)
            losses.append(avg)
            print(f"Epoch {epoch+1} — avg train loss: {avg:.4f}", end="")

            if val_loader is not None:
                val_iou, val_dice = validate(model, val_loader, device, threshold=0.5, target_size=target_size, accelerator=accelerator)
                val_ious.append(val_iou)
                val_dices.append(val_dice)
                print(f"  |  val IoU: {val_iou:.4f}  val Dice: {val_dice:.4f}")

                previous_best_iou = best_val_iou
                if val_iou > previous_best_iou:
                    best_val_iou  = val_iou
                    epochs_no_imp = 0
                else:
                    epochs_no_imp += 1

                if val_dice > best_val_dice:
                    best_val_dice = val_dice

                ckpt_metric_name = args.best_checkpoint_metric
                ckpt_metric_value = val_iou if ckpt_metric_name == 'iou' else val_dice
                if ckpt_metric_value > best_ckpt_score:
                    best_ckpt_score = ckpt_metric_value
                    ckpt = os.path.join(output_dir, "best_model.pth")
                    if accelerator.is_main_process:
                        torch.save({
                            'model_state':     _get_model_state(),
                            'optimizer_state': optimizer.state_dict(),
                            'epoch':           epoch,
                            'best_val_iou':    best_val_iou, # This will be -1.0 if no val_loader
                            'best_val_dice':   best_val_dice,
                            'best_ckpt_score': best_ckpt_score,
                            'best_checkpoint_metric': args.best_checkpoint_metric,
                            'losses':          losses,
                            'val_ious':        val_ious,
                            'val_dices':       val_dices,
                            'epochs_no_imp':   epochs_no_imp,
                            'args':            vars(args),
                        }, ckpt)
                    print(f"  ↳ New best val {ckpt_metric_name.upper()} {best_ckpt_score:.4f} — saved.")

                if args.target_iou is not None and val_iou >= args.target_iou:
                    stop_reason = f"target IoU {args.target_iou} reached"
                    should_stop = True
                elif args.target_dice is not None and val_dice >= args.target_dice:
                    stop_reason = f"target Dice {args.target_dice} reached"
                    should_stop = True
                elif args.patience is not None and epochs_no_imp >= args.patience:
                    stop_reason = f"patience {args.patience} reached"
                    should_stop = True
            else:
                # No validation: fall back to training-loss best
                print()  # newline after train loss
                if not hasattr(locals(), '_best_train_loss') or avg < _best_train_loss:
                    _best_train_loss = avg
                    ckpt = os.path.join(output_dir, "best_model.pth")
                    if accelerator.is_main_process:
                        torch.save({
                            'model_state':     _get_model_state(),
                            'optimizer_state': optimizer.state_dict(),
                            'epoch':           epoch,
                            'best_val_iou':    best_val_iou, # This will be -1.0 if no val_loader
                            'best_val_dice':   best_val_dice,
                            'best_ckpt_score': best_ckpt_score,
                            'best_checkpoint_metric': args.best_checkpoint_metric,
                            'losses':          losses,
                            'val_ious':        val_ious,
                            'val_dices':       val_dices,
                            'epochs_no_imp':   epochs_no_imp,
                            'args':            vars(args),
                        }, ckpt)
                    print(f"  ↳ New best train loss {avg:.4f} — saved to {ckpt}")


            if monitor:
                if accelerator.is_main_process:
                    monitor.print_stats(prefix=f"EP{epoch+1}")
                    save_metrics(args, monitor, epoch, val_iou=val_iou, val_dice=val_dice, out_dir=output_dir)
                if args.target_compute is not None:
                    if monitor.get_stats()['cumulative_joules'] >= args.target_compute:
                        stop_reason = "compute budget reached"
                        should_stop = True

            if accelerator.is_main_process and (epoch + 1) % 5 == 0:
                # Save intermediary plots to output_dir
                plt.figure(figsize=(10, 4))
                plt.subplot(1, 2, 1)
                plt.plot(losses, label='Train Loss')
                plt.title(f"Training Loss — {args.method} len={fixed_length}")
                plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
                if val_ious:
                    plt.subplot(1, 2, 2)
                    plt.plot(val_ious,  label='Val IoU')
                    plt.plot(val_dices, label='Val Dice')
                    plt.title("Validation Metrics (Patch-wise t=0.5)")
                    plt.xlabel("Epoch"); plt.ylabel("Score"); plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "loss_curve.png"))
                plt.close()

            if _sync_stop_flag(should_stop, device):
                break

    except KeyboardInterrupt:
        if accelerator.is_main_process:
            print("\nTraining interrupted session.")
        raise # Pass it up to main
    finally:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process and stop_reason:
            print(f"\n⏹  Early stop: {stop_reason}")
        if monitor: monitor.stop()
        
        # Save final evaluation and model
        # [NEW] Evaluate on TEST split if available, otherwise fallback to val or train
        eval_loader = test_loader if test_loader is not None else (val_loader if val_loader is not None else loader)
        eval_name = "TEST" if test_loader is not None else ("VAL" if val_loader is not None else "TRAIN")

        try:
            if os.path.exists(best_ckpt):
                best_data = torch.load(best_ckpt, map_location=device)
                best_state = best_data['model_state'] if isinstance(best_data, dict) and 'model_state' in best_data else best_data
                _load_model_state(best_state)
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    print(f"Loaded best checkpoint for final evaluation: {best_ckpt}")
            if accelerator.is_main_process:
                print(f"\n--- Final Evaluation Performance (Split: {eval_name}) ---")
            evaluate(model, eval_loader, device, output_dir, target_size, accelerator=accelerator)
        except Exception as e:
            if accelerator.is_main_process:
                print(f"Evaluation failed: {e}")
            
        final_ckpt = os.path.join(output_dir, "final_model.pth")
        if accelerator.is_main_process:
            torch.save({
                'model_state':     _get_model_state(),
                'optimizer_state': optimizer.state_dict(),
                'epoch':           epoch,
                'best_val_iou':    best_val_iou, # This will be -1.0 if no val_loader
                'best_val_dice':   best_val_dice,
                'best_ckpt_score': best_ckpt_score,
                'best_checkpoint_metric': args.best_checkpoint_metric,
                'losses':          losses,
                'val_ious':        val_ious,
                'val_dices':       val_dices,
                'epochs_no_imp':   epochs_no_imp,
                'args':            vars(args),
            }, final_ckpt)
            print(f"Final model saved to {final_ckpt}")
        
        # Cleanup
        if hasattr(loader, '_iterator') and loader._iterator is not None:
            loader._iterator._shutdown_workers()
        if val_loader is not None and hasattr(val_loader, '_iterator') and val_loader._iterator is not None:
            val_loader._iterator._shutdown_workers()
            
        del model, encoder, optimizer, loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()

def main():
    parser = argparse.ArgumentParser(
        description="Train SHFVisionTransformer with automated iteration support."
    )
    # --- Method / data ---
    parser.add_argument('--method', choices=['canny', 'base'], default='base',
        help="Method used during preprocessing (must match preprocess_comparison.py --method). "
             "'base' = uniform 16×16 grid, no SHF.",
    )

    parser.add_argument('--image-dir', default=r'E:\Connor\SHF Data\Tiffs\multiclass_dataset\images_tiles',
        help="Root directory where raw image tiles are located.",
    )
    parser.add_argument('--mask-dir', default="./Xmas",
                        help="Path to full ground truth masks for proper evaluation.")
    parser.add_argument('--preprocessed-dir', default="E:/Connor/comparison_study",
                        help="Path to preprocessed .npz files.")
    parser.add_argument('--bgr', action='store_true', help="Swap BGR channels to RGB upon loading (for datasets where channels are switched).")
    # --- MM-SHF-style coverage / target-size (must match preprocess run) ---
    parser.add_argument('--coverage', type=float, default=0.75,
        help="Coverage fraction used during preprocessing (determines fixed_length).",
    )
    parser.add_argument('--target-size', type=int, default=1024,
        help="Image resolution used during preprocessing.",
    )
    parser.add_argument('--patch-size', type=int, default=16,
        help="Patch size used during preprocessing.",
    )
    # --- Training ---
    parser.add_argument('--epochs',     type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--brightness-jitter', type=float, default=0.15,
                        help="Training-only brightness jitter amount. 0 disables brightness augmentation.")
    parser.add_argument('--train-repeat-factor', type=int, default=2,
                        help="Repeat the training split this many times per epoch. Validation/test are not repeated.")
    parser.add_argument('--output-base', default='output_comparison',
        help="Root directory for checkpoints, loss curves, and visualizations.",
    )
    parser.add_argument('--pretrained', type=str, default=None,
        help="Path to an existing masked autoencoder .pth checkpoint to load encoder weights before training.",
    )
    # --- Early-stopping / accuracy limits ---
    parser.add_argument('--val-split', type=float, default=0.15,
        help="Fraction of data set aside for validation (0 = no validation split).",
    )
    parser.add_argument('--test-split', type=float, default=0.10,
        help="Fraction of data set aside for testing (0 = no test split).",
    )
    parser.add_argument('--seed', type=int, default=42,
        help="Random seed for data splitting (ensures comparable eval across models).",
    )
    
    # --- Full Run / Iteration ---
    parser.add_argument('--full-run', action='store_true', help="Iterate over all methods and seeds for online training.")
    parser.add_argument('--coverages', type=float, nargs='+', help="Specific coverages to test (e.g. 0.5 0.75).")
    parser.add_argument('--fixed-lengths', type=int, nargs='+', help="Specific fixed lengths to test.")
    parser.add_argument('--seeds', type=int, nargs='+', help="Seeds to test (default: 2, 10, 40).")

    # Exactly one of the following three targets may be specified at a time.
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument('--target-iou', type=float, default=None,
        help="Stop training once validation IoU (t=0.5) reaches this value.",
    )
    target_group.add_argument('--target-dice', type=float, default=None,
        help="Stop training once validation Dice (t=0.5) reaches this value.",
    )
    target_group.add_argument('--target-compute', type=float, default=None,
        help="Stop training once cumulative energy consumption reaches this many kilojoules.",
    )
    parser.add_argument('--patience', type=int, default=None,
                        help="Stop training after this many epochs with no improvement in val IoU.")
    parser.add_argument('--best-checkpoint-metric', choices=['iou', 'dice'], default='iou',
                        help="Metric used to decide when to overwrite best_model.pth.")
    parser.add_argument('--gpu', type=str, default='0',
                        help="GPU device ID or 'cpu' (default: 0)")
    parser.add_argument('--resume', type=str, default=None,
                        help="Path to a checkpoint .pth file to resume training from. "
                             "The checkpoint must have been saved by this script (contains "
                             "model_state, optimizer_state, epoch, best_val_iou, best_val_dice, "
                             "losses, val_ious, val_dices).")
    parser.add_argument('--mixed-precision', choices=['no', 'fp16', 'bf16'], default='no',
                        help="Mixed precision mode when running with Hugging Face accelerate.")
    parser.add_argument('--grad-accumulation-steps', type=int, default=1,
                        help="Gradient accumulation steps when running with accelerate.")
    args = parser.parse_args()

    if args.target_compute is not None:
        args.target_compute = args.target_compute * 1000

    # Device selection
    if args.gpu.lower() == 'cpu': device = torch.device("cpu")
    else: device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    print(f"Device      : {device}")
    print(f"Image dir   : {args.image_dir}")
    print(f"Output base : {args.output_base}")

    methods_to_test = ['canny', 'base']
    
    runs = [] # List of (method, coverage, seed, preproc_dir)
    
    if args.full_run:
        seeds_to_test     = args.seeds if args.seeds else [2, 10, 40]
        coverages_to_test = args.coverages if args.coverages else [args.coverage]
        for seed in seeds_to_test:
            for method in methods_to_test:
                is_base = (method == "base")
                # For 'base', coverage is irrelevant/constant
                curr_coverages = [args.coverage] if is_base else coverages_to_test
                for cov in curr_coverages:
                    runs.append((method, cov, seed, None))
    else:
        runs.append((args.method, args.coverage, args.seed, getattr(args, 'preprocessed_dir', None)))

    for method, cov, seed, preproc_dir in runs:
        args.method = method
        args.coverage = cov
        if preproc_dir:
            args.preprocessed_dir = preproc_dir 
            
        is_base = (method == "base")
        
        # Determine fixed_length
        if is_base:
            fl = (args.target_size // args.patch_size) ** 2
        else:
            raw_length = int((args.target_size * args.target_size * cov) // (args.patch_size ** 2))
            fl = (raw_length // 16) * 16
            while (fl - 1) % 3 != 0: fl += 1
        
        param_str = f"{method}_cov{cov}_seed{seed}" if not is_base else f"{method}_seed{seed}"
        output_dir = os.path.join(args.output_base, param_str)

        try:
            run_training_session(args, fl, seed, output_dir, device)
        except KeyboardInterrupt:
            print("\nOverall process interrupted by user.")
            return
        except Exception as e:
            print(f"\n[ERROR] Run failed (Method={method}, Cov={cov}, Seed={seed}): {e}")
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()
