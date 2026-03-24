import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import random
import gc
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from glob import glob
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

# Ensure project root is in path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from model.sam import SHFSAMEncoder, checkpoint_filter_fn, default_cfgs
from torch.hub import load_state_dict_from_url

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torchvision.transforms.functional as TF
from PIL import Image
import cv2
from SHF.shf import ImagePatchify

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class OnlineSHFDataset(Dataset):
    def __init__(self, image_dir, mask_dir, target_size, fixed_length, patch_size=16, is_train=True, method='canny', invert=None, bgr=False):
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
        
        # In this repo, ImagePatchify is in SHF.shf
        self.patchify = ImagePatchify(fixed_length=fixed_length, patch_size=patch_size, num_channels=3, is_train=is_train, method=method, invert=invert)
        
        if len(self.image_paths) == 0:
            print(f"Warning: No images found in {image_dir}")

    def __len__(self):
        return len(self.image_paths)

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
        
        msk_path_found = self._find_mask_path(basename_no_ext)
                
        pil_img = Image.open(img_path).convert("RGB")
        if self.bgr:
            img_rgb_np = np.array(pil_img)
            img_rgb_np = img_rgb_np[..., ::-1]
            pil_img = Image.fromarray(img_rgb_np)

        if msk_path_found:
            pil_mask = Image.open(msk_path_found).convert("L")
        else:
            pil_mask = Image.new("L", pil_img.size, color=0)
            
        pil_img = pil_img.resize((self.target_size, self.target_size), Image.BICUBIC)
        pil_mask = pil_mask.resize((self.target_size, self.target_size), Image.NEAREST)
        
        if self.is_train:
            if random.random() > 0.5:
                pil_img = TF.hflip(pil_img)
                pil_mask = TF.hflip(pil_mask)
            if random.random() > 0.5:
                pil_img = TF.vflip(pil_img)
                pil_mask = TF.vflip(pil_mask)
        
        img_np = np.array(pil_img)
        mask_np = np.array(pil_mask)
        mask_gt = (mask_np > 127).astype(np.float32)
        
        # ImagePatchify call returns patches, sizes, positions, and the quadtree object
        seq_patches, seq_sizes, seq_pos, qdt = self.patchify(img_np)
        mask_3ch = cv2.cvtColor(mask_np, cv2.COLOR_GRAY2RGB)
        mask_patches_raw, _, _ = qdt.serialize(mask_3ch, P, 3)
        
        mask_patches = []
        for mp in mask_patches_raw:
            mp_mono = mp[:, :, 0]
            mp_bin = (mp_mono > 127).astype(np.float32)
            mask_patches.append(mp_bin)
            
        img_patches = np.stack(seq_patches, axis=0)
        mask_patches = np.stack(mask_patches, axis=0)
        coords = np.array([node[0].get_coord() for node in qdt.nodes])
        
        def enforce_len(arr, tgt_len):
            curr = arr.shape[0]
            if curr > tgt_len: return arr[:tgt_len]
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
            
        patches_t = torch.from_numpy(img_patches).permute(0, 3, 1, 2).float()
        mask_t    = torch.from_numpy(mask_patches).unsqueeze(1).float()
        
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

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class GDTModelWrapper(nn.Module):
    def __init__(self, encoder, patch_size=16):
        super().__init__()
        self.encoder = encoder
        self.embed_dim = encoder.embed_dim
        self.patch_size = patch_size
        self.mask_decoder = nn.Linear(self.embed_dim, patch_size * patch_size)

    def forward(self, batch_dict):
        # forward_features expects batch_dict
        x = self.encoder.forward_features(batch_dict)   # [B, N+1, D]
        patch_tokens = x[:, 1:, :]                      # [B, N,   D]
        B, N, _ = patch_tokens.shape
        P = self.patch_size
        logits = self.mask_decoder(patch_tokens)         # [B, N, P*P]
        return {'logits': logits.view(B, N, 1, P, P)}

# ---------------------------------------------------------------------------
# Loss & Metrics
# ---------------------------------------------------------------------------

class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.bce    = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, inputs, targets):
        sig    = torch.sigmoid(inputs)
        flat_i = sig.reshape(-1)
        flat_t = targets.reshape(-1)
        inter  = (flat_i * flat_t).sum()
        dice   = 1 - (2.*inter + self.smooth) / (flat_i.sum() + flat_t.sum() + self.smooth)
        return self.bce(inputs, targets) + dice

def compute_metrics(probs, masks, thresholds=None):
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

# ---------------------------------------------------------------------------
# Training Session
# ---------------------------------------------------------------------------

def run_training_session(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Dataset
    train_dataset = OnlineSHFDataset(args.image_dir, args.mask_dir, args.target_size, args.fixed_length, patch_size=args.patch_size, is_train=True, method=args.method, bgr=args.bgr)
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    # Encoder (SAM-based)
    encoder = SHFSAMEncoder(
        img_size=args.target_size,
        patch_size=args.patch_size,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
    ).to(device)
    
    # Weight Loading (Pretrained SAM)
    if args.pretrained:
        print(f"Loading pretrained SAM-B weights...")
        cfg = default_cfgs['samvit_base_patch16.sa1b']
        state_dict = load_state_dict_from_url(cfg['url'], map_location='cpu')
        
        filtered_sd = checkpoint_filter_fn(state_dict, encoder)
        msg = encoder.load_state_dict(filtered_sd, strict=False)
        print(f"SAM weights loaded with message: {msg}")
        
    model = GDTModelWrapper(encoder, patch_size=args.patch_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = DiceBCELoss()
    
    best_iou = -1.0
    losses, ious = [], []
    
    for epoch in range(args.epochs):
        model.train()
        ep_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            for k, v in batch.items(): batch[k] = v.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out['logits'], batch['mask'])
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        avg_loss = ep_loss / len(loader)
        losses.append(avg_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_iou = 0.0
            count = 0
            for i, batch in enumerate(loader):
                if i > 10: break
                for k, v in batch.items(): batch[k] = v.to(device)
                out = model(batch)
                probs = torch.sigmoid(out['logits'])
                preds_full = reconstruct_batch_preds(probs, batch['coords'], args.target_size)
                m = compute_metrics(preds_full, batch['full_mask'], thresholds=[0.5])
                val_iou += m[0.5]['iou']
                count += 1
            avg_iou = val_iou / max(1, count)
            ious.append(avg_iou)
            print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f} - IoU (t=0.5): {avg_iou:.4f}")
            
            if avg_iou > best_iou:
                best_iou = avg_iou
                torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model_sam.pth"))

        if (epoch + 1) % 5 == 0:
            plt.figure(figsize=(10, 4))
            plt.plot(losses, label='Loss')
            plt.plot(ious, label='IoU')
            plt.legend(); plt.savefig(os.path.join(args.output_dir, "curves.png")); plt.close()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train SAM-SHF with BTH and Canny methods.")
    parser.add_argument('--method', choices=['canny', 'bth'], default='bth')
    parser.add_argument('--image-dir', type=str, required=True)
    parser.add_argument('--mask-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='./output_sam_shf')
    parser.add_argument('--target-size', type=int, default=1024)
    parser.add_argument('--patch-size', type=int, default=16)
    parser.add_argument('--fixed-length', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--pretrained', action='store_true', help="Load pretrained SAM-B weights")
    parser.add_argument('--bgr', action='store_true')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--num-workers', type=int, default=4)
    
    args = parser.parse_args()
    run_training_session(args)

if __name__ == "__main__":
    main()
