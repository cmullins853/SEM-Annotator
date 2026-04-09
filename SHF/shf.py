import numpy as np

import cv2 as cv2
# cv2.setNumThreads(0)

import torch
import random
    
    
class Rect:
    def __init__(self, x1, x2, y1, y2) -> None:
        self.x1 = int(x1); self.x2 = int(x2); self.y1 = int(y1); self.y2 = int(y2)
        assert self.x1 <= self.x2, 'x1 > x2, wrong coordinate.'
        assert self.y1 <= self.y2, 'y1 > y2, wrong coordinate.'
    
    def contains(self, domain):
        if self.y1 >= self.y2 or self.x1 >= self.x2: return 0
        patch = domain[self.y1:self.y2, self.x1:self.x2]
        return int(np.sum(patch)/255)
    
    def get_area(self, img):
        # return img[self.y1:self.y2, self.x1:self.x2, :].copy()
        return img[self.y1:self.y2, self.x1:self.x2, :]
    
    def get_coord(self):
        return self.x1, self.x2, self.y1, self.y2
    
    def get_size(self):
        return self.x2 - self.x1, self.y2 - self.y1
    
    def get_center(self):
        return (self.x2 + self.x1) / 2, (self.y2 + self.y1) / 2
    
    def draw(self, ax, c='cyan', lw=1, **kwargs):
        rect = patches.Rectangle((self.x1, self.y1), 
                                 width=self.x2-self.x1, 
                                 height=self.y2-self.y1, 
                                 linewidth=lw, edgecolor=c, facecolor='none')
        ax.add_patch(rect)

class FixedQuadTree:
    def __init__(self, domain, fixed_length=128) -> None:
        self.domain = domain
        self.fixed_length = fixed_length
        self._build_tree()
            
    def _build_tree(self):
        h, w = self.domain.shape
        assert h > 0 and w > 0, "Wrong img size."
        root = Rect(0, w, 0, h)
        self.nodes = [[root, root.contains(self.domain)]]
        
        while len(self.nodes) < self.fixed_length:
            # --- [CRITICAL FIX] ---
            # Implemented a robust tree-building logic that prevents infinite loops.
            try:
                # Find the best candidate node to split based on its value (e.g., edge count)
                bbox, value = max(self.nodes, key=lambda x: x[1])
            except ValueError:
                # This happens if self.nodes becomes empty, which shouldn't occur in this logic, but is a safe exit.
                break

            width, height = bbox.get_size()

            # If the best candidate is too small to split or has no value (no edges),
            # then no other nodes are better candidates, so we should stop the tree building process.
            if width < 2 or height < 2 or value <= 0:
                break # Exit the loop cleanly.

            # Find the index of the node to replace
            idx = self.nodes.index([bbox, value])

            x1, x2, y1, y2 = bbox.get_coord()
            cx = x1 + width // 2
            cy = y1 + height // 2
            
            children_defs = [
                (x1, cx, y1, cy), (cx, x2, y1, cy),
                (x1, cx, cy, y2), (cx, x2, cy, y2)
            ]
            
            new_nodes = []
            for (nx1, nx2, ny1, ny2) in children_defs:
                child_rect = Rect(nx1, nx2, ny1, ny2)
                child_val = child_rect.contains(self.domain)
                new_nodes.append([child_rect, child_val])
            
            # Replace the parent node with its four children in the list
            self.nodes = self.nodes[:idx] + new_nodes + self.nodes[idx+1:]

    def serialize(self, img, patch_size, num_channels):
        seq_patch = []
        seq_size = []
        seq_pos = []
        
        for bbox, value in self.nodes:
            seq_patch.append(bbox.get_area(img))
            seq_size.append(bbox.get_size()[0])
            seq_pos.append(bbox.get_center())
            
        final_patches = []
        for patch in seq_patch:
            if patch.shape[0] > 0 and patch.shape[1] > 0:
                resized = cv2.resize(patch, (patch_size, patch_size), interpolation=cv2.INTER_CUBIC)
            else:
                # 如果 patch 尺寸无效，创建一个黑色占位符，而不是让程序崩溃
                resized = np.zeros((patch_size, patch_size, num_channels), dtype=img.dtype)
            final_patches.append(resized)

        num_generated = len(final_patches)
        if num_generated < self.fixed_length:
            padding_needed = self.fixed_length - num_generated
            pad_patch = np.zeros((patch_size, patch_size, num_channels), dtype=np.uint8)
            final_patches.extend([pad_patch] * padding_needed)
            seq_size.extend([0] * padding_needed)
            seq_pos.extend([(0, 0)] * padding_needed)
            
        return final_patches, seq_size, seq_pos

class ImagePatchify:
    def __init__(self, fixed_length=196, patch_size=16, num_channels=3, is_train=True, method='canny', invert=None) -> None:
        self.fixed_length = fixed_length
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.is_train = is_train
        self.method = method
        self.invert = invert
        self.sths = [0, 1, 3, 5]
        
        # Stochastic thresholds for 'forest' trees
        self.cannys = list(range(50, 151, 10))               # Canny: 50..150
        self.bth_thresholds = list(range(15, 86, 10))        # BTH: 30..80
        self.localvar_thresholds = list(range(20, 61, 10))   # LocalVar: 10..80
        self.psd_thresholds = list(range(30, 51, 5))        # PSD: 30..50
        self.wavelet_thresholds = list(range(5, 21, 5))    # Wavelet: 5..20
        
    def __call__(self, img_np):
        if img_np.ndim == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        elif img_np.ndim == 3 and img_np.shape[2] == 1:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

        # smooth_factor = random.choice(self.sths)
        smooth_factor = 3
        grey_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        if smooth_factor == 0 :
            edges = (np.random.uniform(low=0, high=255, size=img_np.shape[:2])).astype(np.uint8)
        else:
            if self.method == 'canny':
                blurred = cv2.GaussianBlur(grey_img, (smooth_factor, smooth_factor), 0)
                canny_t1 = random.choice(self.cannys)
                edges = cv2.Canny(blurred, canny_t1, canny_t1 * 2)
                
            elif self.method == 'bth':
                # Black Top-Hat: High response for dark details
                kernel_size = 25
                se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                closed = cv2.morphologyEx(grey_img, cv2.MORPH_CLOSE, se)
                bth = cv2.subtract(closed, grey_img).astype(np.float32)
                
                # Normalize to 0-255
                mn, mx = bth.min(), bth.max()
                if mx - mn > 1e-8:
                    bth = (bth - mn) / (mx - mn) * 255.0
                
                # Threshold to binary map. Default invert for BTH is True.
                do_invert = self.invert if self.invert is not None else True
                thresh_type = cv2.THRESH_BINARY_INV if do_invert else cv2.THRESH_BINARY
                
                t = random.choice(self.bth_thresholds)
                _, edges = cv2.threshold(bth.astype(np.uint8), t, 255, thresh_type)
                # Ensure pure black areas are ignored (masked to 0)
                edges[grey_img == 0] = 0
           
            else:
                raise ValueError(f"Unknown ImagePatchify method: {self.method}")

        qdt = FixedQuadTree(domain=edges, fixed_length=self.fixed_length)
        
        # Serialization now operates on the RGB image
        seq_patches, seq_sizes, seq_pos = qdt.serialize(
            img_np, self.patch_size, self.num_channels
        )
        # Return qdt to get coordinate information in the transform
        return seq_patches, seq_sizes, seq_pos, qdt

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# --- Visualization and deserialization helper functions ---
def denormalize(tensor, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
    """Denormalizes a tensor for display."""
    mean = torch.tensor(mean)
    std = torch.tensor(std)

    if tensor.ndim == 4: # Batch of patches [N, C, H, W]
        mean = mean.view(1, 3, 1, 1)
        std = std.view(1, 3, 1, 1)
        tensor = tensor.cpu() * std + mean
        tensor = torch.clamp(tensor, 0, 1)
        return tensor.permute(0, 2, 3, 1).numpy() # -> [N, H, W, C]
    elif tensor.ndim == 3: # Single image [C, H, W]
        mean = mean.view(3, 1, 1)
        std = std.view(3, 1, 1)
        tensor = tensor.cpu() * std + mean
        tensor = torch.clamp(tensor, 0, 1)
        return tensor.permute(1, 2, 0).numpy() # -> [H, W, C]
    else:
        raise TypeError(f"Unsupported tensor dimension: {tensor.ndim}")


def deserialize_patches(patches_tensor, coords_tensor, image_size):
    """Reconstructs an image from serialized patches."""
    reconstructed_img = np.zeros((image_size, image_size, 3), dtype=np.float32)
    
    # First, denormalize all patches
    denormalized_patches = denormalize(patches_tensor) # This call is now safe
    
    for i in range(patches_tensor.shape[0]):
        patch_np = denormalized_patches[i]
        x1, x2, y1, y2 = coords_tensor[i].numpy()
        
        width, height = x2 - x1, y2 - y1
        if width == 0 or height == 0: continue

        # Resize the patch back to its original size in the quadtree
        resized_patch = cv2.resize(patch_np, (width, height), interpolation=cv2.INTER_CUBIC)
        reconstructed_img[y1:y2, x1:x2, :] = resized_patch
        
    return np.clip(reconstructed_img, 0, 1)