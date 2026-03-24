"""
SAM-B (ViT-Base) 模型工厂，支持动态调整Patch Size以适应超高分辨率输入，并包含分割解码器。
根据图像尺寸自动适配输入通道数 (1k/8k: 1通道, 32k: 3通道)。
"""
import logging
from functools import partial
from typing import Callable, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import (
    PatchEmbed,
    Mlp,
    DropPath,
    LayerNorm2d,
    LayerScale,
    Format,
    resample_abs_pos_embed_nhwc,
    to_2tuple,
    use_fused_attn,
)
from torch.jit import Final
import math
from .vit import SpatioStructuralPosEmbed

# --- Helper for Tiled GPU Upsampling (Avoids INT_MAX limit) ---
def gpu_safe_interpolate(x: torch.Tensor, size: Tuple[int, int], mode: str = 'bilinear', align_corners: bool = False) -> torch.Tensor:
    """
    Performs interpolation on GPU using tiling if the output tensor exceeds INT_MAX elements.
    INT_MAX is approx 2.147 billion. slightly less to be safe: 2.0e9.
    """
    B, C, H_in, W_in = x.shape
    H_out, W_out = size
    
    total_elements = B * C * H_out * W_out
    SAFE_LIMIT = 2 * 10**9
    
    if total_elements <= SAFE_LIMIT:
        return F.interpolate(x, size=size, mode=mode, align_corners=align_corners)
    
    row_size = W_out * B * C
    max_chunk_h_out = max(1, int(SAFE_LIMIT / row_size))
    num_chunks = math.ceil(H_out / max_chunk_h_out)
    
    _logger.info(f"Output tensor size {total_elements} > INT_MAX. Using tiled GPU upsampling with {num_chunks} vertical chunks.")
    
    output_slices = []
    scale_h = H_out / H_in
    
    for i in range(num_chunks):
        h_start_out = i * max_chunk_h_out
        h_end_out = min((i + 1) * max_chunk_h_out, H_out)
        
        if abs(scale_h - round(scale_h)) < 1e-6:
            int_scale = int(round(scale_h))
            chunk_h_in = max(1, int(max_chunk_h_out / int_scale))
            h_start_in_aligned = i * chunk_h_in
            if h_start_in_aligned >= H_in: break
            h_end_in_aligned = min((i + 1) * chunk_h_in, H_in)
            x_s = x[:, :, h_start_in_aligned:h_end_in_aligned, :]
            h_len_out_aligned = int((h_end_in_aligned - h_start_in_aligned) * scale_h)
            out_s = F.interpolate(x_s, size=(h_len_out_aligned, W_out), mode=mode, align_corners=align_corners)
            output_slices.append(out_s)
        else:
            x_cpu = x.cpu().float()
            out_cpu = F.interpolate(x_cpu, size=size, mode=mode, align_corners=align_corners)
            return out_cpu.to(x.device)

    return torch.cat(output_slices, dim=2)

class SafeUpsample(nn.Module):
    def __init__(self, scale_factor=2, mode='bilinear', align_corners=False):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        
    def forward(self, x):
        B, C, H, W = x.shape
        new_H = int(H * self.scale_factor)
        new_W = int(W * self.scale_factor)
        return gpu_safe_interpolate(x, size=(new_H, new_W), mode=self.mode, align_corners=self.align_corners)

_logger = logging.getLogger(__name__)

def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos
    q_coords = torch.arange(q_size, dtype=torch.float32)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size, dtype=torch.float32)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
    return rel_pos_resized[relative_coords.long()]

def get_decomposed_rel_pos_bias(
        q: torch.Tensor,
        rel_pos_h: torch.Tensor,
        rel_pos_w: torch.Tensor,
        q_size: Tuple[int, int],
        k_size: Tuple[int, int],
) -> torch.Tensor:
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)
    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)
    attn_bias = rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    return attn_bias.reshape(-1, q_h * q_w, k_h * k_w)

class Attention(nn.Module):
    fused_attn: Final[bool]
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            use_rel_pos: bool = False,
            input_size: Optional[Tuple[int, int]] = None,
            rope: Optional[nn.Module] = None,
            device=None,
            dtype=None,
    ):
        super().__init__()
        param_dd = {'device': device, 'dtype': dtype}
        layer_dd = {}

        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias, **layer_dd)
        self.q_norm = norm_layer(self.head_dim, **layer_dd) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim, **layer_dd) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, **layer_dd)
        self.proj_drop = nn.Dropout(proj_drop)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert rope is None
            assert (
                    input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, self.head_dim, **param_dd))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, self.head_dim, **param_dd))
        self.rope = rope

    def forward(self, x):
        B, H, W, _ = x.shape
        N = H * W
        x = x.reshape(B, N, -1)
        qkv = self.qkv(x).view(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # qkv with shape (3, B, nHead, N, C)
        q, k, v = qkv.unbind(0)
        # q, k, v with shape (B, nHead, N, C)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.use_rel_pos:
            q_3d = q.reshape(B * self.num_heads, N, -1)
            attn_bias = get_decomposed_rel_pos_bias(q_3d, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))
            attn_bias = attn_bias.view(B, self.num_heads, N, N)
        else:
            attn_bias = None
            if self.rope is not None:
                raise NotImplementedError("ROPE is not implemented in this simplified example.")

        if self.fused_attn:
            attn_mask_for_fused = None
            if attn_bias is not None:
                pass
            
            x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask_for_fused,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if attn_bias is not None:
                attn = attn + attn_bias
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, H, W, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)

def window_unpartition(
    windows: torch.Tensor, window_size: int, hw: Tuple[int, int], pad_hw: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    Hp, Wp = pad_hw if pad_hw is not None else hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
    x = x[:, :H, :W, :].contiguous()
    return x

class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            mlp_layer: Type[nn.Module] = Mlp,
            use_rel_pos: bool = False,
            window_size: int = 0,
            input_size=None,
            rope=None,
            device=None,
            dtype=None,
    ):
        super().__init__()
        layer_dd = {}
        self.window_size = window_size
        self.norm1 = norm_layer(dim, **layer_dd)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm,
            attn_drop=attn_drop, proj_drop=proj_drop, norm_layer=norm_layer,
            use_rel_pos=use_rel_pos, input_size=input_size if window_size == 0 else (window_size, window_size),
            rope=rope, device=device, dtype=dtype,
        )
        self.ls1 = LayerScale(dim, init_values=init_values, **layer_dd) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim, **layer_dd)
        self.mlp = mlp_layer(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=proj_drop)
        self.ls2 = LayerScale(dim, init_values=init_values, **layer_dd) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, H, W, _ = x.shape
        shortcut = x
        x = self.norm1(x)
        pad_hw: Optional[Tuple[int, int]] = None
        if self.window_size > 0:
            x, pad_hw = window_partition(x, self.window_size)
        x = self.drop_path1(self.ls1(self.attn(x)))
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, (H, W), pad_hw)
        x = shortcut + x
        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path2(self.ls2(x))
        x = shortcut + x
        return x

class SimpleSegmentationDecoder(nn.Module):
    def __init__(self, in_channels=256, out_channels=1, upsample_factor=16, decoder_channels=(128, 64, 32, 16)):
        super().__init__()
        num_upsample_layers = int(torch.log2(torch.tensor(float(upsample_factor))).item())
        if len(decoder_channels) < num_upsample_layers:
            decoder_channels = list(decoder_channels) + [decoder_channels[-1]] * (num_upsample_layers - len(decoder_channels))
        elif len(decoder_channels) > num_upsample_layers:
             decoder_channels = decoder_channels[:num_upsample_layers]
        layers = []
        current_channels = in_channels
        for i in range(num_upsample_layers):
            out_ch = decoder_channels[i]
            layers.append(SafeUpsample(scale_factor=2, mode='bilinear', align_corners=False))
            layers.append(nn.Conv2d(current_channels, out_ch, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            current_channels = out_ch
        layers.append(nn.Conv2d(current_channels, out_channels, kernel_size=1))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)

class VisionTransformerSAM(nn.Module):
    def __init__(
            self, img_size: int = 1024, patch_size: int = 16, in_chans: int = 3, num_classes: int = 1,
            embed_dim: int = 768, depth: int = 12, num_heads: int = 12, mlp_ratio: float = 4.,
            qkv_bias: bool = True, qk_norm: bool = False, init_values: Optional[float] = None,
            pre_norm: bool = False, drop_rate: float = 0., pos_drop_rate: float = 0.,
            proj_drop_rate: float = 0., attn_drop_rate: float = 0., drop_path_rate: float = 0.,
            embed_layer: Type[nn.Module] = partial(PatchEmbed, output_fmt=Format.NHWC, strict_img_size=False),
            norm_layer: Optional[Type[nn.Module]] = nn.LayerNorm, act_layer: Optional[Type[nn.Module]] = nn.GELU,
            block_fn: Type[nn.Module] = Block, mlp_layer: Type[nn.Module] = Mlp, use_abs_pos: bool = True,
            use_rel_pos: bool = False, use_rope: bool = False, window_size: int = 14,
            global_attn_indexes: Tuple[int, ...] = (), neck_chans: int = 256,
            decoder_channels: Tuple[int, ...] = (128, 64, 32, 16), device=None, dtype=None,
    ):
        super().__init__()
        param_dd = {'device': device, 'dtype': dtype}
        layer_dd = {}
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, bias=not pre_norm)
        grid_size = self.patch_embed.grid_size
        if use_abs_pos:
            self.pos_embed = nn.Parameter(torch.zeros(1, 64, 64, embed_dim, **param_dd))
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        self.norm_pre = norm_layer(embed_dim, **layer_dd) if pre_norm else nn.Identity()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            block_fn(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_norm=qk_norm,
                     init_values=init_values, proj_drop=proj_drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                     norm_layer=norm_layer, act_layer=act_layer, mlp_layer=mlp_layer, use_rel_pos=use_rel_pos,
                     window_size=window_size if i not in global_attn_indexes else 0, input_size=grid_size,
                     rope=None, device=device, dtype=dtype)
            for i in range(depth)])
        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, neck_chans, kernel_size=1, bias=False, **layer_dd),
            LayerNorm2d(neck_chans, **layer_dd),
            nn.Conv2d(neck_chans, neck_chans, kernel_size=3, padding=1, bias=False, **layer_dd),
            LayerNorm2d(neck_chans, **layer_dd),
        )
        self.decoder = SimpleSegmentationDecoder(in_channels=neck_chans, out_channels=num_classes, upsample_factor=self.patch_size, decoder_channels=decoder_channels)

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + resample_abs_pos_embed_nhwc(self.pos_embed, x.shape[1:3])
        x = self.pos_drop(x)
        x = self.norm_pre(x)
        x = self.blocks(x)
        x = self.neck(x.permute(0, 3, 1, 2))
        return x

    def forward(self, x):
        features = self.forward_features(x)
        segmentation_logits = self.decoder(features)
        if segmentation_logits.shape[2:] != x.shape[2:]:
                segmentation_logits = gpu_safe_interpolate(segmentation_logits, size=x.shape[2:], mode='bilinear', align_corners=False)
        return segmentation_logits

class SHFSAMEncoder(nn.Module):
    def __init__(self, img_size: int = 1024, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768, depth: int = 12, num_heads: int = 12, mlp_ratio: float = 4.0, drop_rate: float = 0.0, pos_drop_rate: float = 0.0, drop_path_rate: float = 0.1, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.img_size = img_size
        patch_dim = in_chans * patch_size * patch_size
        self.patch_embed = nn.Linear(patch_dim, embed_dim)
        self.pos_embed = None 
        self.dynamic_pos_embed = SpatioStructuralPosEmbed(embed_dim, img_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, drop_path=dpr[i], use_rel_pos=False, window_size=0, input_size=None)
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight); 
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)

    def forward_features(self, batch_dict: dict) -> torch.Tensor:
        patches = batch_dict['patches']
        x = self.patch_embed(patches.flatten(2))
        patch_pos_embed = self.dynamic_pos_embed(batch_dict['positions'], batch_dict['sizes'])
        x = x + patch_pos_embed
        cls_token_with_pos = self.cls_token + self.cls_pos_embed
        x = torch.cat([cls_token_with_pos.expand(x.shape[0], -1, -1), x], dim=1)
        x = self.pos_drop(x)
        B, L, C = x.shape
        x = x.view(B, 1, L, C)
        for blk in self.blocks: x = blk(x)
        x = x.view(B, L, C)
        x = self.norm(x)
        return x

    def forward(self, batch_dict: dict) -> torch.Tensor:
        return self.forward_features(batch_dict)

def checkpoint_filter_fn(state_dict, model):
    sam_checkpoint = 'image_encoder.patch_embed.proj.weight' in state_dict
    out_dict = {}
    is_segmentation_model = hasattr(model, 'decoder')
    for k, v in state_dict.items():
        if k.startswith('image_encoder.'):
            k = k[14:]
            k = k.replace('mlp.lin', 'mlp.fc')
            if is_segmentation_model and k.startswith('head.'): continue
        elif sam_checkpoint:
            if k.startswith('head.'): continue
            if not k.startswith('patch_embed.') and not k.startswith('pos_embed') and not k.startswith('blocks.') and not k.startswith('neck.'): continue
        out_dict[k] = v
    return out_dict

def _cfg(url='', **kwargs):
    return {'url': url, 'num_classes': 0, 'input_size': (3, 1024, 1024), 'pool_size': None, 'crop_pct': 1.0, 'interpolation': 'bicubic', 'fixed_input_size': True, 'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 'first_conv': 'patch_embed.proj', 'classifier': 'head.fc', **kwargs}

default_cfgs = {'samvit_base_patch16.sa1b': _cfg(url='https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth', hf_hub_id='timm/')}

def create_sam_b_variant(img_size: int, in_chans: int = 3, target_patch_grid_size: int = 64, pretrained: bool = True, num_classes: int = 1, decoder_channels: Tuple[int, ...] = (128, 64, 32, 16), use_abs_pos: bool = True, use_rel_pos: bool = True) -> VisionTransformerSAM:
    new_patch_size = img_size // target_patch_grid_size
    effective_window_size = min(14, target_patch_grid_size)
    model_args = dict(patch_size=new_patch_size, embed_dim=768, depth=12, num_heads=12, global_attn_indexes=[2, 5, 8, 11], window_size=effective_window_size, use_abs_pos=use_abs_pos, use_rel_pos=use_rel_pos, img_size=img_size, in_chans=in_chans, neck_chans=256, num_classes=num_classes, decoder_channels=decoder_channels)
    model = VisionTransformerSAM(**model_args)
    if pretrained:
        cfg = default_cfgs['samvit_base_patch16.sa1b']
        state_dict = load_state_dict_from_url(cfg['url'], map_location='cpu')
        filtered_state_dict = checkpoint_filter_fn(state_dict, model)
        orig_patch_embed_key = 'patch_embed.proj.weight'
        if orig_patch_embed_key in filtered_state_dict:
            orig_patch_embed_weights = filtered_state_dict[orig_patch_embed_key]
            orig_patch_size = 16
            if new_patch_size != orig_patch_size:
                new_patch_embed_weights = F.interpolate(orig_patch_embed_weights, size=(new_patch_size, new_patch_size), mode='bicubic', align_corners=False)
                filtered_state_dict[orig_patch_embed_key] = new_patch_embed_weights
            if in_chans != 3:
                new_weights = filtered_state_dict[orig_patch_embed_key].mean(dim=1, keepdim=True)
                if in_chans > 1: new_weights = new_weights.repeat(1, in_chans, 1, 1)
                filtered_state_dict[orig_patch_embed_key] = new_weights
        if 'pos_embed' in filtered_state_dict and filtered_state_dict['pos_embed'].shape != model.pos_embed.shape:
             pass
        model.load_state_dict(filtered_state_dict, strict=False)
    return model
