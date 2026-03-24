import torch
from torch import nn
from timm.models.vision_transformer import VisionTransformer 

class SpatioStructuralPosEmbed(nn.Module):
    """
    Spatio-Structural Positional Embedding for SHF QuadTree nodes.
    Projects geometric features (center_x, center_y, size) into the embedding space.
    """
    def __init__(self, embed_dim, img_size=1024):
        super().__init__()
        self.img_size = float(img_size)
        
        # An MLP to project the geometric features [cx, cy, size] into the embedding space
        self.proj = nn.Sequential(
            nn.Linear(3, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )

    def forward(self, positions, sizes):
        # Normalize features to a [0, 1] range based on full image size
        # positions are (cx, cy)
        pos_norm = positions / self.img_size
        sizes_norm = sizes.unsqueeze(-1) / self.img_size
        
        # Concatenate features and project
        features = torch.cat([pos_norm, sizes_norm], dim=-1)
        pos_embed = self.proj(features)
        
        return pos_embed

class SHFVisionTransformer(VisionTransformer):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 num_classes=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 **kwargs):
        
        # --- 1. Call the Parent Constructor ---
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            **kwargs
        )
        
        # --- 2. Override Components for SHF Data Format ---
        patch_dim = in_channels * patch_size * patch_size
        self.patch_embed = nn.Linear(patch_dim, embed_dim)
        
        self.pos_embed = None 
        self.dynamic_pos_embed = SpatioStructuralPosEmbed(embed_dim, img_size)
        self.cls_pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _pos_embed(self, x: torch.Tensor, positions: torch.Tensor, sizes: torch.Tensor) -> torch.Tensor:
        patch_pos_embed = self.dynamic_pos_embed(positions, sizes)
        x = x + patch_pos_embed
        cls_token_with_pos = self.cls_token + self.cls_pos_embed
        x = torch.cat([cls_token_with_pos.expand(x.shape[0], -1, -1), x], dim=1)
        return self.pos_drop(x)

    def forward_features(self, batch_dict: dict) -> torch.Tensor:
        patches = batch_dict['patches']
        x = self.patch_embed(patches.flatten(2))
        x = self._pos_embed(x, batch_dict['positions'], batch_dict['sizes'])
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward(self, batch_dict: dict) -> torch.Tensor:
        x = self.forward_features(batch_dict)
        x = self.forward_head(x)
        return x
