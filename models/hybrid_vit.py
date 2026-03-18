import torch 
import torch.nn as nn
from .vit import VisionTransformer
from .efficient_kan import EfficientKAN
from .wavelet_kan import WaveletKAN

class HybKANViT(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., hybrid_type=1, wavelet_type='dog', qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__(img_size, patch_size, in_chans, num_classes, embed_dim, depth, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate, drop_path_rate)
        
        self.blocks = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        for i in range(depth):
            if hybrid_type == 1:
                mlp_block = WaveletKAN(
                    in_dim=embed_dim,
                    out_dim=int(embed_dim * mlp_ratio),
                    wavelet_type=wavelet_type
                )
            else:
                mlp_block = EfficientKAN(
                    in_dim=embed_dim,
                    out_dim=int(embed_dim * mlp_ratio)
                )
            
            block = ViTBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                mlp=mlp_block
            )
            self.blocks.append(block)
        
        if hybrid_type == 1:
            self.head = EfficientKAN(
                in_dim=embed_dim,
                out_dim=num_classes
            )
        else:
            self.head = WaveletKAN(
                in_dim=embed_dim,
                out_dim=num_classes,
                wavelet_type=wavelet_type
            )

class ViTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., drop_path=0., mlp=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = mlp or nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
