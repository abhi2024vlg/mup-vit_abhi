import math
from collections import OrderedDict
from functools import partial
from typing import Callable, Optional
import torch
import torch.nn as nn
from torchvision.models.vision_transformer import MLPBlock
import torch.nn.functional as F
from typing import List

# # Taken from https://github.com/lucidrains/vit-pytorch, likely ported from https://github.com/google-research/big_vision/
# def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
#     y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
#     assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
#     omega = torch.arange(dim // 4) / (dim // 4 - 1)
#     omega = 1.0 / (temperature ** omega)

#     y = y.flatten()[:, None] * omega[None, :]
#     x = x.flatten()[:, None] * omega[None, :]
#     pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
#     return pe.type(dtype)


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

        # Fix init discrepancy between nn.MultiheadAttention and that of big_vision
        bound = math.sqrt(3 / hidden_dim)
        nn.init.uniform_(self.self_attention.in_proj_weight, -bound, bound)
        nn.init.uniform_(self.self_attention.out_proj.weight, -bound, bound)
    
    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x, _ = self.self_attention(x, x, x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y

class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        return self.ln(self.layers(self.dropout(input)))

class DilatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))

class TokenGenerator(nn.Module):
    def __init__(self, image_size=256, patch_size=16, in_channels=3, embed_dim=256):
        super().__init__()
        
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        self.stage1 = DilatedConvBlock(embed_dim, embed_dim, dilation=1)
        self.stage2 = DilatedConvBlock(embed_dim, embed_dim, dilation=4)
        
        self._init_weights()
        
    def _init_weights(self):
        # Initialize patch_embed (equivalent to conv_proj in the original code)
        fan_in = self.patch_embed.in_channels * self.patch_embed.kernel_size[0] * self.patch_embed.kernel_size[1]
        std = math.sqrt(1 / fan_in) / .87962566103423978
        nn.init.trunc_normal_(self.patch_embed.weight, std=std, a=-2 * std, b=2 * std)
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)
        
        # Initialize dilated convolutions (similar to conv_last in the original code)
        for m in [self.stage1.conv, self.stage2.conv]:
            nn.init.normal_(m.weight, mean=0.0, std=math.sqrt(2.0 / m.out_channels))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        
        # Initialize batch norm layers
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        # Initial patching: 16x16 patches of 16x16 resolution
        print(x.shape)
        x = self.patch_embed(x)  # Shape: [B, embed_dim, 16, 16]
        tokens_stage1 = x.flatten(2).transpose(1, 2)  # Shape: [B, 256, embed_dim]
        print(tokens_stage1.shape)
        # Stage 1: 4x4 patches with 64x64 receptive field
        x = self.stage1(x)
        x = F.avg_pool2d(x, kernel_size=4)  # Shape: [B, embed_dim, 4, 4]
        tokens_stage2 = x.flatten(2).transpose(1, 2)  # Shape: [B, 16, embed_dim]
        print(tokens_stage2.shape)
        # Stage 2: 1x1 patch with global receptive field
        x = self.stage2(x)
        x = F.adaptive_avg_pool2d(x, 1)  # Shape: [B, embed_dim, 1, 1]
        tokens_stage3 = x.flatten(2).transpose(1, 2)  # Shape: [B, 1, embed_dim]
        print(tokens_stage3.shape)
        # Combine tokens from all stages
        tokens = torch.cat([tokens_stage1, tokens_stage2, tokens_stage3], dim=1)  # Shape: [B, 273, embed_dim]
        
        return tokens

class SimpleVisionTransformer(nn.Module):
    """Vision Transformer modified per https://arxiv.org/abs/2205.01580."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 1000,
        seq_length: int = 273,  # 256 + 16 + 1
        representation_size: Optional[int] = None,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        
        torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.representation_size = representation_size
        self.norm_layer = norm_layer

        # Add TokenGenerator
        self.token_generator = TokenGenerator(image_size=image_size, patch_size=patch_size, in_channels=3, embed_dim=hidden_dim)
    
        # Update seq_length to match the number of tokens from TokenGenerator
        self.seq_length = seq_length
        
        self.encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
        )

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        if representation_size is None:
            heads_layers["head"] = nn.Linear(hidden_dim, num_classes)
        else:
            heads_layers["pre_logits"] = nn.Linear(hidden_dim, representation_size)
            heads_layers["act"] = nn.Tanh()
            heads_layers["head"] = nn.Linear(representation_size, num_classes)

        self.heads = nn.Sequential(heads_layers)
        
        # Initialize weights for the heads
        if hasattr(self.heads, "pre_logits") and isinstance(self.heads.pre_logits, nn.Linear):
            fan_in = self.heads.pre_logits.in_features
            nn.init.trunc_normal_(self.heads.pre_logits.weight, std=math.sqrt(1 / fan_in))
            nn.init.zeros_(self.heads.pre_logits.bias)

        if isinstance(self.heads.head, nn.Linear):
            nn.init.zeros_(self.heads.head.weight)
            nn.init.zeros_(self.heads.head.bias)

    def forward(self, x: torch.Tensor):
        # Use TokenGenerator to get tokens
        x = self.token_generator(x)  # Shape: [B, 273, hidden_dim]
        
        # No need for position embeddings as they're implicitly handled by TokenGenerator
        
        x = self.encoder(x)
        x = x.mean(dim=1)
        x = self.heads(x)
        
        return x
