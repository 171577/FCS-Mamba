"""
DynamicConvBlock: Context-guided dynamic convolution with dual-branch processing.

This module implements the OverLoCK DynamicConvBlock for unified global-local feature processing
in change detection networks. It uses query-key attention to generate dynamic convolution kernels,
applies dual-branch convolution with different kernel sizes, and incorporates gate-based filtering,
LEPE position encoding, and SE channel attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import natten for neighborhood attention
try:
    from natten import NeighborhoodAttention2D
    from natten.functional import na2d_av
    NATTEN_AVAILABLE = True
except ImportError:
    NATTEN_AVAILABLE = False
    na2d_av = None

# Try to import einops for tensor operations
try:
    from einops import rearrange, einsum
    EINOPS_AVAILABLE = True
except ImportError:
    EINOPS_AVAILABLE = False


class SELayer(nn.Module):
    """Squeeze-and-Excitation channel attention layer."""
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """Apply SE attention to channel features."""
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class DynamicConvBlock(nn.Module):
    """
    Context-guided dynamic convolution block with dual-branch processing.
    
    This module unifies global perception (from context features) with local detail processing
    (from spatial features) through dynamic convolution kernels generated via query-key attention.
    
    Args:
        in_channels (int): Number of input channels for local features
        context_channels (int, optional): Number of channels for context prior. Defaults to in_channels.
        num_heads (int): Number of attention heads. Default: 8
        kernel_size_large (int): Large kernel size for dynamic convolution. Default: 13
        kernel_size_small (int): Small kernel size for dynamic convolution. Default: 5
        mlp_ratio (float): Expansion ratio for MLP layers. Default: 4.0
        drop_rate (float): Dropout rate. Default: 0.0
    """
    
    def __init__(
        self,
        in_channels,
        context_channels=None,
        num_heads=8,
        kernel_size_large=13,
        kernel_size_small=5,
        mlp_ratio=4.0,
        drop_rate=0.0,
    ):
        super().__init__()
        
        # Validate dependencies
        if not NATTEN_AVAILABLE:
            raise ImportError(
                "natten library is required for DynamicConvBlock. "
                "Install with: pip install natten"
            )
        if not EINOPS_AVAILABLE:
            raise ImportError(
                "einops library is required for DynamicConvBlock. "
                "Install with: pip install einops"
            )
        
        # Validate parameters
        assert num_heads > 0, "num_heads must be positive"
        assert kernel_size_large > kernel_size_small, \
            "Large kernel must be larger than small kernel"
        assert 0 <= drop_rate < 1, "drop_rate must be in [0, 1)"
        
        self.in_channels = in_channels
        self.context_channels = context_channels if context_channels is not None else in_channels
        self.num_heads = num_heads
        self.kernel_size_large = kernel_size_large
        self.kernel_size_small = kernel_size_small
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        
        # Working dimension for internal processing
        self.work_dim = in_channels
        
        # Query and key dimensions
        self.query_dim = in_channels // num_heads
        self.key_dim = self.context_channels // num_heads
        
        # Feature fusion layer: concatenate x and h_x, then project
        self.fusion_proj = nn.Conv2d(
            in_channels + self.context_channels,
            self.work_dim,
            kernel_size=1,
            bias=True
        )
        
        # Query-key attention for dynamic kernel generation
        self.weight_query = nn.Linear(in_channels, self.query_dim * num_heads, bias=True)
        self.weight_key = nn.Linear(self.context_channels, self.key_dim * num_heads, bias=True)
        
        # Kernel weight projection
        kernel_weights_dim = (kernel_size_small ** 2) + (kernel_size_large ** 2)
        self.weight_proj = nn.Linear(self.query_dim, kernel_weights_dim, bias=True)
        
        # Scale factor for attention
        self.scale = self.query_dim ** -0.5
        
        # Gate-based noise filtering
        self.gate = nn.Sequential(
            nn.Conv2d(self.work_dim, self.work_dim, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(self.work_dim),
            nn.Sigmoid()
        )
        
        # LEPE (Locally Enhanced Positional Encoding)
        self.lepe = nn.Conv2d(
            self.work_dim,
            self.work_dim,
            kernel_size=3,
            padding=1,
            groups=self.work_dim,
            bias=True
        )
        
        # SE (Squeeze-and-Excitation) channel attention
        self.se_layer = SELayer(self.work_dim, reduction=16)
        
        # Dropout
        self.dropout = nn.Dropout(drop_rate)
    
    def forward(self, x, h_x, h_r=None):
        """
        Forward pass of DynamicConvBlock.
        
        Args:
            x (torch.Tensor): Local features [B, C, H, W]
            h_x (torch.Tensor): Context prior features [B, C_ctx, H_ctx, W_ctx]
            h_r (torch.Tensor, optional): Residual features [B, C, H, W]
        
        Returns:
            torch.Tensor: Processed features [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # Validate input channels
        assert C == self.in_channels, \
            f"Expected {self.in_channels} input channels, got {C}"
        
        # Interpolate context features if spatial dimensions differ
        if h_x.shape[2:] != x.shape[2:]:
            h_x = F.interpolate(
                h_x,
                size=x.shape[2:],
                mode='bilinear',
                align_corners=True
            )
        
        # 1. Feature fusion: concatenate x and h_x, then project
        x_fused = torch.cat([x, h_x], dim=1)
        x_fused = self.fusion_proj(x_fused)
        
        # 2. Dynamic kernel generation via query-key attention
        # Reshape for attention computation
        x_flat = rearrange(x, 'b c h w -> b (h w) c')
        h_x_flat = rearrange(h_x, 'b c h w -> b (h w) c')
        
        # Project to query and key spaces
        query = self.weight_query(x_flat)  # [B, N, query_dim * num_heads]
        key = self.weight_key(h_x_flat)    # [B, N, key_dim * num_heads]
        
        # Reshape for multi-head attention
        query = rearrange(query, 'b n (g d) -> b g n d', g=self.num_heads)
        key = rearrange(key, 'b n (g d) -> b g n d', g=self.num_heads)
        
        # Compute attention weights
        weight = einsum(query, key, 'b g n d, b g m d -> b g n m') * self.scale
        weight = torch.clamp(weight, min=-10, max=10)
        weight = F.softmax(weight, dim=-1)
        
        # Project weights to kernel space
        weight_flat = rearrange(weight, 'b g n m -> b (g n) m')
        weight_proj = self.weight_proj(weight_flat)  # [B, N, kernel_weights_dim]
        
        # 3. Split weights for dual-branch convolution
        kernel_small_size = self.kernel_size_small ** 2
        kernel_large_size = self.kernel_size_large ** 2
        
        weight_small = weight_proj[:, :, :kernel_small_size]  # [B, N, small_kernel_size²]
        weight_large = weight_proj[:, :, kernel_small_size:]  # [B, N, large_kernel_size²]
        
        # Normalize weights
        weight_small = F.softmax(weight_small, dim=-1)
        weight_large = F.softmax(weight_large, dim=-1)
        
        # 4. Apply neighborhood attention with dynamic weights
        # Reshape fused features for neighborhood attention
        value = rearrange(x_fused, 'b c h w -> b (h w) c')
        
        # Apply small kernel branch
        x_small = na2d_av(
            weight_small.reshape(B, H, W, kernel_small_size),
            value.reshape(B, H, W, self.work_dim),
            kernel_size=self.kernel_size_small
        )
        x_small = rearrange(x_small, 'b h w c -> b c h w')
        
        # Apply large kernel branch
        x_large = na2d_av(
            weight_large.reshape(B, H, W, kernel_large_size),
            value.reshape(B, H, W, self.work_dim),
            kernel_size=self.kernel_size_large
        )
        x_large = rearrange(x_large, 'b h w c -> b c h w')
        
        # Combine branches
        x_out = x_small + x_large
        
        # 5. Gate-based filtering
        gate = self.gate(x_out)
        x_out = gate * x_out
        
        # 6. LEPE position encoding
        x_out = x_out + self.lepe(x_out)
        
        # 7. SE channel attention
        x_out = self.se_layer(x_out)
        
        # 8. Dropout
        x_out = self.dropout(x_out)
        
        # 9. Residual connection
        if h_r is not None:
            assert h_r.shape == x_out.shape, \
                f"Residual shape {h_r.shape} doesn't match output shape {x_out.shape}"
            x_out = x_out + h_r
        
        return x_out
