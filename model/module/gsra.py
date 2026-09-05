import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def lambda_init_fn(depth: int) -> float:
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


class DifferentialLinearProjection(nn.Module):
    """Modal-specific differential projection (base mode)."""

    def __init__(self, dim, heads=8, dim_head=64, bias=True, geo_dim=3, dino_dim=1024):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.head_dim = dim_head

        self.to_q = nn.Linear(dim, inner_dim, bias=bias)
        self.to_kv_geometric = nn.Linear(dim, inner_dim * 2, bias=bias)
        self.to_kv_semantic = nn.Linear(dim, inner_dim * 2, bias=bias)

        self.geo_proj = nn.Linear(geo_dim, dim, bias=bias)
        self.dino_proj = nn.Linear(dino_dim, dim, bias=bias)

        self.geo_weight = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.sem_weight = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, x, geo_feat, dino_feat, attn_kv=None):
        bsz, n_tokens, _ = x.shape
        attn_kv = x if attn_kv is None else attn_kv

        q = self.to_q(x).reshape(bsz, n_tokens, self.heads, self.head_dim).permute(0, 2, 1, 3)

        geo_feat_proj = self.geo_proj(geo_feat)
        dino_feat_proj = self.dino_proj(dino_feat)

        geo_enhanced = attn_kv + self.geo_weight * geo_feat_proj
        semantic_enhanced = attn_kv + self.sem_weight * dino_feat_proj

        kv_geo = self.to_kv_geometric(geo_enhanced).reshape(bsz, n_tokens, 2, self.heads, self.head_dim)
        kv_sem = self.to_kv_semantic(semantic_enhanced).reshape(bsz, n_tokens, 2, self.heads, self.head_dim)

        kv_geo = kv_geo.permute(2, 0, 3, 1, 4)
        kv_sem = kv_sem.permute(2, 0, 3, 1, 4)

        k_geo, v_geo = kv_geo[0], kv_geo[1]
        k_sem, v_sem = kv_sem[0], kv_sem[1]
        return q, k_geo, v_geo, k_sem, v_sem


class DifferentialLinearProjectionConcatKV(nn.Module):
    """Concat mode: keep base KV and append modal-specific KVs."""

    def __init__(self, dim, heads=8, dim_head=64, bias=True, geo_dim=3, dino_dim=1024):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.head_dim = dim_head

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=bias)
        self.to_kv_geometric = nn.Linear(dim, inner_dim * 2, bias=bias)
        self.to_kv_semantic = nn.Linear(dim, inner_dim * 2, bias=bias)

        self.geo_proj = nn.Linear(geo_dim, dim, bias=bias)
        self.dino_proj = nn.Linear(dino_dim, dim, bias=bias)

        self.geo_weight = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.sem_weight = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, x, geo_feat, dino_feat, attn_kv=None):
        bsz, n_tokens, _ = x.shape
        attn_kv = x if attn_kv is None else attn_kv

        qkv = self.to_qkv(x).reshape(bsz, n_tokens, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k_base, v_base = qkv[0], qkv[1], qkv[2]

        geo_feat_proj = self.geo_proj(geo_feat)
        dino_feat_proj = self.dino_proj(dino_feat)

        geo_enhanced = attn_kv + self.geo_weight * geo_feat_proj
        semantic_enhanced = attn_kv + self.sem_weight * dino_feat_proj

        kv_geo = self.to_kv_geometric(geo_enhanced).reshape(bsz, n_tokens, 2, self.heads, self.head_dim)
        kv_sem = self.to_kv_semantic(semantic_enhanced).reshape(bsz, n_tokens, 2, self.heads, self.head_dim)

        kv_geo = kv_geo.permute(2, 0, 3, 1, 4)
        kv_sem = kv_sem.permute(2, 0, 3, 1, 4)

        k_geo, v_geo = kv_geo[0], kv_geo[1]
        k_sem, v_sem = kv_sem[0], kv_sem[1]
        return q, k_base, v_base, k_geo, v_geo, k_sem, v_sem


class DifferentialWindowAttention(nn.Module):
    """GSRA core attention over window tokens."""

    def __init__(
        self,
        dim,
        num_heads,
        depth=1,
        token_projection="base",
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        geo_dim=3,
        dino_dim=1024,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.token_projection = token_projection

        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.ones(1) * 0.5)
        self.lambda_k1 = nn.Parameter(torch.ones(1) * 0.5)

        if token_projection == "concat":
            self.qkv = DifferentialLinearProjectionConcatKV(
                dim,
                heads=num_heads,
                dim_head=self.head_dim,
                bias=qkv_bias,
                geo_dim=geo_dim,
                dino_dim=dino_dim,
            )
            self.branch_logits = nn.Parameter(torch.zeros(3, dtype=torch.float32))
        else:
            self.qkv = DifferentialLinearProjection(
                dim,
                heads=num_heads,
                dim_head=self.head_dim,
                bias=qkv_bias,
                geo_dim=geo_dim,
                dino_dim=dino_dim,
            )
            self.branch_logits = nn.Parameter(torch.zeros(2, dtype=torch.float32))

        self.subln = nn.LayerNorm(dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def _scaled_attn(self, q, k):
        return torch.matmul(q, k.transpose(-2, -1))

    def forward(self, x, dino_mat, point_feature, attn_kv=None):
        bsz, n_tokens, _ = x.shape

        if self.token_projection == "concat":
            q, k_base, v_base, k_geo, v_geo, k_sem, v_sem = self.qkv(x, point_feature, dino_mat, attn_kv)
            q = q * self.scale

            attn_base = self.softmax(self._scaled_attn(q, k_base))
            attn_geo = self.softmax(self._scaled_attn(q, k_geo))
            attn_sem = self.softmax(self._scaled_attn(q, k_sem))

            lambda_val = torch.sigmoid(self.lambda_q1 * self.lambda_k1) + self.lambda_init
            attn_diff = attn_sem - lambda_val * attn_geo

            attn_base = self.attn_drop(attn_base)
            attn_geo = self.attn_drop(attn_geo)
            attn_diff = self.attn_drop(attn_diff)

            x_base = torch.matmul(attn_base, v_base)
            x_geo = torch.matmul(attn_geo, v_geo)
            x_diff = torch.matmul(attn_diff, v_sem)

            branch_w = F.softmax(self.branch_logits, dim=0)
            x_out = branch_w[0] * x_base + branch_w[1] * x_geo + branch_w[2] * x_diff
        else:
            q, k_geo, v_geo, k_sem, v_sem = self.qkv(x, point_feature, dino_mat, attn_kv)
            q = q * self.scale

            attn_geo = self.softmax(self._scaled_attn(q, k_geo))
            attn_sem = self.softmax(self._scaled_attn(q, k_sem))

            lambda_val = torch.sigmoid(self.lambda_q1 * self.lambda_k1) + self.lambda_init
            attn_diff = attn_sem - lambda_val * attn_geo

            attn_geo = self.attn_drop(attn_geo)
            attn_diff = self.attn_drop(attn_diff)

            x_geo = torch.matmul(attn_geo, v_geo)
            x_diff = torch.matmul(attn_diff, v_sem)

            branch_w = F.softmax(self.branch_logits, dim=0)
            x_out = branch_w[0] * x_geo + branch_w[1] * x_diff

        x_out = x_out.transpose(1, 2).contiguous().view(bsz, n_tokens, self.dim)
        x_out = self.subln(x_out)
        x_out = x_out * (1.0 - self.lambda_init)
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        return x_out

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, num_heads={self.num_heads}, head_dim={self.head_dim}, "
            f"lambda_init={self.lambda_init:.3f}, token_projection={self.token_projection}"
        )


class GSRA2DBlock(nn.Module):
    """2D wrapper for GSRA. Input/Output: [B, C, H, W]."""

    def __init__(
        self,
        dim,
        num_heads=4,
        window_size=8,
        token_projection="base",
        depth=1,
        geo_dim=3,
        dino_dim=1024,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.window_size = int(window_size)
        self.dino_dim = int(dino_dim)
        self.semantic_adapter = nn.Linear(dim, self.dino_dim, bias=True)
        self.attn = DifferentialWindowAttention(
            dim=dim,
            num_heads=num_heads,
            depth=depth,
            token_projection=token_projection,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            geo_dim=geo_dim,
            dino_dim=self.dino_dim,
        )
        self.res_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def _window_partition(self, x, ws):
        # x: [B, C, H, W] -> [B*nW, ws*ws, C]
        bsz, channels, h, w = x.shape
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        hp, wp = x.shape[2], x.shape[3]

        x_hwc = x.permute(0, 2, 3, 1).contiguous()
        x_win = x_hwc.view(bsz, hp // ws, ws, wp // ws, ws, channels)
        x_win = x_win.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws * ws, channels)
        return x_win, (h, w, hp, wp, pad_h, pad_w)

    def _window_reverse(self, x_win, shape_meta, ws, channels):
        h, w, hp, wp, pad_h, pad_w = shape_meta
        bsz = x_win.shape[0] // ((hp // ws) * (wp // ws))

        x = x_win.view(bsz, hp // ws, wp // ws, ws, ws, channels)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(bsz, hp, wp, channels)
        x = x.permute(0, 3, 1, 2).contiguous()
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :h, :w]
        return x

    def _build_geometric_feature(self, x_tokens, ws):
        # Geometry cue: normalized (x, y) coordinates + token intensity.
        bwn, n_tokens, _ = x_tokens.shape
        device = x_tokens.device
        dtype = x_tokens.dtype

        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, ws, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, ws, device=device, dtype=dtype),
            indexing="ij",
        )
        coord = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)  # [N, 2]
        coord = coord.unsqueeze(0).expand(bwn, -1, -1)

        intensity = x_tokens.mean(dim=-1, keepdim=True)
        return torch.cat([coord, intensity], dim=-1)

    def forward(self, x):
        bsz, channels, h, w = x.shape
        ws = max(1, min(self.window_size, h, w))

        x_tokens, shape_meta = self._window_partition(x, ws)
        dino_tokens = self.semantic_adapter(x_tokens)
        geo_tokens = self._build_geometric_feature(x_tokens, ws)

        x_rect = self.attn(x_tokens, dino_tokens, geo_tokens)
        x_rect = self._window_reverse(x_rect, shape_meta, ws, channels)

        return x + self.res_scale * x_rect
