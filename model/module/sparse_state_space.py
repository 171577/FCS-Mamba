import math
import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class WithBiasLayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight + self.bias


class ChannelLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.body = WithBiasLayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


def pairwise_cos_sim(x1: torch.Tensor, x2: torch.Tensor):
    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)
    return torch.matmul(x1, x2.transpose(-2, -1))


class SparseStateSpaceModule(nn.Module):
    def __init__(
        self,
        d_model,
        proposal_hw,
        fold_hw,
        heads,
        d_state=8,
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.0,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.proposal_hw = proposal_hw
        self.fold_hw = fold_hw
        self.heads = heads
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model) // self.heads
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.x_conv = nn.Conv1d(
            in_channels=(self.dt_rank + self.d_state * 2),
            out_channels=(self.dt_rank + self.d_state * 2),
            kernel_size=7,
            padding=3,
            groups=(self.dt_rank + self.d_state * 2),
        )

        self.dt_projs = (
            self.dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            ),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

        self.f = nn.Conv2d(self.d_inner, self.d_inner * self.heads, kernel_size=1)
        self.proj = nn.Conv2d(self.d_inner * self.heads, self.d_inner, kernel_size=1)
        self.v = nn.Conv2d(self.d_inner, self.d_inner * self.heads, kernel_size=1)
        self.sim_alpha = nn.Parameter(torch.ones(1))
        self.sim_beta = nn.Parameter(torch.zeros(1))
        self.centers_proposal = nn.AdaptiveAvgPool2d((self.proposal_hw, self.proposal_hw))

    @staticmethod
    def dt_init(
        dt_rank,
        d_inner,
        dt_scale=1.0,
        dt_init="random",
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        **factory_kwargs,
    ):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        a = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        a_log = torch.log(a)
        if copies > 1:
            a_log = repeat(a_log, "d n -> r d n", r=copies)
            if merge:
                a_log = a_log.flatten(0, 1)
        a_log = nn.Parameter(a_log)
        a_log._no_weight_decay = True
        return a_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        d = torch.ones(d_inner, device=device)
        if copies > 1:
            d = repeat(d, "n1 -> r n1", r=copies)
            if merge:
                d = d.flatten(0, 1)
        d = nn.Parameter(d)
        d._no_weight_decay = True
        return d

    def forward_core(self, x):
        value = self.v(x)
        x = self.f(x)
        x = rearrange(x, "b (e c) w h -> (b e) c w h", e=self.heads)
        value = rearrange(value, "b (e c) w h -> (b e) c w h", e=self.heads)

        if self.fold_hw > 1:
            b0, c0, w0, h0 = x.shape
            assert w0 % self.fold_hw == 0 and h0 % self.fold_hw == 0, (
                f"Ensure the feature map size ({w0}*{h0}) can be divided by fold "
                f"{self.fold_hw}*{self.fold_hw}"
            )
            x = rearrange(
                x,
                "b c (f1 w) (f2 h) -> (b f1 f2) c w h",
                f1=self.fold_hw,
                f2=self.fold_hw,
            )
            value = rearrange(
                value,
                "b c (f1 w) (f2 h) -> (b f1 f2) c w h",
                f1=self.fold_hw,
                f2=self.fold_hw,
            )

        b, c, w, h = x.shape
        centers = self.centers_proposal(x)
        value_centers = rearrange(self.centers_proposal(value), "b c w h -> b (w h) c")

        b, c, ww, hh = centers.shape
        sim = torch.sigmoid(
            self.sim_beta
            + self.sim_alpha
            * pairwise_cos_sim(
                centers.reshape(b, c, -1).permute(0, 2, 1),
                x.reshape(b, c, -1).permute(0, 2, 1),
            )
        )

        _, sim_max_idx = sim.max(dim=1, keepdim=True)
        mask = torch.zeros_like(sim)
        mask.scatter_(1, sim_max_idx, 1.0)
        sim = sim * mask

        value2 = rearrange(value, "b c w h -> b (w h) c")
        out = ((value2.unsqueeze(dim=1) * sim.unsqueeze(dim=-1)).sum(dim=2) + value_centers) / (
            sim.sum(dim=-1, keepdim=True) + 1.0
        )

        b, l, c = out.shape
        k = 1
        xs = rearrange(out, "b l c -> b c l")
        xs = torch.stack([xs], dim=1).view(b, 1, -1, l)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(b, k, -1, l), self.x_proj_weight)
        x_dbl = self.x_conv(x_dbl.squeeze(1)).unsqueeze(1)
        dts, bs_, cs_ = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(b, k, -1, l), self.dt_projs_weight)

        xs = xs.float().view(b, -1, l)
        dts = dts.contiguous().float().view(b, -1, l)
        bs_ = bs_.float().view(b, k, -1, l)
        cs_ = cs_.float().view(b, k, -1, l)
        ds = self.Ds.float().view(-1)
        as_ = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs,
            dts,
            as_,
            bs_,
            cs_,
            ds,
            z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(b, k, -1, l)
        out = rearrange(out_y[:, 0], "b c l -> b l c")

        out = (out.unsqueeze(dim=2) * sim.unsqueeze(dim=-1)).sum(dim=1)
        out = rearrange(out, "b (w h) c -> b c w h", w=w)

        if self.fold_hw > 1:
            out = rearrange(
                out,
                "(b f1 f2) c w h -> b c (f1 w) (f2 h)",
                f1=self.fold_hw,
                f2=self.fold_hw,
            )

        out = rearrange(out, "(b e) c w h -> b (e c) w h", e=self.heads)
        out = self.proj(out)
        return out

    def forward(self, x: torch.Tensor, **kwargs):
        x = rearrange(x, "b c h w -> b h w c")
        b, h, w, c = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))

        y = self.forward_core(x)
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(b, h, w, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)

        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        out = rearrange(out, "b h w c -> b c h w")
        return out
