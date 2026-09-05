import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

try:
    from .SDA_MDEM import MDEM
except ImportError:
    try:
        from model.module.SDA_MDEM import MDEM
    except ImportError:
        from SDA_MDEM import MDEM


def _try_import_vmamba():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    mamba_models_dir = os.path.join(repo_root, "ChangeMamba-master", "classification", "models")
    if os.path.isdir(mamba_models_dir) and (mamba_models_dir not in sys.path):
        sys.path.append(mamba_models_dir)
    try:
        from vmamba import VSSBlock, LayerNorm2d
        return VSSBlock, LayerNorm2d
    except Exception:
        return None, None


class MambaBranch_MultiScale(nn.Module):
    def __init__(
        self,
        in_channels_list,
        out_channels_list,
        n_blocks_per_scale=1,
        use_weak_guidance=True,
        weak_conf_gamma=2.0,
        use_mdem=False,
        use_sg_delta=False,
        use_mask_scan=False,
        use_cross_gate=False,
        sg_delta_scale=0.3,
        mask_scan_thresh=0.6,
        mamba_forward_type="v3noz",
        mamba_ssm_d_state=16,
        mamba_ssm_ratio=2.0,
        mamba_ssm_dt_rank="auto",
        mamba_ssm_conv=3,
        mamba_ssm_conv_bias=True,
        mamba_ssm_drop_rate=0.0,
        mamba_ssm_init="v0",
        mamba_mlp_ratio=4.0,
        mamba_mlp_drop_rate=0.0,
        drop_path=0.0,
    ):
        super().__init__()

        VSSBlock, LayerNorm2d = _try_import_vmamba()
        if VSSBlock is None or LayerNorm2d is None:
            raise ImportError(
                "Failed to import vmamba (VSSBlock/LayerNorm2d). "
                "Ensure ChangeMamba-master exists and dependencies (timm, fvcore, triton) are installed."
            )

        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        self.use_weak_guidance = use_weak_guidance
        self.weak_conf_gamma = float(weak_conf_gamma)
        self.use_mdem = use_mdem
        self.use_sg_delta = bool(use_sg_delta)
        self.use_mask_scan = bool(use_mask_scan)
        self.use_cross_gate = bool(use_cross_gate)
        self.mask_scan_thresh = float(mask_scan_thresh)

        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch * 3 + (1 if use_weak_guidance else 0), out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
            )
            for in_ch, out_ch in zip(in_channels_list, out_channels_list)
        ])

        self.fuse_topdown = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels_list[i + 1] + out_channels_list[i], out_channels_list[i], 1, bias=False),
                nn.BatchNorm2d(out_channels_list[i]),
                nn.GELU(),
            )
            for i in range(3)
        ])

        self.fuse_bottomup = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels_list[i] + out_channels_list[i + 1], out_channels_list[i + 1], 1, bias=False),
                nn.BatchNorm2d(out_channels_list[i + 1]),
                nn.GELU(),
                nn.Conv2d(
                    out_channels_list[i + 1],
                    out_channels_list[i + 1],
                    3,
                    padding=1,
                    groups=out_channels_list[i + 1],
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels_list[i + 1]),
                nn.GELU(),
            )
            for i in range(3)
        ])

        self.blocks = nn.ModuleList()
        for i, dim in enumerate(out_channels_list):
            stage = []
            for b in range(n_blocks_per_scale):
                stage.append(
                    VSSBlock(
                        hidden_dim=dim,
                        drop_path=drop_path,
                        norm_layer=LayerNorm2d,
                        channel_first=True,
                        ssm_d_state=mamba_ssm_d_state,
                        ssm_ratio=mamba_ssm_ratio,
                        ssm_dt_rank=mamba_ssm_dt_rank,
                        ssm_act_layer=nn.SiLU,
                        ssm_conv=mamba_ssm_conv,
                        ssm_conv_bias=mamba_ssm_conv_bias,
                        ssm_drop_rate=mamba_ssm_drop_rate,
                        ssm_init=mamba_ssm_init,
                        forward_type=mamba_forward_type,
                        mlp_ratio=mamba_mlp_ratio,
                        mlp_act_layer=nn.GELU,
                        mlp_drop_rate=mamba_mlp_drop_rate,
                        gmlp=False,
                        use_checkpoint=False,
                        post_norm=False,
                    )
                )
            self.blocks.append(nn.Sequential(*stage))

        if self.use_sg_delta:
            self.sg_delta_adapters = nn.ModuleList([
                nn.Conv2d(1, out_ch, kernel_size=1, bias=True)
                for out_ch in out_channels_list
            ])
            self.sg_delta_alpha = nn.Parameter(torch.full((len(out_channels_list),), float(sg_delta_scale)))
        else:
            self.sg_delta_adapters = None
            self.sg_delta_alpha = None

        if self.use_mask_scan:
            self.mask_scan_bypass = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, groups=out_ch, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.GELU(),
                    nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.GELU(),
                )
                for out_ch in out_channels_list
            ])
        else:
            self.mask_scan_bypass = None

        if self.use_cross_gate:
            self.cross_gate_generators = nn.ModuleList([
                nn.Conv2d(in_ch * 2 + 1, 1, kernel_size=1, bias=True)
                for in_ch in in_channels_list
            ])
        else:
            self.cross_gate_generators = None

        # Initialize MDEM modules if enabled
        if self.use_mdem:
            # MDEM for each scale: input_dim and diff_dim are both out_channels
            self.mdem1 = MDEM(out_channels_list[0], out_channels_list[0], ds=4, beta_init=0.2)
            self.mdem2 = MDEM(out_channels_list[1], out_channels_list[1], ds=4, beta_init=0.3)
            self.mdem3 = MDEM(out_channels_list[2], out_channels_list[2], ds=8, beta_init=0.4)
            self.mdem4 = MDEM(out_channels_list[3], out_channels_list[3], ds=8, beta_init=0.5)

    def _make_input(self, x1, x2, sim=None):
        diff = torch.abs(x1 - x2)
        if self.use_weak_guidance:
            if sim is None:
                sim = torch.zeros(
                    (x1.shape[0], 1, x1.shape[2], x1.shape[3]),
                    device=x1.device,
                    dtype=x1.dtype,
                )
            else:
                sim = F.interpolate(sim, size=x1.shape[2:], mode="bilinear", align_corners=True)
                conf = (sim - 0.5).abs() * 2.0
                gate = conf.clamp(min=0.0, max=1.0).pow(self.weak_conf_gamma)
                sim = sim * gate
            return torch.cat([x1, x2, diff, sim], dim=1)

        return torch.cat([x1, x2, diff], dim=1)

    def _resize_sim(self, sim, ref):
        if sim is None:
            return torch.zeros((ref.shape[0], 1, ref.shape[2], ref.shape[3]), device=ref.device, dtype=ref.dtype)
        return F.interpolate(sim, size=ref.shape[2:], mode="bilinear", align_corners=True)

    def _apply_cross_gate(self, x1, x2, sim, scale_idx):
        if (not self.use_cross_gate) or (self.cross_gate_generators is None):
            return x1, x2

        sim_r = self._resize_sim(sim, x1)
        gate_logits = self.cross_gate_generators[scale_idx](torch.cat([x1, x2, sim_r], dim=1))
        gate = torch.sigmoid(gate_logits)

        x1_g = gate * x1 + (1.0 - gate) * x2
        x2_g = gate * x2 + (1.0 - gate) * x1
        return x1_g, x2_g

    def _apply_sg_delta(self, feat, sim, scale_idx):
        if (not self.use_sg_delta) or (self.sg_delta_adapters is None):
            return feat

        sim_r = self._resize_sim(sim, feat)
        delta = torch.tanh(self.sg_delta_adapters[scale_idx](sim_r))
        alpha = torch.tanh(self.sg_delta_alpha[scale_idx]).view(1, 1, 1, 1)
        return feat * (1.0 + alpha * delta)

    def _apply_mask_scan(self, full_out, light_out, sim):
        if not self.use_mask_scan:
            return full_out

        sim_r = self._resize_sim(sim, full_out)
        conf = (sim_r - 0.5).abs() * 2.0
        hard_mask = torch.sigmoid((conf - self.mask_scan_thresh) * 10.0)
        return hard_mask * full_out + (1.0 - hard_mask) * light_out

    def forward(
        self,
        x1_1, x1_2, x1_3, x1_4,
        x2_1, x2_2, x2_3, x2_4,
        s_wsi_list=None,
        **kwargs,
    ):
        sim4 = sim3 = sim2 = sim1 = None
        if self.use_weak_guidance and s_wsi_list is not None:
            sim4, sim3, sim2, sim1 = s_wsi_list

        x1_1, x2_1 = self._apply_cross_gate(x1_1, x2_1, sim1, 0)
        x1_2, x2_2 = self._apply_cross_gate(x1_2, x2_2, sim2, 1)
        x1_3, x2_3 = self._apply_cross_gate(x1_3, x2_3, sim3, 2)
        x1_4, x2_4 = self._apply_cross_gate(x1_4, x2_4, sim4, 3)

        d1 = self.proj[0](self._make_input(x1_1, x2_1, sim1))
        d2 = self.proj[1](self._make_input(x1_2, x2_2, sim2))
        d3 = self.proj[2](self._make_input(x1_3, x2_3, sim3))
        d4 = self.proj[3](self._make_input(x1_4, x2_4, sim4))

        d3 = self.fuse_topdown[2](
            torch.cat(
                [d3, F.interpolate(d4, size=d3.shape[2:], mode="bilinear", align_corners=True)],
                dim=1,
            )
        )

        d2 = self.fuse_topdown[1](
            torch.cat(
                [d2, F.interpolate(d3, size=d2.shape[2:], mode="bilinear", align_corners=True)],
                dim=1,
            )
        )

        d1 = self.fuse_topdown[0](
            torch.cat(
                [d1, F.interpolate(d2, size=d1.shape[2:], mode="bilinear", align_corners=True)],
                dim=1,
            )
        )

        d2 = self.fuse_bottomup[0](
            torch.cat(
                [d2, F.adaptive_avg_pool2d(d1, d2.shape[2:])],
                dim=1,
            )
        )

        d3 = self.fuse_bottomup[1](
            torch.cat(
                [d3, F.adaptive_avg_pool2d(d2, d3.shape[2:])],
                dim=1,
            )
        )

        d4 = self.fuse_bottomup[2](
            torch.cat(
                [d4, F.adaptive_avg_pool2d(d3, d4.shape[2:])],
                dim=1,
            )
        )

        d1 = self._apply_sg_delta(d1, sim1, 0)
        d2 = self._apply_sg_delta(d2, sim2, 1)
        d3 = self._apply_sg_delta(d3, sim3, 2)
        d4 = self._apply_sg_delta(d4, sim4, 3)

        # Apply Mamba blocks
        d1_full = self.blocks[0](d1)
        d2_full = self.blocks[1](d2)
        d3_full = self.blocks[2](d3)
        d4_full = self.blocks[3](d4)

        if self.use_mask_scan and self.mask_scan_bypass is not None:
            d1_light = self.mask_scan_bypass[0](d1)
            d2_light = self.mask_scan_bypass[1](d2)
            d3_light = self.mask_scan_bypass[2](d3)
            d4_light = self.mask_scan_bypass[3](d4)

            d1 = self._apply_mask_scan(d1_full, d1_light, sim1)
            d2 = self._apply_mask_scan(d2_full, d2_light, sim2)
            d3 = self._apply_mask_scan(d3_full, d3_light, sim3)
            d4 = self._apply_mask_scan(d4_full, d4_light, sim4)
        else:
            d1, d2, d3, d4 = d1_full, d2_full, d3_full, d4_full

        # Apply MDEM enhancement if enabled
        if self.use_mdem:
            if s_wsi_list is not None:
                sim_mask_4, sim_mask_3, sim_mask_2, sim_mask_1 = s_wsi_list
                # MDEM takes (input, diff, s_wsi) - here we use the Mamba output as both input and diff
                # since we want to enhance the Mamba features with spatial attention
                d4 = self.mdem4(d4, d4, s_wsi=sim_mask_4)
                d3 = self.mdem3(d3, d3, s_wsi=sim_mask_3)
                d2 = self.mdem2(d2, d2, s_wsi=sim_mask_2)
                d1 = self.mdem1(d1, d1, s_wsi=sim_mask_1)
            else:
                # If no weak supervision, still apply MDEM without s_wsi
                d4 = self.mdem4(d4, d4, s_wsi=None)
                d3 = self.mdem3(d3, d3, s_wsi=None)
                d2 = self.mdem2(d2, d2, s_wsi=None)
                d1 = self.mdem1(d1, d1, s_wsi=None)

        return d4, d3, d2, d1