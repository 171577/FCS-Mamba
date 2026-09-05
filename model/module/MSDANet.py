import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
try:
    from .resnet import resnet18, resnet34, resnet50, resnet101
    from .decoder import Decoder_sim
    from .decoder_feedback import Decoder_MultiScale_WithFeedback
    from .SDA_multiscale import MambaBranch_MultiScale
    from .lightweight_align import LightweightFeatureAlign
    from .cross_scale_attention import LightweightCrossScaleAttention
    from .gsra import GSRA2DBlock
except ImportError:
    try:
        from model.module.resnet import resnet18, resnet34, resnet50, resnet101
        from model.module.decoder import Decoder_sim
        from model.module.decoder_feedback import Decoder_MultiScale_WithFeedback
        from model.module.SDA_multiscale import MambaBranch_MultiScale
        from model.module.lightweight_align import LightweightFeatureAlign
        from model.module.cross_scale_attention import LightweightCrossScaleAttention
        from model.module.gsra import GSRA2DBlock
    except ImportError:
        from resnet import resnet18, resnet34, resnet50, resnet101
        from decoder import Decoder_sim
        from decoder_feedback import Decoder_MultiScale_WithFeedback
        from SDA_multiscale import MambaBranch_MultiScale
        from lightweight_align import LightweightFeatureAlign
        from cross_scale_attention import LightweightCrossScaleAttention
        from gsra import GSRA2DBlock


class MSDANet(nn.Module):
    def __init__(
        self,
        input_nc=3,
        output_nc=2,
        in_channels_list=[64, 128, 256, 512],
        out_channels_list=[64, 128, 256, 256],
        use_depthwise=True,
        main_branch_type="mamba",
        backbone_type="vssm",
        weak_feature_source="vssm",
        resnet_pretrained=True,
        vssm_pretrained=None,
        vssm_patch_size=4,
        vssm_depths=(2, 2, 4, 2),
        vssm_dims=96,
        vssm_norm_layer="ln2d",
        vssm_forward_type="v3noz",
        vssm_ssm_d_state=16,
        vssm_ssm_ratio=2.0,
        vssm_ssm_dt_rank="auto",
        vssm_ssm_conv=3,
        vssm_ssm_conv_bias=True,
        vssm_ssm_drop_rate=0.0,
        vssm_ssm_init="v0",
        vssm_mlp_ratio=4.0,
        vssm_mlp_drop_rate=0.0,
        use_mdem=False,
        use_feature_align=True,
        use_weak_feature_align=None,
        use_weak_supervision=True,            
        use_cross_scale_attn=False,
        use_feedback_decoder=True,
        use_fem=True,
        use_im=True,
        use_weak_guidance=True,
        use_gsra=False,
        gsra_token_projection="base",
        gsra_window_size=8,
        gsra_num_heads=4,
        gsra_geo_dim=3,
        gsra_dino_dim=1024,
        use_sg_delta=False,
        use_mask_scan=False,
        use_cross_gate=False,
    ):
        super().__init__()

        self.use_feature_align = use_feature_align
        self.use_weak_feature_align = use_feature_align if use_weak_feature_align is None else use_weak_feature_align
        self.use_feedback_decoder = use_feedback_decoder
        self.use_cross_scale_attn = use_cross_scale_attn
        self.use_weak_supervision = use_weak_supervision
        self.use_mdem = use_mdem
        self.use_fem = use_fem
        self.use_im = use_im
        self.use_weak_guidance = use_weak_guidance
        self.use_gsra = use_gsra

        self.main_branch_type = main_branch_type
        self.backbone_type = backbone_type
        self.weak_feature_source = weak_feature_source

        def _resolve_heads(channels, preferred_heads):
            h = int(max(1, min(preferred_heads, channels)))
            while h > 1 and channels % h != 0:
                h -= 1
            return h

        def _compute_vssm_dims(dims_value, n_layers=4):
            if isinstance(dims_value, int):
                return [int(dims_value * (2 ** i)) for i in range(n_layers)]
            return list(dims_value)

        self.vssm_dims_list = _compute_vssm_dims(vssm_dims, n_layers=4)

        resnet_channels_map = {
            "resnet18": [64, 128, 256, 512],
            "resnet34": [64, 128, 256, 512],
            "resnet50": [256, 512, 1024, 2048],
            "resnet101": [256, 512, 1024, 2048],
        }

        if in_channels_list == [64, 128, 256, 512]:
            if self.backbone_type == "vssm":
                in_channels_list = self.vssm_dims_list
            elif self.backbone_type in resnet_channels_map:
                in_channels_list = resnet_channels_map[self.backbone_type]
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list

        def _try_import_backbone_vssm():
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
            mamba_models_dir = os.path.join(repo_root, "ChangeMamba-master", "classification", "models")
            if os.path.isdir(mamba_models_dir) and (mamba_models_dir not in sys.path):
                sys.path.append(mamba_models_dir)
            try:
                from vmamba import Backbone_VSSM, LayerNorm2d
                return Backbone_VSSM, LayerNorm2d
            except Exception:
                return None, None

                  
        self.res = None
        self.vssm = None

        resnet_builders = {
            "resnet18": resnet18,
            "resnet34": resnet34,
            "resnet50": resnet50,
            "resnet101": resnet101,
        }

        backbone_is_resnet = self.backbone_type in resnet_builders
        weak_is_resnet = self.weak_feature_source in resnet_builders
        needs_resnet = backbone_is_resnet or weak_is_resnet
        needs_vssm = (self.backbone_type == "vssm") or (self.weak_feature_source == "vssm")

        if needs_resnet:
            if backbone_is_resnet and weak_is_resnet and (self.backbone_type != self.weak_feature_source):
                raise ValueError(
                    f"backbone_type ({self.backbone_type}) and weak_feature_source ({self.weak_feature_source}) "
                    "are different ResNet variants. This implementation expects them to be the same."
                )

        if (self.backbone_type != "vssm") and (not backbone_is_resnet):
            raise ValueError(
                f"Unsupported backbone_type={self.backbone_type!r}. Supported: 'vssm', "
                + ", ".join([repr(k) for k in resnet_builders.keys()])
                + ". If you meant ResNet34, use 'resnet34' (not 'renet34')."
            )

        if needs_resnet:
            resnet_name = self.backbone_type if backbone_is_resnet else self.weak_feature_source
            self.res = resnet_builders[resnet_name](pretrained=resnet_pretrained)

        if needs_vssm:
            Backbone_VSSM, LayerNorm2d = _try_import_backbone_vssm()
            if Backbone_VSSM is None:
                raise ImportError(
                    "Failed to import Backbone_VSSM from vmamba. Ensure ChangeMamba-master exists and is importable."
                )
            self.vssm = Backbone_VSSM(
                out_indices=(0, 1, 2, 3),
                pretrained=vssm_pretrained,
                norm_layer=vssm_norm_layer,
                patch_size=vssm_patch_size,
                in_chans=input_nc,
                depths=list(vssm_depths),
                dims=self.vssm_dims_list,
                ssm_d_state=vssm_ssm_d_state,
                ssm_ratio=vssm_ssm_ratio,
                ssm_dt_rank=vssm_ssm_dt_rank,
                ssm_act_layer="silu",
                ssm_conv=vssm_ssm_conv,
                ssm_conv_bias=vssm_ssm_conv_bias,
                ssm_drop_rate=vssm_ssm_drop_rate,
                ssm_init=vssm_ssm_init,
                forward_type=vssm_forward_type,
                mlp_ratio=vssm_mlp_ratio,
                mlp_act_layer="gelu",
                mlp_drop_rate=vssm_mlp_drop_rate,
                drop_path_rate=0.1,
                patch_norm=True,
                downsample_version="v3",
                patchembed_version="v2",
                use_checkpoint=False,
            )

        self.dropout = nn.Dropout2d(p=0.3)
        self.gsra_blocks = None

        # ========== ResNet101 通道适配层 ==========
        # 将 ResNet50/101/152 的高通道数适配到标准配置
        self.channel_adapters = None
        self.weak_channel_adapters = None
        
        if self.backbone_type in ["resnet50", "resnet101", "resnet152"]:
            # 检查是否需要通道适配
            # 标准配置期望的通道数（用于后续模块）
            expected_channels = [64, 128, 256, 512]
            actual_channels = self.in_channels_list
            
            # 如果实际通道数与期望不符，添加适配层
            if actual_channels != expected_channels:
                print(f"[MSDANet] 检测到 {self.backbone_type}，添加通道适配层")
                print(f"[MSDANet]   输入通道: {actual_channels}")
                print(f"[MSDANet]   目标通道: {expected_channels}")
                
                self.channel_adapters = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv2d(actual_channels[i], expected_channels[i], 1, bias=False),
                        nn.BatchNorm2d(expected_channels[i]),
                        nn.ReLU(inplace=True)
                    )
                    for i in range(4)
                ])
                
                # 更新 in_channels_list 为适配后的通道
                self.in_channels_list = expected_channels
                print(f"[MSDANet]   适配后通道: {self.in_channels_list}")

        if self.use_gsra:
            if len(self.in_channels_list) != 4:
                raise ValueError(f"GSRA expects 4 feature scales, got in_channels_list={self.in_channels_list}")
            self.gsra_blocks = nn.ModuleList([
                GSRA2DBlock(
                    dim=self.in_channels_list[i],
                    num_heads=_resolve_heads(self.in_channels_list[i], gsra_num_heads),
                    window_size=gsra_window_size,
                    token_projection=gsra_token_projection,
                    depth=i + 1,
                    geo_dim=gsra_geo_dim,
                    dino_dim=gsra_dino_dim,
                )
                for i in range(4)
            ])

        if self.use_feature_align:
            align_class = LightweightFeatureAlign
            self.align_scale1 = align_class(self.in_channels_list[0])
            self.align_scale2 = align_class(self.in_channels_list[1])
            self.align_scale3 = align_class(self.in_channels_list[2])
            self.align_scale4 = align_class(self.in_channels_list[3])

        if self.use_weak_supervision:
            weak_in_d = self.in_channels_list
            if self.weak_feature_source == "vssm":
                weak_in_d = self.vssm_dims_list
            elif self.weak_feature_source in resnet_channels_map:
                weak_in_d = resnet_channels_map[self.weak_feature_source]
            self.weak_in_channels_list = weak_in_d
            
            # 如果 weak_feature_source 也是 ResNet50/101/152，需要适配器
            if self.weak_feature_source in ["resnet50", "resnet101", "resnet152"]:
                expected_channels = [64, 128, 256, 512]
                if weak_in_d != expected_channels:
                    print(f"[MSDANet] 为 weak_feature_source ({self.weak_feature_source}) 添加通道适配层")
                    print(f"[MSDANet]   输入通道: {weak_in_d}")
                    print(f"[MSDANet]   目标通道: {expected_channels}")
                    
                    self.weak_channel_adapters = nn.ModuleList([
                        nn.Sequential(
                            nn.Conv2d(weak_in_d[i], expected_channels[i], 1, bias=False),
                            nn.BatchNorm2d(expected_channels[i]),
                            nn.ReLU(inplace=True)
                        )
                        for i in range(4)
                    ])
                    
                    # 更新 weak_in_channels_list
                    self.weak_in_channels_list = expected_channels
                    weak_in_d = expected_channels
                    print(f"[MSDANet]   适配后通道: {self.weak_in_channels_list}")

            need_weak_align = self.use_weak_feature_align and (
                (self.weak_feature_source != self.backbone_type) or (not self.use_feature_align)
            )
            if need_weak_align:
                weak_align_class = LightweightFeatureAlign
                self.align_weak_scale1 = weak_align_class(self.weak_in_channels_list[0])
                self.align_weak_scale2 = weak_align_class(self.weak_in_channels_list[1])
                self.align_weak_scale3 = weak_align_class(self.weak_in_channels_list[2])
                self.align_weak_scale4 = weak_align_class(self.weak_in_channels_list[3])
            else:
                self.align_weak_scale1 = None
                self.align_weak_scale2 = None
                self.align_weak_scale3 = None
                self.align_weak_scale4 = None
            self.decoder_sim = Decoder_sim(
                in_d=weak_in_d,
                out_d=64,
                use_depthwise=use_depthwise,
                use_fem=self.use_fem,
                use_im=self.use_im
            )
        else:
            self.decoder_sim = None

        if self.main_branch_type != "mamba":
            raise ValueError(f"Unsupported main_branch_type={self.main_branch_type!r}. Only 'mamba' is supported.")

        self.mamba_branch = MambaBranch_MultiScale(
                in_channels_list=self.in_channels_list,
                out_channels_list=self.out_channels_list,
                n_blocks_per_scale=1,
                use_weak_guidance=self.use_weak_guidance,
                use_mdem=use_mdem,
            use_sg_delta=use_sg_delta,
            use_mask_scan=use_mask_scan,
            use_cross_gate=use_cross_gate,
                mamba_forward_type=vssm_forward_type,
                mamba_ssm_d_state=vssm_ssm_d_state,
                mamba_ssm_ratio=vssm_ssm_ratio,
                mamba_ssm_dt_rank=vssm_ssm_dt_rank,
                mamba_ssm_conv=vssm_ssm_conv,
                mamba_ssm_conv_bias=vssm_ssm_conv_bias,
                mamba_ssm_drop_rate=vssm_ssm_drop_rate,
                mamba_ssm_init=vssm_ssm_init,
                mamba_mlp_ratio=vssm_mlp_ratio,
                mamba_mlp_drop_rate=vssm_mlp_drop_rate,
                drop_path=0.0,
            )

        if use_cross_scale_attn:
            self.cross_scale_bridge = LightweightCrossScaleAttention(out_channels_list)

        if use_feedback_decoder:
            self.decoder = Decoder_MultiScale_WithFeedback(
                in_d_list=self.out_channels_list,
                out_d=output_nc,
                use_depthwise=use_depthwise,
                use_feedback=True,
            )
        else:
                     
            try:
                from .decoder import Decoder_MultiScale
            except ImportError:
                from decoder import Decoder_MultiScale
            self.decoder = Decoder_MultiScale(
                out_channels_list,
                output_nc,
                use_depthwise=use_depthwise,
            )
        self.sim_fusion_weights = nn.Parameter(torch.ones(4, dtype=torch.float32))

    def _sanitize_prob_map(self, x: torch.Tensor, name: str = "prob") -> torch.Tensor:
        if x is None:
            return x
        x = torch.nan_to_num(x, nan=0.5, posinf=1.0, neginf=0.0)
        x = x.clamp(min=0.0, max=1.0)
        return x

    def _sanitize_sim_list(self, sims):
        if sims is None:
            return sims
        out = []
        for i, s in enumerate(sims):
            if torch.is_tensor(s):
                out.append(self._sanitize_prob_map(s, name=f"sim{i+1}"))
            else:
                out.append(s)
        return out

    def forward(self, t1, t2, gt_mask=None, profile_model=False, **kwargs):
        B, C, H, W = t1.shape
        aux_losses = {}

        if self.backbone_type == "vssm":
            f_t1 = self.vssm(t1)
            f_t2 = self.vssm(t2)
            xr1_1, xr1_2, xr1_3, xr1_4 = f_t1
            xr2_1, xr2_2, xr2_3, xr2_4 = f_t2
        else:
            f_t1 = self.res.base_forward(t1)
            f_t2 = self.res.base_forward(t2)
            f_t1 = list(f_t1[1:])
            f_t2 = list(f_t2[1:])
            xr1_1, xr1_2, xr1_3, xr1_4 = f_t1
            xr2_1, xr2_2, xr2_3, xr2_4 = f_t2

        # 应用通道适配器（如果需要）
        if self.channel_adapters is not None:
            xr1_1 = self.channel_adapters[0](xr1_1)
            xr1_2 = self.channel_adapters[1](xr1_2)
            xr1_3 = self.channel_adapters[2](xr1_3)
            xr1_4 = self.channel_adapters[3](xr1_4)
            xr2_1 = self.channel_adapters[0](xr2_1)
            xr2_2 = self.channel_adapters[1](xr2_2)
            xr2_3 = self.channel_adapters[2](xr2_3)
            xr2_4 = self.channel_adapters[3](xr2_4)

        if self.use_feature_align:
            xr1_1, xr2_1 = self.align_scale1(xr1_1, xr2_1)
            xr1_2, xr2_2 = self.align_scale2(xr1_2, xr2_2)
            xr1_3, xr2_3 = self.align_scale3(xr1_3, xr2_3)
            xr1_4, xr2_4 = self.align_scale4(xr1_4, xr2_4)

        if self.use_gsra and self.gsra_blocks is not None:
            xr1_1 = self.gsra_blocks[0](xr1_1)
            xr1_2 = self.gsra_blocks[1](xr1_2)
            xr1_3 = self.gsra_blocks[2](xr1_3)
            xr1_4 = self.gsra_blocks[3](xr1_4)
            xr2_1 = self.gsra_blocks[0](xr2_1)
            xr2_2 = self.gsra_blocks[1](xr2_2)
            xr2_3 = self.gsra_blocks[2](xr2_3)
            xr2_4 = self.gsra_blocks[3](xr2_4)

        bxr1_1, bxr1_2, bxr1_3, bxr1_4 = xr1_1, xr1_2, xr1_3, xr1_4
        bxr2_1, bxr2_2, bxr2_3, bxr2_4 = xr2_1, xr2_2, xr2_3, xr2_4

        if self.use_weak_supervision and self.decoder_sim is not None:
            if self.weak_feature_source == self.backbone_type:
                wx1_1, wx1_2, wx1_3, wx1_4 = bxr1_1, bxr1_2, bxr1_3, bxr1_4
                wx2_1, wx2_2, wx2_3, wx2_4 = bxr2_1, bxr2_2, bxr2_3, bxr2_4
                if self.use_weak_feature_align and self.align_weak_scale1 is not None:
                    wx1_1, wx2_1 = self.align_weak_scale1(wx1_1, wx2_1)
                    wx1_2, wx2_2 = self.align_weak_scale2(wx1_2, wx2_2)
                    wx1_3, wx2_3 = self.align_weak_scale3(wx1_3, wx2_3)
                    wx1_4, wx2_4 = self.align_weak_scale4(wx1_4, wx2_4)
            elif self.weak_feature_source == "vssm":
                wx1_1, wx1_2, wx1_3, wx1_4 = self.vssm(t1)
                wx2_1, wx2_2, wx2_3, wx2_4 = self.vssm(t2)
                if self.use_weak_feature_align and self.align_weak_scale1 is not None:
                    wx1_1, wx2_1 = self.align_weak_scale1(wx1_1, wx2_1)
                    wx1_2, wx2_2 = self.align_weak_scale2(wx1_2, wx2_2)
                    wx1_3, wx2_3 = self.align_weak_scale3(wx1_3, wx2_3)
                    wx1_4, wx2_4 = self.align_weak_scale4(wx1_4, wx2_4)
            else:
                f_wx1 = self.res.base_forward(t1)
                f_wx2 = self.res.base_forward(t2)
                f_wx1 = list(f_wx1[1:])
                f_wx2 = list(f_wx2[1:])
                wx1_1, wx1_2, wx1_3, wx1_4 = f_wx1
                wx2_1, wx2_2, wx2_3, wx2_4 = f_wx2
                
                # 应用 weak 通道适配器（如果需要）
                if self.weak_channel_adapters is not None:
                    wx1_1 = self.weak_channel_adapters[0](wx1_1)
                    wx1_2 = self.weak_channel_adapters[1](wx1_2)
                    wx1_3 = self.weak_channel_adapters[2](wx1_3)
                    wx1_4 = self.weak_channel_adapters[3](wx1_4)
                    wx2_1 = self.weak_channel_adapters[0](wx2_1)
                    wx2_2 = self.weak_channel_adapters[1](wx2_2)
                    wx2_3 = self.weak_channel_adapters[2](wx2_3)
                    wx2_4 = self.weak_channel_adapters[3](wx2_4)
                
                if self.use_weak_feature_align and self.align_weak_scale1 is not None:
                    wx1_1, wx2_1 = self.align_weak_scale1(wx1_1, wx2_1)
                    wx1_2, wx2_2 = self.align_weak_scale2(wx1_2, wx2_2)
                    wx1_3, wx2_3 = self.align_weak_scale3(wx1_3, wx2_3)
                    wx1_4, wx2_4 = self.align_weak_scale4(wx1_4, wx2_4)

            sim4, sim3, sim2, sim1 = self.decoder_sim(
                wx1_1, wx1_2, wx1_3, wx1_4,
                wx2_1, wx2_2, wx2_3, wx2_4,
            )
            sim4, sim3, sim2, sim1 = self._sanitize_sim_list([sim4, sim3, sim2, sim1])
        else:
            sim1 = torch.zeros((B, 1, xr1_1.shape[2], xr1_1.shape[3]), device=xr1_1.device, dtype=xr1_1.dtype)
            sim2 = torch.zeros((B, 1, xr1_2.shape[2], xr1_2.shape[3]), device=xr1_2.device, dtype=xr1_2.dtype)
            sim3 = torch.zeros((B, 1, xr1_3.shape[2], xr1_3.shape[3]), device=xr1_3.device, dtype=xr1_3.dtype)
            sim4 = torch.zeros((B, 1, xr1_4.shape[2], xr1_4.shape[3]), device=xr1_4.device, dtype=xr1_4.dtype)

        dr4, dr3, dr2, dr1 = self.mamba_branch(
            xr1_1, xr1_2, xr1_3, xr1_4,
            xr2_1, xr2_2, xr2_3, xr2_4,
            s_wsi_list=[sim4, sim3, sim2, sim1] if (self.use_weak_supervision and self.use_weak_guidance) else None
        )

        if self.use_cross_scale_attn:
            dr1, dr2, dr3, dr4 = self.cross_scale_bridge(dr1, dr2, dr3, dr4)

        if self.use_feedback_decoder:
            decoder_output = self.decoder(
                dr4, dr3, dr2, dr1,
                sim4, sim3, sim2, sim1,
            )

            if self.training and isinstance(decoder_output, tuple):
                                             
                if len(decoder_output) == 4:
                    mask, aux_mask_d2, aux_mask_d3, aux_mask_d4 = decoder_output
                else:
                    mask, aux_mask_d3, aux_mask_d4 = decoder_output
                    aux_mask_d2 = None
                
                          
                mask = F.interpolate(mask, size=t1.shape[2:], mode='bilinear', align_corners=True)
                
                if aux_mask_d2 is not None:
                    aux_mask_d2 = F.interpolate(aux_mask_d2, size=t1.shape[2:], mode='bilinear', align_corners=True)
                    aux_losses['aux_d2'] = aux_mask_d2
                
                aux_mask_d3 = F.interpolate(aux_mask_d3, size=t1.shape[2:], mode='bilinear', align_corners=True)
                aux_mask_d4 = F.interpolate(aux_mask_d4, size=t1.shape[2:], mode='bilinear', align_corners=True)
                
                aux_losses['aux_d3'] = aux_mask_d3
                aux_losses['aux_d4'] = aux_mask_d4
            else:
                if isinstance(decoder_output, tuple):
                    mask = decoder_output[0]
                else:
                    mask = decoder_output
                mask = F.interpolate(mask, size=t1.shape[2:], mode='bilinear', align_corners=True)
        else:
            decoder_output = self.decoder(
                dr4, dr3, dr2, dr1,
                sim4, sim3, sim2, sim1,
            )

            if self.training and isinstance(decoder_output, tuple):
                if len(decoder_output) == 3:
                    mask, aux_mask_d3, aux_mask_d4 = decoder_output
                else:
                    mask = decoder_output[0]
                    aux_mask_d3 = None
                    aux_mask_d4 = None

                mask = F.interpolate(mask, size=t1.shape[2:], mode='bilinear', align_corners=True)
                if torch.is_tensor(aux_mask_d3):
                    aux_mask_d3 = F.interpolate(aux_mask_d3, size=t1.shape[2:], mode='bilinear', align_corners=True)
                    aux_losses['aux_d3'] = aux_mask_d3
                if torch.is_tensor(aux_mask_d4):
                    aux_mask_d4 = F.interpolate(aux_mask_d4, size=t1.shape[2:], mode='bilinear', align_corners=True)
                    aux_losses['aux_d4'] = aux_mask_d4
            else:
                mask = decoder_output[0] if isinstance(decoder_output, tuple) else decoder_output
                mask = F.interpolate(mask, size=t1.shape[2:], mode='bilinear', align_corners=True)

        sim_fused = self._fuse_multi_scale_sim(sim4, sim3, sim2, sim1) if self.use_weak_supervision else sim1
        sim_fused = self._sanitize_prob_map(sim_fused, name="sim_fused")

                        
        if self.training:
            return mask, aux_losses, sim_fused
        else:
            return mask, sim_fused

    def _fuse_multi_scale_sim(self, sim4, sim3, sim2, sim1):
                       
        target_size = sim1.shape[2:]
        s4 = F.interpolate(sim4, target_size, mode='bilinear', align_corners=True)
        s3 = F.interpolate(sim3, target_size, mode='bilinear', align_corners=True)
        s2 = F.interpolate(sim2, target_size, mode='bilinear', align_corners=True)

                         
        weights = F.softmax(self.sim_fusion_weights, dim=0)
        fused_sim = s4 * weights[0] + s3 * weights[1] + s2 * weights[2] + sim1 * weights[3]
        return fused_sim

    @classmethod
    def from_opt(cls, opt, **overrides):
        def _get(name, default):
            return getattr(opt, name, default)

        def _parse_int_tuple(val):
            if isinstance(val, (tuple, list)):
                return tuple(int(x) for x in val)
            if isinstance(val, str):
                s = val.strip()
                if s == "":
                    return None
                return tuple(int(x.strip()) for x in s.split(',') if x.strip() != '')
            return val

        def _parse_int_or_list(val):
            if isinstance(val, (tuple, list)):
                return [int(x) for x in val]
            if isinstance(val, str):
                s = val.strip()
                if s == "":
                    return None
                if ',' in s:
                    return [int(x.strip()) for x in s.split(',') if x.strip() != '']
                return int(s)
            return val

        def _parse_stage_tuple(val):
            if val is None:
                return None
            if isinstance(val, (tuple, list)):
                return tuple(int(x) for x in val)
            if isinstance(val, str):
                s = val.strip().lower()
                if s in ('', 'auto', 'all'):
                    return None
                if s == 'none':
                    return tuple()
                return tuple(int(x.strip()) for x in s.split(',') if x.strip() != '')
            return val

        def _parse_auto_or_int(val, default="auto"):
            if val is None:
                return default
            if isinstance(val, int):
                return val
            if isinstance(val, str):
                s = val.strip().lower()
                if s == "":
                    return default
                if s == "auto":
                    return "auto"
                return int(s)
            return val

        backbone_type = _get('backbone_type', 'resnet18')
        weak_feature_source = _get('weak_feature_source', backbone_type)

        resnet_variants = {"resnet18", "resnet34", "resnet50", "resnet101"}
        if (backbone_type in resnet_variants) and (weak_feature_source in resnet_variants) and (backbone_type != weak_feature_source):
            weak_feature_source = backbone_type

        kwargs = dict(
            input_nc=_get('input_nc', 3),
            output_nc=_get('output_nc', 2),
            use_depthwise=_get('use_depthwise', True),
            main_branch_type=_get('main_branch_type', 'mamba'),
            backbone_type=backbone_type,
            weak_feature_source=weak_feature_source,
            resnet_pretrained=_get('resnet_pretrained', True),
            vssm_pretrained=_get('vssm_pretrained', None),
            vssm_patch_size=_get('vssm_patch_size', 4),
            vssm_depths=_parse_int_tuple(_get('vssm_depths', (2, 2, 4, 2))) or (2, 2, 4, 2),
            vssm_dims=_parse_int_or_list(_get('vssm_dims', 96)) or 96,
            vssm_norm_layer=_get('vssm_norm_layer', 'ln2d'),
            vssm_forward_type=_get('vssm_forward_type', 'v3noz'),
            vssm_ssm_d_state=_get('vssm_ssm_d_state', 16),
            vssm_ssm_ratio=_get('vssm_ssm_ratio', 2.0),
            vssm_ssm_dt_rank=_get('vssm_ssm_dt_rank', 'auto'),
            vssm_ssm_conv=_get('vssm_ssm_conv', 3),
            vssm_ssm_conv_bias=_get('vssm_ssm_conv_bias', True),
            vssm_ssm_drop_rate=_get('vssm_ssm_drop_rate', 0.0),
            vssm_ssm_init=_get('vssm_ssm_init', 'v0'),
            vssm_mlp_ratio=_get('vssm_mlp_ratio', 4.0),
            vssm_mlp_drop_rate=_get('vssm_mlp_drop_rate', 0.0),
            use_mdem=_get('use_mdem', True),
            use_feature_align=_get('use_feature_align', True),
            use_weak_feature_align=_get('use_weak_feature_align', None),
            use_weak_supervision=_get('use_weak_supervision', True),
            use_fem=_get('use_fem', True),
            use_im=_get('use_im', True),
            use_weak_guidance=_get('use_weak_guidance', True),
            use_sg_delta=_get('use_sg_delta', False),
            use_mask_scan=_get('use_mask_scan', False),
            use_cross_gate=_get('use_cross_gate', False),
            use_gsra=_get('use_gsra', False),
            gsra_token_projection=_get('gsra_token_projection', 'base'),
            gsra_window_size=_get('gsra_window_size', 8),
            gsra_num_heads=_get('gsra_num_heads', 4),
            gsra_geo_dim=_get('gsra_geo_dim', 3),
            gsra_dino_dim=_get('gsra_dino_dim', 1024),
        )

        kwargs.update(overrides)
        return cls(**kwargs)

    def get_param_count(self):
                     
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total': total,
            'trainable': trainable,
            'total_M': total / 1e6,
            'trainable_M': trainable / 1e6
        }

    @staticmethod
    def remap_legacy_state_dict_keys(state_dict):
        if not isinstance(state_dict, dict):
            return state_dict

        remapped = {}
        for k, v in state_dict.items():
            if k.startswith("sda_branch."):
                remapped["mamba_branch." + k[len("sda_branch."):]] = v
            else:
                remapped[k] = v
        return remapped