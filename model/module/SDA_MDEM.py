import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence

try:
    from .depthwise_separable import DepthwiseSeparableConvWithReLU
    from .sim_agg import Poly_Kernel_Inception_Block
except ImportError:
    try:
        from model.module.depthwise_separable import DepthwiseSeparableConvWithReLU
        from model.module.sim_agg import Poly_Kernel_Inception_Block
    except ImportError:
        from depthwise_separable import DepthwiseSeparableConvWithReLU
        from sim_agg import Poly_Kernel_Inception_Block


class ConvModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1,
                 padding: int = 0, dilation: int = 1, groups: int = 1,
                 norm_cfg: Optional[dict] = None, act_cfg: Optional[dict] = None):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                                dilation=dilation, groups=groups, bias=(norm_cfg is None)))
        if norm_cfg:
            layers.append(
                nn.BatchNorm2d(out_channels, momentum=norm_cfg.get('momentum', 0.1), eps=norm_cfg.get('eps', 1e-5)))
        if act_cfg:
            if act_cfg['type'] == 'ReLU':
                layers.append(nn.ReLU(inplace=True))
            elif act_cfg['type'] == 'SiLU':
                layers.append(nn.SiLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))

        out = identity * a_h * a_w
        return out


class MSFF(nn.Module):
                                                                 

    def __init__(self, inchannel, mid_channel=None, use_depthwise=True):
        super(MSFF, self).__init__()

                                                      
                                            
        self.pkib = Poly_Kernel_Inception_Block(
            in_channels=inchannel,
            out_channels=inchannel,
            kernel_sizes=[3, 5, 7, 9],              
            expansion=1.0
        )

                                    
                                      
        self.se_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(inchannel, inchannel // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(inchannel // 4, inchannel, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
                   
        x_feat = self.pkib(x)

                    
        att = self.se_attn(x_feat)
        out = x_feat * att

                          
                                     
        return out


class SDA(nn.Module):
                                                           

    def __init__(self, in_d, out_d, dw=False, use_depthwise=True):
        super().__init__()
        self.in_d = in_d
        self.out_d = out_d

                   
        self.MPFL = MSFF(inchannel=in_d, mid_channel=64, use_depthwise=use_depthwise)

                
        if use_depthwise:
            self.conv_diff_enh = DepthwiseSeparableConvWithReLU(in_d, in_d, kernel_size=3, padding=1)
            self.conv_dr = DepthwiseSeparableConvWithReLU(in_d, out_d, kernel_size=3, padding=1)
            self.conv_sub = DepthwiseSeparableConvWithReLU(in_d, in_d, kernel_size=3, padding=1)
        else:
            self.conv_diff_enh = nn.Sequential(nn.Conv2d(in_d, in_d, 3, 1, 1, bias=False), nn.BatchNorm2d(in_d),
                                               nn.ReLU(inplace=True))
            self.conv_dr = nn.Sequential(nn.Conv2d(in_d, out_d, 3, 1, 1, bias=False), nn.BatchNorm2d(out_d),
                                         nn.ReLU(inplace=True))
            self.conv_sub = nn.Sequential(nn.Conv2d(in_d, in_d, 3, 1, 1, bias=False), nn.BatchNorm2d(in_d),
                                          nn.ReLU(inplace=True))

        if use_depthwise:
            self.convmix = DepthwiseSeparableConvWithReLU(2 * in_d, in_d, 3, padding=1)
        else:
            self.convmix = nn.Sequential(
                nn.Conv2d(2 * in_d, in_d, 3, padding=1, bias=False),
                nn.BatchNorm2d(in_d),
                nn.ReLU(inplace=True)
            )

                                     
        self.coord_att = CoordAtt(in_d, in_d)

                                         
                                                     
        self.diff_fusion = nn.Conv2d(in_d * 3, in_d, 1, bias=False)

    def forward(self, x1, x2):
        b, c, h, w = x1.shape
        x_diff = x1 - x2
        x_sub = torch.abs(x_diff)

                
        x_att = torch.sigmoid(self.conv_sub(x_sub))

              
        x1_enh = (x1 * x_att) + self.MPFL(self.conv_diff_enh(x1))
        x2_enh = (x2 * x_att) + self.MPFL(self.conv_diff_enh(x2))

                    
        x_f = torch.stack((x1_enh, x2_enh), dim=2)
        x_f = torch.reshape(x_f, (b, -1, h, w))
        x_f = self.convmix(x_f)

                       
        x_f = self.coord_att(x_f)

                                   
                                                           
        fusion_in = torch.cat([x_f * x_att, x_diff, x_sub], dim=1)
        x_final = self.diff_fusion(fusion_in)

        out = self.conv_dr(x_final)
        return out


                                                                    
                                           
                           
class Refine(nn.Module):
    def __init__(self, inchannel, outchannel, use_depthwise=True):
        super(Refine, self).__init__()
        if use_depthwise:
            self.conv1 = DepthwiseSeparableConvWithReLU(inchannel, inchannel, kernel_size=3, padding=1)
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(inchannel, inchannel, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(inchannel),
                nn.ReLU(inplace=True)
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inchannel + outchannel, outchannel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, x2.size()[2:], mode='bilinear', align_corners=True)
        x1 = self.conv1(x1)
        x_f = torch.cat([x1, x2], dim=1)
        x_f = self.conv2(x_f)
        return x_f


class CIEM(nn.Module):
    def __init__(self, base_dim=64, out_d=[64, 128, 256, 256], use_depthwise=True):
        super(CIEM, self).__init__()
        self.base_dim = base_dim
        self.refine1 = Refine(out_d[3], out_d[2], use_depthwise=use_depthwise)
        self.refine2 = Refine(out_d[2], out_d[1], use_depthwise=use_depthwise)
        self.refine3 = Refine(out_d[1], out_d[0], use_depthwise=use_depthwise)
        if use_depthwise:
            self.conv_dr = DepthwiseSeparableConvWithReLU(out_d[0], base_dim, kernel_size=3, padding=1)
        else:
            self.conv_dr = nn.Sequential(
                nn.Conv2d(out_d[0], base_dim, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(base_dim),
                nn.ReLU(inplace=True)
            )
        self.pools_sizes = [2, 4, 8]
        if use_depthwise:
            self.conv_pool1 = nn.Sequential(nn.AvgPool2d(self.pools_sizes[0]),
                                            DepthwiseSeparableConvWithReLU(base_dim, out_d[1], 3, padding=1))
        else:
            self.conv_pool1 = nn.Sequential(nn.AvgPool2d(self.pools_sizes[0]),
                                            nn.Conv2d(base_dim, out_d[1], 3, 1, 1, bias=False))
        if use_depthwise:
            self.conv_pool2 = nn.Sequential(nn.AvgPool2d(self.pools_sizes[1]),
                                            DepthwiseSeparableConvWithReLU(base_dim, out_d[2], 3, padding=1))
        else:
            self.conv_pool2 = nn.Sequential(nn.AvgPool2d(self.pools_sizes[1]),
                                            nn.Conv2d(base_dim, out_d[2], 3, 1, 1, bias=False))
        if use_depthwise:
            self.conv_pool3 = nn.Sequential(nn.AvgPool2d(self.pools_sizes[2]),
                                            DepthwiseSeparableConvWithReLU(base_dim, out_d[3], 3, padding=1))
        else:
            self.conv_pool3 = nn.Sequential(nn.AvgPool2d(self.pools_sizes[2]),
                                            nn.Conv2d(base_dim, out_d[3], 3, 1, 1, bias=False))

    def forward(self, d5, d4, d3, d2):
        r1 = self.refine1(d5, d4)
        r2 = self.refine2(r1, d3)
        x = self.refine3(r2, d2)
        x = self.conv_dr(x)
        return self.conv_pool3(x), self.conv_pool2(x), self.conv_pool1(x), x


class GRM(nn.Module):
    def __init__(self, out_d=[64, 128, 256, 256], use_depthwise=True):
        super(GRM, self).__init__()
        if use_depthwise:
            self.conv_d5 = DepthwiseSeparableConvWithReLU(out_d[3] * 2, out_d[3], 3, padding=1)
            self.conv_d4 = DepthwiseSeparableConvWithReLU(out_d[2] * 2, out_d[2], 3, padding=1)
            self.conv_d3 = DepthwiseSeparableConvWithReLU(out_d[1] * 2, out_d[1], 3, padding=1)
            self.conv_d2 = DepthwiseSeparableConvWithReLU(out_d[0] * 2, out_d[0], 3, padding=1)
        else:
                           
            self.conv_d5 = nn.Sequential(nn.Conv2d(out_d[3] * 2, out_d[3], 3, 1, 1), nn.BatchNorm2d(out_d[3]),
                                         nn.ReLU(True))
            self.conv_d4 = nn.Sequential(nn.Conv2d(out_d[2] * 2, out_d[2], 3, 1, 1), nn.BatchNorm2d(out_d[2]),
                                         nn.ReLU(True))
            self.conv_d3 = nn.Sequential(nn.Conv2d(out_d[1] * 2, out_d[1], 3, 1, 1), nn.BatchNorm2d(out_d[1]),
                                         nn.ReLU(True))
            self.conv_d2 = nn.Sequential(nn.Conv2d(out_d[0] * 2, out_d[0], 3, 1, 1), nn.BatchNorm2d(out_d[0]),
                                         nn.ReLU(True))

    def stack(self, x1, x2):
        b, c, h, w = x1.size()
        x_f = torch.stack((x1, x2), dim=2)
        return torch.reshape(x_f, (b, -1, h, w))

    def forward(self, d5, d4, d3, d2, d5_p, d4_p, d3_p, d2_p):
        return self.conv_d5(self.stack(d5_p, d5)), self.conv_d4(self.stack(d4_p, d4)),\
            self.conv_d3(self.stack(d3_p, d3)), self.conv_d2(self.stack(d2_p, d2))


class DAMBlock(nn.Module):
    def __init__(self, base_dim=64, out_d=[64, 128, 256, 256], use_depthwise=True):
        super(DAMBlock, self).__init__()
        self.ciem = CIEM(base_dim, out_d, use_depthwise)
        self.grm = GRM(out_d, use_depthwise)

    def forward(self, d5, d4, d3, d2):
        d5_p, d4_p, d3_p, d2_p = self.ciem(d5, d4, d3, d2)
        return self.grm(d5, d4, d3, d2, d5_p, d4_p, d3_p, d2_p)


                                                                     
class DAMBlockV2(nn.Module):
    def __init__(self, base_dim=64, out_d=[64, 128, 256, 256], use_depthwise=True, use_skip=True):
        super(DAMBlockV2, self).__init__()
        self.use_skip = use_skip

        self.ciem = CIEM(base_dim, out_d, use_depthwise)
        self.grm = GRM(out_d, use_depthwise)

                     
        if use_skip:
            self.skip_weights = nn.ParameterList([
                nn.Parameter(torch.tensor(0.1)),             
                nn.Parameter(torch.tensor(0.2)),             
                nn.Parameter(torch.tensor(0.3)),             
                nn.Parameter(torch.tensor(0.4))             
            ])

                         
        try:
            from cross_scale_attention import GatedResidualFusion
            self.gated_fusion = nn.ModuleList([
                GatedResidualFusion(out_d[i], n_features=2) for i in range(4)
            ])
            self.use_gated = True
        except ImportError:
            self.use_gated = False

    def forward(self, d5, d4, d3, d2):
                    
        d5_input, d4_input, d3_input, d2_input = d5, d4, d3, d2

                
        d5_p, d4_p, d3_p, d2_p = self.ciem(d5, d4, d3, d2)

               
        d5_out, d4_out, d3_out, d2_out = self.grm(d5, d4, d3, d2, d5_p, d4_p, d3_p, d2_p)

                     
        if self.use_gated:
            d5_out = self.gated_fusion[3](d5_out, d5_input)
            d4_out = self.gated_fusion[2](d4_out, d4_input)
            d3_out = self.gated_fusion[1](d3_out, d3_input)
            d2_out = self.gated_fusion[0](d2_out, d2_input)
        elif self.use_skip:
                       
            d5_out = d5_out + d5_input * self.skip_weights[3]
            d4_out = d4_out + d4_input * self.skip_weights[2]
            d3_out = d3_out + d3_input * self.skip_weights[1]
            d2_out = d2_out + d2_input * self.skip_weights[0]

        return d5_out, d4_out, d3_out, d2_out


class MDEM(nn.Module):
                                                                         

    def __init__(self, input_dim, diff_dim, ds=8, beta_init=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.diff_dim = diff_dim
        self.key_channel = self.diff_dim // 8

        self.ds = ds

        self.pool = nn.AvgPool2d(self.ds)
        self.query_conv = nn.Conv2d(diff_dim, diff_dim // 8, 1)
        self.key_conv = nn.Conv2d(diff_dim, diff_dim // 8, 1)
        self.value_conv = nn.Conv2d(input_dim, input_dim, 1)

        self.beta = nn.Parameter(torch.tensor(beta_init))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

        self.proj = nn.Conv2d(input_dim, diff_dim, 1) if input_dim != diff_dim else nn.Identity()

    def forward(self, input, diff, s_wsi=None):
        _, _, h_orig, w_orig = input.size()

        # Downsample for efficient attention, but ensure we can always restore
        # to the exact original spatial size. When the feature map size is not
        # divisible by the pooling factor, using (h * ds) will not match (h_orig).
        ds_adaptive = int(min(self.ds, max(1, int(h_orig) // 2)))
        if ds_adaptive > 1:
            x = F.avg_pool2d(input, kernel_size=ds_adaptive, stride=ds_adaptive)
            diff = F.avg_pool2d(diff, kernel_size=ds_adaptive, stride=ds_adaptive)
        else:
            x = input
            diff = diff

        b, _, h, w = diff.size()

        proj_query = self.query_conv(diff).view(b, -1, h * w).permute(0, 2, 1)
        proj_key = self.key_conv(diff).view(b, -1, h * w)

        energy = torch.bmm(proj_query, proj_key)
        energy = (self.key_channel ** -0.5) * energy

        if s_wsi is not None:
            s_wsi_resized = F.interpolate(s_wsi, size=(h, w), mode='bilinear', align_corners=True)
            change_map = 1.0 - s_wsi_resized
            sim_flat = change_map.view(b, 1, h * w)
            energy = energy + sim_flat * self.beta

        attention = self.softmax(energy)

        proj_value = self.value_conv(x).view(b, -1, h * w)
        if s_wsi is not None:
            sim_gate = change_map.view(b, -1, h * w)
            proj_value = proj_value * (1.0 + 0.1 * sim_gate)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(b, self.input_dim, h, w)
        out = F.interpolate(out, size=(h_orig, w_orig), mode='bilinear', align_corners=True)
        out = self.gamma * out + input

        out = self.proj(out)
        return out


class MDEM_V2(nn.Module):
    def __init__(self, input_dim, diff_dim, ds=8, beta_init=0.3, n_heads=4, use_uncertainty=False):
        super().__init__()
        self.input_dim = input_dim
        self.diff_dim = diff_dim
        self.ds = ds
        self.n_heads = n_heads
        self.use_uncertainty = use_uncertainty

        self.head_dim = max(1, diff_dim // (n_heads * 8))

        self.query_conv = nn.Conv2d(diff_dim, self.head_dim * n_heads, 1)
        self.key_conv = nn.Conv2d(diff_dim, self.head_dim * n_heads, 1)
        self.value_conv = nn.Conv2d(input_dim, input_dim, 1)

        self.multi_head_fusion = nn.Conv2d(input_dim * n_heads, input_dim, 1)

        self.beta = nn.Parameter(torch.tensor(beta_init))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

        self.proj = nn.Conv2d(input_dim, diff_dim, 1) if input_dim != diff_dim else nn.Identity()

        if use_uncertainty:
            try:
                from model.module.cross_scale_attention import UncertaintyModule
            except ImportError:
                try:
                    from cross_scale_attention import UncertaintyModule
                except ImportError:
                    UncertaintyModule = None

            if UncertaintyModule is not None:
                self.uncertainty = UncertaintyModule(diff_dim, dropout_rate=0.1, n_samples=5)
            else:
                self.use_uncertainty = False

    def _adaptive_downsample(self, x, diff):
        _, _, h, w = x.size()

        if h <= 8 or w <= 8:
            ds = 2
        elif h <= 16 or w <= 16:
            ds = 4
        else:
            ds = min(self.ds, max(2, min(h, w) // 4))

        if ds > 1:
            pool = nn.AvgPool2d(ds)
            return pool(x), pool(diff), ds
        return x, diff, 1

    def _multi_head_attention(self, query, key, value, sim_bias=None):
        b, c, h, w = value.size()

        query = query.view(b, self.n_heads, self.head_dim, h * w).permute(0, 1, 3, 2)
        key = key.view(b, self.n_heads, self.head_dim, h * w)

        energy = torch.matmul(query, key)
        energy = energy * (self.head_dim ** -0.5)

        if sim_bias is not None:
            sim_flat = sim_bias.view(b, 1, 1, h * w).expand(-1, self.n_heads, h * w, -1)
            energy = energy + sim_flat * self.beta

        attention = self.softmax(energy)

        value_heads = []
        value_flat = value.view(b, c, h * w)
        for i in range(self.n_heads):
            attn_i = attention[:, i, :, :]
            out_i = torch.bmm(value_flat, attn_i.permute(0, 2, 1))
            value_heads.append(out_i.view(b, c, h, w))

        multi_head_out = torch.cat(value_heads, dim=1)
        fused_out = self.multi_head_fusion(multi_head_out)
        return fused_out

    def forward(self, input, diff, s_wsi=None):
        x, diff_down, ds = self._adaptive_downsample(input, diff)
        b, _, h, w = diff_down.size()

        proj_query = self.query_conv(diff_down)
        proj_key = self.key_conv(diff_down)
        proj_value = self.value_conv(x)

        sim_bias = None
        if s_wsi is not None:
            s_wsi_resized = F.interpolate(s_wsi, size=(h, w), mode='bilinear', align_corners=True)
            change_map = 1.0 - s_wsi_resized
            sim_bias = change_map
            proj_value = proj_value * (1.0 + 0.1 * change_map)

        out = self._multi_head_attention(proj_query, proj_key, proj_value, sim_bias)

        if ds > 1:
            out = F.interpolate(out, input.size()[2:], mode='bilinear', align_corners=True)

        out = self.gamma * out + input
        out = self.proj(out)
        return out