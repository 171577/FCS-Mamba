import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence

try:
    from .depthwise_separable import DepthwiseSeparableConvWithReLU, DepthwiseSeparableConv
    from .cross_scale_attention import GatedResidualFusion
except ImportError:
    try:
        from model.module.depthwise_separable import DepthwiseSeparableConvWithReLU, DepthwiseSeparableConv
        from model.module.cross_scale_attention import GatedResidualFusion
    except ImportError:
        from depthwise_separable import DepthwiseSeparableConvWithReLU, DepthwiseSeparableConv
        from cross_scale_attention import GatedResidualFusion


class Spatial_Attention(nn.Module):
    def __init__(self, spatial_kernel=7):
        super(Spatial_Attention, self).__init__()
                                                                                          
        self.mlp = nn.Sequential(DepthwiseSeparableConv(3, 1, kernel_size=spatial_kernel, padding=spatial_kernel // 2, bias=False),
                                 nn.Sigmoid())
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        merge = avg_out + max_out
        return x * self.mlp(torch.concat([merge, avg_out, max_out], dim=1))


class Channel_Attention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(Channel_Attention, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(nn.Conv2d(channel, channel // reduction, 1, bias=False),
                                 nn.ReLU(),
                                 nn.Conv2d(channel // reduction, channel, 1, bias=False),
                                 nn.Sigmoid())
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        p = self.pool(x)
        return x * self.mlp(p)

                                            
                             
                                            

def autopad(kernel_size: int, padding: Optional[int] = None, dilation: int = 1) -> int:
                            
    if padding is None:
        padding = (kernel_size - 1) * dilation // 2
    return padding

def make_divisible(value: int, divisor: int = 8) -> int:
                        
    return int((value + divisor // 2) // divisor * divisor)

class ConvModule(nn.Module):
                  
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
            norm_cfg: Optional[dict] = None,
            act_cfg: Optional[dict] = None):
        super().__init__()
        layers = []
             
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=(norm_cfg is None)))
              
        if norm_cfg:
            layers.append(self._get_norm_layer(out_channels, norm_cfg))
             
        if act_cfg:
            layers.append(self._get_act_layer(act_cfg))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

    def _get_norm_layer(self, num_features, norm_cfg):
        if norm_cfg['type'] == 'BN':
            return nn.BatchNorm2d(num_features, momentum=norm_cfg.get('momentum', 0.1), eps=norm_cfg.get('eps', 1e-5))
        return nn.BatchNorm2d(num_features)

    def _get_act_layer(self, act_cfg):
        if act_cfg['type'] == 'ReLU':
            return nn.ReLU(inplace=True)
        if act_cfg['type'] == 'SiLU':
            return nn.SiLU(inplace=True)
        return nn.ReLU(inplace=True)

class Poly_Kernel_Inception_Block(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: Optional[int] = None,
            kernel_sizes: Sequence[int] = (3, 5, 7, 9, 11),
            dilations: Sequence[int] = (1, 1, 1, 1, 1),
            expansion: float = 1.0,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU')):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = make_divisible(int(out_channels * expansion), 8)

                                   
        self.pre_conv = ConvModule(in_channels, hidden_channels, 1, 1, 0, norm_cfg=norm_cfg, act_cfg=act_cfg)

                                                               
        self.dw_convs = nn.ModuleList()
        for k, d in zip(kernel_sizes, dilations):
            self.dw_convs.append(
                ConvModule(hidden_channels, hidden_channels, k, 1,
                           autopad(k, None, d),
                           dilation=d, groups=hidden_channels,
                           norm_cfg=None, act_cfg=None) 
            )

                                   
        self.pw_conv = ConvModule(hidden_channels, hidden_channels, 1, 1, 0, norm_cfg=norm_cfg, act_cfg=act_cfg)
        
                                   
        self.post_conv = ConvModule(hidden_channels, out_channels, 1, 1, 0, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x):
        x = self.pre_conv(x)
        
                
        x_sum = x 
        for conv in self.dw_convs:
            x_sum = x_sum + conv(x)
            
        x = self.pw_conv(x_sum)
        x = self.post_conv(x)
        return x

                                            
               
                                            

class FeatureFusionModule(nn.Module):
    def __init__(self, dim):
        super(FeatureFusionModule, self).__init__()

                                
        self.proj_in = ConvModule(
            dim * 4, dim, 1, 
            norm_cfg=dict(type='BN'), 
            act_cfg=dict(type='SiLU') 
        )

                                       
        self.pki_block = Poly_Kernel_Inception_Block(
            in_channels=dim,
            out_channels=dim,
            kernel_sizes=(3, 5, 7, 9, 11),
            dilations=(1, 1, 1, 1, 1),
            expansion=1.0,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='SiLU')
        )

        
                   
        self.proj_out = ConvModule(dim, dim, 1, norm_cfg=dict(type='BN'), act_cfg=None)

                             
        self.gated_fusion = GatedResidualFusion(dim, n_features=2)

    def forward(self, x):
                             
        x_in = self.proj_in(x)
        
                         
        x_context = self.pki_block(x_in)
        
                          
        out = self.gated_fusion(x_in, x_context)

        return self.proj_out(out)

                                            
                          
                                            

class Feature_Enhancement_Module(nn.Module):
    def __init__(self, in_d=None, out_d=64):
        super(Feature_Enhancement_Module, self).__init__()
        if in_d is None:
            in_d = [64, 128, 256, 512]
        self.in_d = in_d
        self.out_d = out_d

                                      
        self.projections = nn.ModuleList([
            nn.Conv2d(c, out_d, 1) for c in in_d
        ])

                                       
        self.fusion_s1 = FeatureFusionModule(out_d)
        self.fusion_s2 = FeatureFusionModule(out_d)
        self.fusion_s3 = FeatureFusionModule(out_d)
        self.fusion_s4 = FeatureFusionModule(out_d)

    def forward(self, c1, c2, c3, c4):
        features = [c1, c2, c3, c4]
        projected_features = [self.projections[i](f) for i, f in enumerate(features)]

                                                         
        s1_in = torch.cat([
            projected_features[0],
            F.interpolate(projected_features[1], scale_factor=2, mode='bilinear'),
            F.interpolate(projected_features[2], scale_factor=4, mode='bilinear'),
            F.interpolate(projected_features[3], scale_factor=8, mode='bilinear'),
        ], dim=1)

        s2_in = torch.cat([
            F.interpolate(projected_features[0], scale_factor=0.5, mode='bilinear'),
            projected_features[1],
            F.interpolate(projected_features[2], scale_factor=2, mode='bilinear'),
            F.interpolate(projected_features[3], scale_factor=4, mode='bilinear'),
        ], dim=1)

        s3_in = torch.cat([
            F.interpolate(projected_features[0], scale_factor=0.25, mode='bilinear'),
            F.interpolate(projected_features[1], scale_factor=0.5, mode='bilinear'),
            projected_features[2],
            F.interpolate(projected_features[3], scale_factor=2, mode='bilinear'),
        ], dim=1)

        s4_in = torch.cat([
            F.interpolate(projected_features[0], scale_factor=0.125, mode='bilinear'),
            F.interpolate(projected_features[1], scale_factor=0.25, mode='bilinear'),
            F.interpolate(projected_features[2], scale_factor=0.5, mode='bilinear'),
            projected_features[3],
        ], dim=1)

        s1 = self.fusion_s1(s1_in)
        s2 = self.fusion_s2(s2_in)
        s3 = self.fusion_s3(s3_in)
        s4 = self.fusion_s4(s4_in)

        return s1, s2, s3, s4
