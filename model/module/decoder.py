import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .sim_agg import Feature_Enhancement_Module
    from .depthwise_separable import DepthwiseSeparableConvWithReLU
except ImportError:
    try:
        from model.module.sim_agg import Feature_Enhancement_Module
        from model.module.depthwise_separable import DepthwiseSeparableConvWithReLU
    except ImportError:
        from sim_agg import Feature_Enhancement_Module
        from depthwise_separable import DepthwiseSeparableConvWithReLU


class _ConvBlockAdapter(nn.Module):
    def __init__(self, op: nn.Module):
        super().__init__()
        self.op = op

    def forward(self, x):
        return self.op(x)
class Similarity_Fusion_Module(nn.Module):
    def __init__(self, channel, reduction=16, use_depthwise=True):
        super(Similarity_Fusion_Module, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

                                       
        self.mlp1 = nn.Sequential(nn.Conv2d(channel * 2, channel // reduction, 1, bias=False),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(channel // reduction, channel, 1, bias=False))

        self.mlp2 = nn.Sequential(nn.Conv2d(channel * 2, channel // reduction, 1, bias=False),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(channel // reduction, channel, 1, bias=False))

                           
                          
        if use_depthwise:
            self.conv = nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=1, bias=False),                 
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True),
                DepthwiseSeparableConvWithReLU(channel, channel, kernel_size=3, padding=1)
            )
        else:
            self.conv = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False),                
                                      nn.BatchNorm2d(channel),
                                      nn.ReLU(inplace=True))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, sim):
                                       
        max_self = self.max_pool(x)
        avg_self = self.avg_pool(x)
        channel_self = self.sigmoid(self.mlp1(torch.cat([max_self, avg_self], dim=1)))
        
                              
                             
        sim_weight = sim
        
                               
                                   
                                
                                                            
        
                             
                  
        fused = x * channel_self + x * sim_weight * 0.5
        out = self.conv(fused)
        
        max_out = self.max_pool(out)
        avg_out = self.avg_pool(out)
        channel_out = self.sigmoid(self.mlp2(torch.cat([max_out, avg_out], dim=1)))
        out = channel_out * out

        return out


class Interaction_Module(nn.Module):
    def __init__(self, channels, num_paths=2):
        super(Interaction_Module, self).__init__()
        self.num_paths = num_paths
        attn_channels = channels // 16
        attn_channels = max(attn_channels, 8)

        self.fc_reduce = nn.Conv2d(channels, attn_channels, kernel_size=1, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.fc_select = nn.Conv2d(attn_channels, channels * num_paths, kernel_size=1, bias=False)

    def forward(self, x1, x2):

        x = torch.stack([x1, x2], dim=1)
        attn = x.sum(1).mean((2, 3), keepdim=True)
        attn = self.fc_reduce(attn)
        attn = self.act(attn)
        attn = self.fc_select(attn)
        B, C, H, W = attn.shape
        attn1, attn2 = attn.reshape(B, self.num_paths, C // self.num_paths, H, W).transpose(0, 1)
        attn1 = torch.sigmoid(attn1)
        attn2 = torch.sigmoid(attn2)

        return x1 * attn1, x2 * attn2



class GlobalLowDimensionalFeature(nn.Module):

       
    def __init__(self, in_channels_list=[64, 128, 256, 256], low_dim=32):
        super(GlobalLowDimensionalFeature, self).__init__()
        self.low_dim = low_dim
        
                
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
                     
        total_channels = sum(in_channels_list)
        self.compress = nn.Sequential(
            nn.Conv2d(total_channels, low_dim * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(low_dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(low_dim * 2, low_dim, kernel_size=1, bias=False)
        )
    
    def forward(self, d2, d3, d4, d5):

        d2_pool = self.global_pool(d2)                 
        d3_pool = self.global_pool(d3)                  
        d4_pool = self.global_pool(d4)                  
        d5_pool = self.global_pool(d5)                  
        
                  
        global_feat = torch.cat([d2_pool, d3_pool, d4_pool, d5_pool], dim=1)
        
                
        global_feat = self.compress(global_feat)                      
        
        return global_feat


class Decoder(nn.Module):
    def __init__(self, in_d, out_d):
        super(Decoder, self).__init__()
        self.in_d = in_d
        self.out_d = out_d

        self.conv5 = nn.Sequential(nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(self.in_d),
                                   nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(self.in_d),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(self.in_d),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(self.in_d),
                                   nn.ReLU(inplace=True))

        self.cls = nn.Conv2d(self.in_d, self.out_d, kernel_size=1)

        self.SFM5 = Similarity_Fusion_Module(in_d)
        self.SFM4 = Similarity_Fusion_Module(in_d)
        self.SFM3 = Similarity_Fusion_Module(in_d)
        self.SFM2 = Similarity_Fusion_Module(in_d)

    def forward(self, d5, d4, d3, d2, sim5, sim4, sim3, sim2):

        d5 = self.conv5(d5)
        d5 = self.SFM5(d5, sim5)
        d5 = F.interpolate(d5, d4.size()[2:], mode='bilinear')

        d4 = self.conv4(d4 + d5)
        d4 = self.SFM4(d4, sim4)
        d4 = F.interpolate(d4, d3.size()[2:], mode='bilinear')

        d3 = self.conv3(d3 + d4)
        d3 = self.SFM3(d3, sim3)
        d3 = F.interpolate(d3, d2.size()[2:], mode='bilinear')

        d2 = self.conv2(d2 + d3)
        d2 = self.SFM2(d2, sim2)

        mask = self.cls(d2)

        return mask


class Decoder_MultiScale(nn.Module):
                                                              
    def __init__(self, in_d_list=[64, 128, 256, 256], out_d=2, use_depthwise=True):
        super(Decoder_MultiScale, self).__init__()
        self.in_d_list = in_d_list                    
        self.out_d = out_d
        
                                                       
                          
        if use_depthwise:
            self.conv5 = _ConvBlockAdapter(DepthwiseSeparableConvWithReLU(in_d_list[3], in_d_list[3], kernel_size=3, padding=1))
            self.conv4 = _ConvBlockAdapter(DepthwiseSeparableConvWithReLU(in_d_list[2], in_d_list[2], kernel_size=3, padding=1))
            self.conv3 = _ConvBlockAdapter(DepthwiseSeparableConvWithReLU(in_d_list[1], in_d_list[1], kernel_size=3, padding=1))
            self.conv2 = _ConvBlockAdapter(DepthwiseSeparableConvWithReLU(in_d_list[0], in_d_list[0], kernel_size=3, padding=1))
        else:
            self.conv5 = _ConvBlockAdapter(nn.Sequential(
                nn.Conv2d(in_d_list[3], in_d_list[3], kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(in_d_list[3]),
                nn.ReLU(inplace=True)
            ))
            self.conv4 = _ConvBlockAdapter(nn.Sequential(
                nn.Conv2d(in_d_list[2], in_d_list[2], kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(in_d_list[2]),
                nn.ReLU(inplace=True)
            ))
            self.conv3 = _ConvBlockAdapter(nn.Sequential(
                nn.Conv2d(in_d_list[1], in_d_list[1], kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(in_d_list[1]),
                nn.ReLU(inplace=True)
            ))
            self.conv2 = _ConvBlockAdapter(nn.Sequential(
                nn.Conv2d(in_d_list[0], in_d_list[0], kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(in_d_list[0]),
                nn.ReLU(inplace=True)
            ))
        
                                          
                                 
        self.fuse4 = nn.Sequential(
            nn.Conv2d(in_d_list[3] + in_d_list[2], int(in_d_list[2] * 1.5), kernel_size=1),
            nn.BatchNorm2d(int(in_d_list[2] * 1.5)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_d_list[2] * 1.5), in_d_list[2], kernel_size=1)
        )
        self.fuse3 = nn.Sequential(
            nn.Conv2d(in_d_list[2] + in_d_list[1], int(in_d_list[1] * 1.5), kernel_size=1),
            nn.BatchNorm2d(int(in_d_list[1] * 1.5)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_d_list[1] * 1.5), in_d_list[1], kernel_size=1)
        )
        self.fuse2 = nn.Sequential(
            nn.Conv2d(in_d_list[1] + in_d_list[0], int(in_d_list[0] * 1.5), kernel_size=1),
            nn.BatchNorm2d(int(in_d_list[0] * 1.5)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_d_list[0] * 1.5), in_d_list[0], kernel_size=1)
        )
        
                          
        self.cls = nn.Conv2d(in_d_list[0], out_d, kernel_size=1)
        
                                   
        self.SFM5 = Similarity_Fusion_Module(in_d_list[3])
        self.SFM4 = Similarity_Fusion_Module(in_d_list[2])
        self.SFM3 = Similarity_Fusion_Module(in_d_list[1])
        self.SFM2 = Similarity_Fusion_Module(in_d_list[0])
        
                                   
        self.global_low_dim = GlobalLowDimensionalFeature(in_d_list, low_dim=32)
        
                         
        self.global_fusion_d5 = nn.Sequential(
            nn.Conv2d(in_d_list[3] + 32, in_d_list[3], kernel_size=1, bias=False),
            nn.BatchNorm2d(in_d_list[3]),
            nn.ReLU(inplace=True)
        )
        self.global_fusion_d4 = nn.Sequential(
            nn.Conv2d(in_d_list[2] + 32, in_d_list[2], kernel_size=1, bias=False),
            nn.BatchNorm2d(in_d_list[2]),
            nn.ReLU(inplace=True)
        )
        self.global_fusion_d3 = nn.Sequential(
            nn.Conv2d(in_d_list[1] + 32, in_d_list[1], kernel_size=1, bias=False),
            nn.BatchNorm2d(in_d_list[1]),
            nn.ReLU(inplace=True)
        )
        self.global_fusion_d2 = nn.Sequential(
            nn.Conv2d(in_d_list[0] + 32, in_d_list[0], kernel_size=1, bias=False),
            nn.BatchNorm2d(in_d_list[0]),
            nn.ReLU(inplace=True)
        )
        
                                       
                                
        self.global_weight_d5 = nn.Parameter(torch.tensor(0.2))        
        self.global_weight_d4 = nn.Parameter(torch.tensor(0.3))        
        self.global_weight_d3 = nn.Parameter(torch.tensor(0.4))       
        self.global_weight_d2 = nn.Parameter(torch.tensor(0.5))       
        
                            
                                
                                  
        self.residual_alpha_d4 = nn.Parameter(torch.tensor(0.4))          
        self.residual_alpha_d3 = nn.Parameter(torch.tensor(0.3))         
        self.residual_alpha_d2 = nn.Parameter(torch.tensor(0.2))         

                                     
                             
        feedback_dim = 32         
        
        self.feedback_compress_d2 = nn.Sequential(
            nn.Conv2d(in_d_list[0], feedback_dim, 1, bias=False),
            nn.BatchNorm2d(feedback_dim),
            nn.ReLU(inplace=True)
        )
        self.feedback_compress_d3 = nn.Sequential(
            nn.Conv2d(in_d_list[1], feedback_dim, 1, bias=False),
            nn.BatchNorm2d(feedback_dim),
            nn.ReLU(inplace=True)
        )
        self.feedback_compress_d4 = nn.Sequential(
            nn.Conv2d(in_d_list[2], feedback_dim, 1, bias=False),
            nn.BatchNorm2d(feedback_dim),
            nn.ReLU(inplace=True)
        )
        
                            
        self.feedback_fusion_d3 = nn.Sequential(
            nn.Conv2d(in_d_list[1] + feedback_dim, in_d_list[1], 1, bias=False),
            nn.BatchNorm2d(in_d_list[1]),
            nn.ReLU(inplace=True)
        )
        self.feedback_fusion_d4 = nn.Sequential(
            nn.Conv2d(in_d_list[2] + feedback_dim, in_d_list[2], 1, bias=False),
            nn.BatchNorm2d(in_d_list[2]),
            nn.ReLU(inplace=True)
        )
        self.feedback_fusion_d5 = nn.Sequential(
            nn.Conv2d(in_d_list[3] + feedback_dim, in_d_list[3], 1, bias=False),
            nn.BatchNorm2d(in_d_list[3]),
            nn.ReLU(inplace=True)
        )
        
                         
        self.feedback_gate_d3 = nn.Parameter(torch.tensor(0.1))
        self.feedback_gate_d4 = nn.Parameter(torch.tensor(0.1))
        self.feedback_gate_d5 = nn.Parameter(torch.tensor(0.1))
        
                             
        self.aux_cls_d3 = nn.Conv2d(in_d_list[1], out_d, 1)           
        self.aux_cls_d4 = nn.Conv2d(in_d_list[2], out_d, 1)            
        
                                
                        
        self.multi_scale_aggregator = nn.Sequential(
            nn.Conv2d(sum(in_d_list), in_d_list[0], 1, bias=False),
            nn.BatchNorm2d(in_d_list[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_d_list[0], in_d_list[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(in_d_list[0]),
            nn.ReLU(inplace=True)
        )

    def forward(self, d5, d4, d3, d2, sim5, sim4, sim3, sim2):
                                            
        global_feat = self.global_low_dim(d2, d3, d4, d5)                 
        
                                                   
                    
        d5 = self.conv5(d5)
        
                                   
        global_feat_d5 = F.interpolate(global_feat, d5.size()[2:], mode='bilinear', align_corners=True)
        d5 = torch.cat([d5, global_feat_d5 * self.global_weight_d5], dim=1)
        d5 = self.global_fusion_d5(d5)
        
        d5 = self.SFM5(d5, sim5)
        d5_up = F.interpolate(d5, d4.size()[2:], mode='bilinear', align_corners=True)
        
                                   
        d4_input = d4                
        d4 = self.conv4(d4)
        
                                   
        global_feat_d4 = F.interpolate(global_feat, d4.size()[2:], mode='bilinear', align_corners=True)
        d4 = torch.cat([d4, global_feat_d4 * self.global_weight_d4], dim=1)
        d4 = self.global_fusion_d4(d4)
        
        d4 = self.fuse4(torch.cat([d4, d5_up], dim=1))
        d4 = self.SFM4(d4, sim4)
        d4 = d4 + d4_input * self.residual_alpha_d4             
        d4_up = F.interpolate(d4, d3.size()[2:], mode='bilinear', align_corners=True)
        
                                   
        d3_input = d3                
        d3 = self.conv3(d3)
        
                                   
        global_feat_d3 = F.interpolate(global_feat, d3.size()[2:], mode='bilinear', align_corners=True)
        d3 = torch.cat([d3, global_feat_d3 * self.global_weight_d3], dim=1)
        d3 = self.global_fusion_d3(d3)
        
        d3 = self.fuse3(torch.cat([d3, d4_up], dim=1))
        d3 = self.SFM3(d3, sim3)
        d3 = d3 + d3_input * self.residual_alpha_d3             
        d3_up = F.interpolate(d3, d2.size()[2:], mode='bilinear', align_corners=True)
        
                                   
        d2_input = d2                
        d2 = self.conv2(d2)
        
                                   
        global_feat_d2 = F.interpolate(global_feat, d2.size()[2:], mode='bilinear', align_corners=True)
        d2 = torch.cat([d2, global_feat_d2 * self.global_weight_d2], dim=1)
        d2 = self.global_fusion_d2(d2)
        
        d2 = self.fuse2(torch.cat([d2, d3_up], dim=1))
        d2 = self.SFM2(d2, sim2)
        d2 = d2 + d2_input * self.residual_alpha_d2             
        
                                                    
                                    
                               
        
                    
        fb_d2 = self.feedback_compress_d2(d2)
        fb_d2_up = F.interpolate(fb_d2, d3.size()[2:], mode='bilinear', align_corners=True)
        d3_with_fb = torch.cat([d3, fb_d2_up], dim=1)
        d3_refined = self.feedback_fusion_d3(d3_with_fb)
        d3 = d3 + d3_refined * self.feedback_gate_d3
        
                    
        fb_d3 = self.feedback_compress_d3(d3)
        fb_d3_up = F.interpolate(fb_d3, d4.size()[2:], mode='bilinear', align_corners=True)
        d4_with_fb = torch.cat([d4, fb_d3_up], dim=1)
        d4_refined = self.feedback_fusion_d4(d4_with_fb)
        d4 = d4 + d4_refined * self.feedback_gate_d4
        
                    
        fb_d4 = self.feedback_compress_d4(d4)
        fb_d4_up = F.interpolate(fb_d4, d5.size()[2:], mode='bilinear', align_corners=True)
        d5_with_fb = torch.cat([d5, fb_d4_up], dim=1)
        d5_refined = self.feedback_fusion_d5(d5_with_fb)
        d5 = d5 + d5_refined * self.feedback_gate_d5
        
                                         
                                   
        
                                
        d5_up = F.interpolate(d5, d4.size()[2:], mode='bilinear', align_corners=True)
                              
        if d5_up.shape[1] != d4.shape[1]:
            d5_up = F.interpolate(d5_up, size=(d4.shape[2], d4.shape[3]), mode='bilinear', align_corners=True)
                       
            d54 = self.fuse4(torch.cat([d5_up, d4], dim=1))
        else:
            d54 = d4 + d5_up * 0.5
        
        d54_up = F.interpolate(d54, d3.size()[2:], mode='bilinear', align_corners=True)
        if d54_up.shape[1] != d3.shape[1]:
            d543 = self.fuse3(torch.cat([d54_up, d3], dim=1))
        else:
            d543 = d3 + d54_up * 0.5
        
        d543_up = F.interpolate(d543, d2.size()[2:], mode='bilinear', align_corners=True)
        if d543_up.shape[1] != d2.shape[1]:
            aggregated_feat = self.fuse2(torch.cat([d543_up, d2], dim=1))
        else:
            aggregated_feat = d2 + d543_up * 0.5
        
              
        mask = self.cls(aggregated_feat)
        
                     
        if self.training:
            aux_mask_d3 = self.aux_cls_d3(d3)
            aux_mask_d4 = self.aux_cls_d4(d4)
            return mask, aux_mask_d3, aux_mask_d4
        else:
            return mask


class Decoder_sim(nn.Module):
       
    def __init__(self, in_d, out_d=64, use_depthwise=True, use_fem=True, use_im=True):
        super(Decoder_sim, self).__init__()
        self.use_fem = use_fem
        self.use_im = use_im

        # 辅助多通道处理
        if isinstance(in_d, list):
            self.in_d_list = in_d                       
            self.use_multi_channel = True
        else:
            self.in_d_list = [in_d] * 4             
            self.use_multi_channel = False

        self.out_d = out_d

        # 消融实验 2：是否开启 FEM
        if self.use_fem:
            self.FEM = Feature_Enhancement_Module(in_d=self.in_d_list, out_d=out_d)
        else:
            # 如果不使用 FEM，则简单的进行通道投影
            self.projections = nn.ModuleList([
                nn.Conv2d(c, out_d, 1) for c in self.in_d_list
            ])

                                
                          
        if use_depthwise:
            self.conv4 = DepthwiseSeparableConvWithReLU(out_d, out_d, kernel_size=3, padding=1)
            self.conv3 = DepthwiseSeparableConvWithReLU(out_d, out_d, kernel_size=3, padding=1)
            self.conv2 = DepthwiseSeparableConvWithReLU(out_d, out_d, kernel_size=3, padding=1)
            self.conv1 = DepthwiseSeparableConvWithReLU(out_d, out_d, kernel_size=3, padding=1)
        else:
            self.conv4 = nn.Sequential(nn.Conv2d(out_d, out_d, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(out_d),
                                       nn.ReLU(inplace=True))
            self.conv3 = nn.Sequential(nn.Conv2d(out_d, out_d, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(out_d),
                                       nn.ReLU(inplace=True))
            self.conv2 = nn.Sequential(nn.Conv2d(out_d, out_d, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(out_d),
                                       nn.ReLU(inplace=True))
            self.conv1 = nn.Sequential(nn.Conv2d(out_d, out_d, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(out_d),
                                       nn.ReLU(inplace=True))

                                           
        # 消融实验 3：是否开启 Interaction Module (IM)
        if self.use_im:
            self.IM4 = Interaction_Module(channels=out_d)
            self.IM3 = Interaction_Module(channels=out_d)
            self.IM2 = Interaction_Module(channels=out_d)
            self.IM1 = Interaction_Module(channels=out_d)
        else:
            self.IM4 = self.IM3 = self.IM2 = self.IM1 = None

    def cal_sim(self, x1, x2):
                                                      
        sim = F.cosine_similarity(x1, x2, dim=1)
                                        
        sim = torch.sigmoid(sim)
                                                      
        return sim.unsqueeze(dim=1)

    def forward(self, x1_1, x1_2, x1_3, x1_4, x2_1, x2_2, x2_3, x2_4):

        x1_features = [x1_1, x1_2, x1_3, x1_4]
        x2_features = [x2_1, x2_2, x2_3, x2_4]

        if self.use_fem:
            x1_1, x1_2, x1_3, x1_4 = self.FEM(*x1_features)
            x2_1, x2_2, x2_3, x2_4 = self.FEM(*x2_features)
        else:
            # 消融 FEM：直接使用投影后的特征
            x1_1, x1_2, x1_3, x1_4 = [self.projections[i](f) for i, f in enumerate(x1_features)]
            x2_1, x2_2, x2_3, x2_4 = [self.projections[i](f) for i, f in enumerate(x2_features)]

        x1_4 = self.conv4(x1_4)
        x2_4 = self.conv4(x2_4)
        if self.use_im:
            x1_4, x2_4 = self.IM4(x1_4, x2_4)
        sim4 = self.cal_sim(x1_4, x2_4)
        x1_4 = F.interpolate(x1_4, x1_3.size()[2:], mode='bilinear')
        x2_4 = F.interpolate(x2_4, x2_3.size()[2:], mode='bilinear')

        x1_3 = self.conv3(x1_4 + x1_3)
        x2_3 = self.conv3(x2_4 + x2_3)
        if self.use_im:
            x1_3, x2_3 = self.IM3(x1_3, x2_3)
        sim3 = self.cal_sim(x1_3, x2_3)
        x1_3 = F.interpolate(x1_3, x1_2.size()[2:], mode='bilinear')
        x2_3 = F.interpolate(x2_3, x2_2.size()[2:], mode='bilinear')

        x1_2 = self.conv2(x1_3 + x1_2)
        x2_2 = self.conv2(x2_3 + x2_2)
        if self.use_im:
            x1_2, x2_2 = self.IM2(x1_2, x2_2)
        sim2 = self.cal_sim(x1_2, x2_2)
        x1_2 = F.interpolate(x1_2, x1_1.size()[2:], mode='bilinear')
        x2_2 = F.interpolate(x2_2, x2_1.size()[2:], mode='bilinear')

        x1_1 = self.conv1(x1_2 + x1_1)
        x2_1 = self.conv1(x2_2 + x2_1) 
        if self.use_im:
            x1_1, x2_1 = self.IM1(x1_1, x2_1)
        sim1 = self.cal_sim(x1_1, x2_1)

        return sim4, sim3, sim2, sim1
