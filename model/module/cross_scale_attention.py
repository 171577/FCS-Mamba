import torch
import torch.nn as nn
import torch.nn.functional as F
class CrossScaleAttentionBridge(nn.Module):

    def __init__(self, channels_list, reduction=4):

        super().__init__()
        self.channels_list = channels_list

                          
                       
        self.coarse_to_fine_54 = self._build_attention_module(
            channels_list[3], channels_list[2], reduction
        )

                       
        self.coarse_to_fine_43 = self._build_attention_module(
            channels_list[2], channels_list[1], reduction
        )

                       
        self.coarse_to_fine_32 = self._build_attention_module(
            channels_list[1], channels_list[0], reduction
        )

                          
                       
        self.fine_to_coarse_23 = self._build_attention_module(
            channels_list[0], channels_list[1], reduction
        )

                       
        self.fine_to_coarse_34 = self._build_attention_module(
            channels_list[1], channels_list[2], reduction
        )

                       
        self.fine_to_coarse_45 = self._build_attention_module(
            channels_list[2], channels_list[3], reduction
        )

                    
        self.alpha_d2 = nn.Parameter(torch.tensor(0.3))              
        self.alpha_d3 = nn.Parameter(torch.tensor(0.25))
        self.alpha_d4 = nn.Parameter(torch.tensor(0.2))
        self.alpha_d5 = nn.Parameter(torch.tensor(0.15))              

    def _build_attention_module(self, from_channels, to_channels, reduction):

        mid_channels = max(to_channels // reduction, 16)

        return nn.Sequential(
                  
            nn.Conv2d(from_channels, to_channels, 1, bias=False),
            nn.BatchNorm2d(to_channels),
            nn.ReLU(inplace=True),

                  
            nn.Conv2d(to_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, to_channels, 1, bias=False),
            nn.Sigmoid()           
        )

    def forward(self, d2, d3, d4, d5):

                 
        d5_to_d4 = F.interpolate(d5, d4.shape[2:], mode='bilinear', align_corners=True)
        attn_54 = self.coarse_to_fine_54(d5_to_d4)
        d4_semantic = d4 * attn_54

                 
        d4_to_d3 = F.interpolate(d4, d3.shape[2:], mode='bilinear', align_corners=True)
        attn_43 = self.coarse_to_fine_43(d4_to_d3)
        d3_semantic = d3 * attn_43

                 
        d3_to_d2 = F.interpolate(d3, d2.shape[2:], mode='bilinear', align_corners=True)
        attn_32 = self.coarse_to_fine_32(d3_to_d2)
        d2_semantic = d2 * attn_32

                               
                 
        d2_to_d3 = F.adaptive_avg_pool2d(d2, d3.shape[2:])
        attn_23 = self.fine_to_coarse_23(d2_to_d3)
        d3_detail = d3 * attn_23

                 
        d3_to_d4 = F.adaptive_avg_pool2d(d3, d4.shape[2:])
        attn_34 = self.fine_to_coarse_34(d3_to_d4)
        d4_detail = d4 * attn_34

                 
        d4_to_d5 = F.adaptive_avg_pool2d(d4, d5.shape[2:])
        attn_45 = self.fine_to_coarse_45(d4_to_d5)
        d5_detail = d5 * attn_45

                           
        d2_out = d2 + d2_semantic * self.alpha_d2
        d3_out = d3 + (d3_semantic + d3_detail) * self.alpha_d3
        d4_out = d4 + (d4_semantic + d4_detail) * self.alpha_d4
        d5_out = d5 + d5_detail * self.alpha_d5

        return d2_out, d3_out, d4_out, d5_out


class GatedResidualFusion(nn.Module):

    def __init__(self, channels, n_features=2):

        super().__init__()
        self.n_features = n_features

              
        self.gate = nn.Sequential(
            nn.Conv2d(channels * n_features, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, n_features, 1),
            nn.Softmax(dim=1)         
        )

              
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * n_features, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

              
        self.residual_weight = nn.Parameter(torch.tensor(0.2))

    def forward(self, *features):

        if len(features) != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {len(features)}")

              
        concat_feat = torch.cat(features, dim=1)                  

                
        gate_weights = self.gate(concat_feat)                                       

              
        weighted_features = []
        for i, feat in enumerate(features):
            weight = gate_weights[:, i:i + 1, :, :]
                       
            weight = weight.expand_as(feat)
            weighted_features.append(feat * weight)

            
        fused_weighted = sum(weighted_features)

              
        fused = self.fusion(concat_feat)

              
        return fused_weighted + fused * self.residual_weight


class UncertaintyModule(nn.Module):
    def __init__(self, channels, dropout_rate=0.1, n_samples=5):

        super().__init__()
        self.dropout_rate = dropout_rate
        self.n_samples = n_samples

                        
        self.mc_dropout = nn.Dropout2d(dropout_rate)

                  
        self.uncertainty_estimator = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 1),
            nn.Sigmoid()                   
        )

    def forward(self, x, return_uncertainty=False):
        if self.training or not return_uncertainty:
                                    
            return self.mc_dropout(x)

        else:
                               
            samples = []
            for _ in range(self.n_samples):
                samples.append(self.mc_dropout(x))

                     
            samples_stacked = torch.stack(samples, dim=0)                           
            x_mean = samples_stacked.mean(dim=0)                
            x_var = samples_stacked.var(dim=0)                

                              
            uncertainty_map = x_var.mean(dim=1, keepdim=True)                

                     
            uncertainty_map = self.uncertainty_estimator(x_mean)

            return x_mean, uncertainty_map


class LightweightCrossScaleAttention(nn.Module):
    def __init__(self, channels_list):
        super().__init__()

                     
        self.guide_54 = nn.Conv2d(channels_list[3], channels_list[2], 1)
        self.guide_43 = nn.Conv2d(channels_list[2], channels_list[1], 1)
        self.guide_32 = nn.Conv2d(channels_list[1], channels_list[0], 1)

              
        self.alpha = nn.Parameter(torch.tensor(0.2))

    def forward(self, d2, d3, d4, d5):
                       
               
        d5_to_d4 = F.interpolate(d5, d4.shape[2:], mode='bilinear', align_corners=True)
        d4_out = d4 + self.guide_54(d5_to_d4) * self.alpha

        d4_to_d3 = F.interpolate(d4_out, d3.shape[2:], mode='bilinear', align_corners=True)
        d3_out = d3 + self.guide_43(d4_to_d3) * self.alpha

        d3_to_d2 = F.interpolate(d3_out, d2.shape[2:], mode='bilinear', align_corners=True)
        d2_out = d2 + self.guide_32(d3_to_d2) * self.alpha

        return d2_out, d3_out, d4_out, d5