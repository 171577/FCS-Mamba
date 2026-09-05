import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightFeatureAlign(nn.Module):
    def __init__(self, channels, offset_groups=4):

        super().__init__()
        self.channels = channels
        self.offset_groups = offset_groups

                               
                             
        self.offset_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels // 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, 2, 3, padding=1, bias=True),
            nn.Tanh()                   
        )

                    
        self.offset_scale = nn.Parameter(torch.tensor(2.0))

                      
        self.feature_enhance = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=offset_groups, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )

                
        self.residual_weight = nn.Parameter(torch.tensor(0.1))

    def _grid_sample(self, x, offset):

        B, C, H, W = x.shape

                
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)                
        grid = grid.repeat(B, 1, 1, 1)                

                               
                                         
        offset_normalized = offset * self.offset_scale / max(H, W)
        grid = grid + offset_normalized

                                          
        grid = grid.permute(0, 2, 3, 1)

                 
        aligned = F.grid_sample(
            x, grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )

        return aligned

    def forward(self, x1, x2):

                  
        concat_feat = torch.cat([x1, x2], dim=1)                 
        offset = self.offset_conv(concat_feat)                

                    
                            
        x2_aligned = self._grid_sample(x2, offset)

                 
        x1_enhanced = self.feature_enhance(x1)
        x2_enhanced = self.feature_enhance(x2_aligned)

                 
        x1_out = x1 + x1_enhanced * self.residual_weight
        x2_out = x2_aligned + x2_enhanced * self.residual_weight

        return x1_out, x2_out