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

class Decoder_MultiScale_WithFeedback(nn.Module):
    def __init__(self, in_d_list=[64, 128, 256, 256], out_d=2, use_depthwise=True, use_feedback=True):
        super(Decoder_MultiScale_WithFeedback, self).__init__()
        self.in_d_list = in_d_list                    
        self.out_d = out_d
        self.use_feedback = use_feedback

                                                  
                                                       
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
            nn.Conv2d(in_d_list[3] + in_d_list[2], in_d_list[2]*2, kernel_size=1),
            nn.BatchNorm2d(in_d_list[2]*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_d_list[2]*2, in_d_list[2], kernel_size=1)
        )
        self.fuse3 = nn.Sequential(
            nn.Conv2d(in_d_list[2] + in_d_list[1], in_d_list[1]*2, kernel_size=1),
            nn.BatchNorm2d(in_d_list[1]*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_d_list[1]*2, in_d_list[1], kernel_size=1)
        )
        self.fuse2 = nn.Sequential(
            nn.Conv2d(in_d_list[1] + in_d_list[0], in_d_list[0]*2, kernel_size=1),
            nn.BatchNorm2d(in_d_list[0]*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_d_list[0]*2, in_d_list[0], kernel_size=1)
        )

                                   
        try:
            from .decoder import Similarity_Fusion_Module
        except ImportError:
            try:
                from model.module.decoder import Similarity_Fusion_Module
            except ImportError:
                from decoder import Similarity_Fusion_Module
        self.SFM5 = Similarity_Fusion_Module(in_d_list[3])
        self.SFM4 = Similarity_Fusion_Module(in_d_list[2])
        self.SFM3 = Similarity_Fusion_Module(in_d_list[1])
        self.SFM2 = Similarity_Fusion_Module(in_d_list[0])

                            
                                
                                  
        self.residual_alpha_d4 = nn.Parameter(torch.tensor(0.4))          
        self.residual_alpha_d3 = nn.Parameter(torch.tensor(0.3))         
        self.residual_alpha_d2 = nn.Parameter(torch.tensor(0.2))         

                                                       
        if use_feedback:
            feedback_dim = 48                                

                                     
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

                                            
                               
            self.feedback_gate_d3 = nn.Parameter(torch.tensor(0.4))                
            self.feedback_gate_d4 = nn.Parameter(torch.tensor(0.3))                  
            self.feedback_gate_d5 = nn.Parameter(torch.tensor(0.2))                  

                                                     
            self.aux_cls_d2 = nn.Conv2d(in_d_list[0], out_d, 1)                          
            self.aux_cls_d3 = nn.Conv2d(in_d_list[1], out_d, 1)              
            self.aux_cls_d4 = nn.Conv2d(in_d_list[2], out_d, 1)               

                                                        
                            
            self.multi_scale_aggregator = nn.Sequential(
                nn.Conv2d(sum(in_d_list), in_d_list[0], 1, bias=False),
                nn.BatchNorm2d(in_d_list[0]),
                nn.ReLU(inplace=True),
                DepthwiseSeparableConvWithReLU(in_d_list[0], in_d_list[0], 3, padding=1)
            )

               
        self.cls = nn.Conv2d(in_d_list[0], out_d, 1)

                                                
        self.edge_branch = nn.Sequential(
            DepthwiseSeparableConvWithReLU(in_d_list[0], 32, 3, padding=1),
            nn.Conv2d(32, 1, 1)
        )

    def forward(self, d5, d4, d3, d2, sim5, sim4, sim3, sim2):
 
                                                         
                                                     

                    
        d5 = self.conv5(d5)
        d5 = self.SFM5(d5, sim5)
        d5_up = F.interpolate(d5, d4.size()[2:], mode='bilinear', align_corners=True)

                            
        d4_input = d4                
        d4 = self.conv4(d4)
        d4 = self.fuse4(torch.cat([d4, d5_up], dim=1))
        d4 = self.SFM4(d4, sim4)
        d4 = d4 + d4_input * self.residual_alpha_d4
        d4_up = F.interpolate(d4, d3.size()[2:], mode='bilinear', align_corners=True)

                            
        d3_input = d3
        d3 = self.conv3(d3)
        d3 = self.fuse3(torch.cat([d3, d4_up], dim=1))
        d3 = self.SFM3(d3, sim3)
        d3 = d3 + d3_input * self.residual_alpha_d3
        d3_up = F.interpolate(d3, d2.size()[2:], mode='bilinear', align_corners=True)

                            
        d2_input = d2
        d2 = self.conv2(d2)
        d2 = self.fuse2(torch.cat([d2, d3_up], dim=1))
        d2 = self.SFM2(d2, sim2)
        d2 = d2 + d2_input * self.residual_alpha_d2

                                                            
        if self.use_feedback:
                                                         
                        

                        
            fb_d2 = self.feedback_compress_d2(d2)                     
            fb_d2_down = F.adaptive_avg_pool2d(fb_d2, d3.size()[2:])                     
            d3_with_fb = torch.cat([d3, fb_d2_down], dim=1)                         
            d3_refined = self.feedback_fusion_d3(d3_with_fb)                      
            d3 = d3 + d3_refined * self.feedback_gate_d3        

                        
            fb_d3 = self.feedback_compress_d3(d3)                     
            fb_d3_down = F.adaptive_avg_pool2d(fb_d3, d4.size()[2:])                       
            d4_with_fb = torch.cat([d4, fb_d3_down], dim=1)                           
            d4_refined = self.feedback_fusion_d4(d4_with_fb)                        
            d4 = d4 + d4_refined * self.feedback_gate_d4

                        
            fb_d4 = self.feedback_compress_d4(d4)                       
            fb_d4_down = F.adaptive_avg_pool2d(fb_d4, d5.size()[2:])                       
            d5_with_fb = torch.cat([d5, fb_d4_down], dim=1)                           
            d5_refined = self.feedback_fusion_d5(d5_with_fb)                        
            d5 = d5 + d5_refined * self.feedback_gate_d5

                                                 
        if self.use_feedback:
                                                           
                                                    
            d5_to_d4 = F.interpolate(d5, d4.size()[2:], mode='bilinear', align_corners=True)
            d5_to_d3 = F.interpolate(d5_to_d4, d3.size()[2:], mode='bilinear', align_corners=True)
            d5_to_d2 = F.interpolate(d5_to_d3, d2.size()[2:], mode='bilinear', align_corners=True)
            
                                             
            d4_to_d3 = F.interpolate(d4, d3.size()[2:], mode='bilinear', align_corners=True)
            d4_to_d2 = F.interpolate(d4_to_d3, d2.size()[2:], mode='bilinear', align_corners=True)
            
                                  
            d3_to_d2 = F.interpolate(d3, d2.size()[2:], mode='bilinear', align_corners=True)

                      
            multi_scale_feat = torch.cat([d2, d3_to_d2, d4_to_d2, d5_to_d2], dim=1)
                                                                
            aggregated_feat = self.multi_scale_aggregator(multi_scale_feat)

                  
            main_mask = self.cls(aggregated_feat)

                  
            edge_pred = self.edge_branch(d2)
            edge_pred_sig = torch.sigmoid(edge_pred)

                        
            mask = main_mask + edge_pred_sig

                         
            if self.training:
                aux_mask_d2 = self.aux_cls_d2(d2)                   
                aux_mask_d3 = self.aux_cls_d3(d3)
                aux_mask_d4 = self.aux_cls_d4(d4)
                return mask, aux_mask_d2, aux_mask_d3, aux_mask_d4
            else:
                return mask
        else:
                                    
            main_mask = self.cls(d2)
            edge_pred = self.edge_branch(d2)
            edge_pred_sig = torch.sigmoid(edge_pred)
            mask = main_mask + edge_pred_sig

            if self.training:
                                     
                return mask, None, None, edge_pred
            return mask


           
Decoder_MultiScale_V2 = Decoder_MultiScale_WithFeedback

