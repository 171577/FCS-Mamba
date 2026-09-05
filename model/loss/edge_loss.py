import torch
import torch.nn as nn
import torch.nn.functional as F
from model.loss.dice import dice_loss_v1

class Sobel(nn.Module):
                                        
    def __init__(self):
        super().__init__()
        self.filter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1, bias=False)

        Gx = torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
        Gy = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
        G = G.unsqueeze(1)
        self.filter.weight = nn.Parameter(G, requires_grad=False)

    def forward(self, img):
        x = self.filter(img)
        x = torch.mul(x, x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.sqrt(x + 1e-6)
        return x

class EdgeAwareLoss(nn.Module):
                                                           
    def __init__(self, main_loss_ce, main_loss_dice, edge_loss=nn.BCEWithLogitsLoss(), lambda_main_dice=1.0, lambda_edge=0.5):
        super().__init__()
        self.main_loss_ce = main_loss_ce
        self.main_loss_dice = main_loss_dice
        self.edge_loss = edge_loss
        self.sobel = Sobel().cuda()
        self.lambda_main_dice = lambda_main_dice
        self.lambda_edge = lambda_edge

    def forward(self, pred, gt, edge_pred):
                                              
        loss_main_ce = self.main_loss_ce(pred, gt)
        loss_main_dice = self.main_loss_dice(pred, gt.unsqueeze(1).float())
        loss_main = loss_main_ce + self.lambda_main_dice * loss_main_dice

                                                   
        gt_edge = self.sobel(gt.unsqueeze(1).float()).detach()
        loss_edge = self.edge_loss(edge_pred, gt_edge)

        return loss_main + self.lambda_edge * loss_edge
