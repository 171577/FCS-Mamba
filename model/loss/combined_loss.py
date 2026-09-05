import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.losses import dice_loss


class FocalLoss(nn.Module):

       
    def __init__(self, alpha=0.25, gamma=2.0, weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
    
    def forward(self, pred, target):

        ce_loss = F.cross_entropy(pred, target, weight=self.weight, reduction='none')
        
                        
        p = torch.exp(-ce_loss)
        
                                                
        focal_loss = self.alpha * (1 - p) ** self.gamma * ce_loss
        
        return focal_loss.mean()


class BoundaryLoss(nn.Module):

    def __init__(self, weight=2.0):
        super().__init__()
        self.weight = weight
    
    def forward(self, pred, target):

        boundary = self._compute_boundary(target)
        
                
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        weighted_loss = ce_loss * (1 + self.weight * boundary)
        
        return weighted_loss.mean()
    
    def _compute_boundary(self, target):

        kernel = torch.tensor([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ], dtype=torch.float32, device=target.device)
        
                  
        target_float = target.float().unsqueeze(1)                
        
                       
        boundary = F.conv2d(target_float, kernel.view(1, 1, 3, 3), padding=1)
        
                        
        boundary = (boundary > 0).float().squeeze(1)             
        
        return boundary


class DiceLoss(nn.Module):
                         
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):

        pred_probs = F.softmax(pred, dim=1)
        
                          
        target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
        
                  
        inter = (pred_probs * target_one_hot).sum(dim=(2, 3))
        union = pred_probs.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        dice = (2 * inter + 1e-5) / (union + 1e-5)
        
        return (1 - dice).mean()


class CombinedLoss(nn.Module):

    def __init__(
        self,
        lambda_focal=1.0,
        lambda_boundary=0.5,
        lambda_dice=0.5,
        focal_alpha=0.25,
        focal_gamma=2.0,
        boundary_weight=2.0,
        ce_weight=None
    ):
        super().__init__()
        self.lambda_focal = lambda_focal
        self.lambda_boundary = lambda_boundary
        self.lambda_dice = lambda_dice
        
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, weight=ce_weight)
        self.boundary_loss = BoundaryLoss(weight=boundary_weight)
        self.dice_loss = DiceLoss()
    
    def forward(self, pred, target):

        loss_focal = self.focal_loss(pred, target)
        loss_boundary = self.boundary_loss(pred, target)
        loss_dice = self.dice_loss(pred, target)
        
              
        total_loss = (
            self.lambda_focal * loss_focal +
            self.lambda_boundary * loss_boundary +
            self.lambda_dice * loss_dice
        )
        
                    
        loss_dict = {
            'focal': loss_focal.item(),
            'boundary': loss_boundary.item(),
            'dice': loss_dice.item(),
            'total': total_loss.item()
        }
        
        return total_loss, loss_dict


      
def create_combined_loss(device, num_classes=2, class_weights=None):

    if class_weights is not None:
        class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    
    loss_fn = CombinedLoss(
        lambda_focal=1.0,
        lambda_boundary=0.5,
        lambda_dice=0.5,
        focal_alpha=0.25,
        focal_gamma=2.0,
        boundary_weight=2.0,
        ce_weight=class_weights
    )
    
    return loss_fn
