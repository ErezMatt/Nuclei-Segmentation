import torch
import torch.nn as nn
import torch.nn.functional as F
from metrics import dice_coef

class MultiLoss(nn.Module):
    """
    Multi loss function that combines Binary Cross Entropy (BCE) Loss, Focal Loss, and Dice Loss.
    """
    
    def __init__(self, alpha=0.25, gamma=2, epsilon=1e-6):
        """
        Args:
            alpha (float): Weighting factor for focal loss.
            gamma (float): Focusing parameter for focal loss.
            epsilon (float): Smoothing factor for Dice loss to avoid division by zero.
        """
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def focal_loss(self, logits, targets):
        preds = torch.sigmoid(logits)
        pt = targets * preds + (1 - targets) * (1 - preds)  # p_t
        focal_weight = self.alpha * (1 - pt) ** self.gamma  # (1 - pt)^gamma
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        return (focal_weight * bce).mean()

    def dice_loss(self, logits, targets):
        preds = torch.sigmoid(logits)
        return 1 - dice_coef(preds, targets, self.epsilon)

    def forward(self, logits, targets):
        bce = self.bce_loss(logits, targets)
        focal = self.focal_loss(logits, targets)
        dice = self.dice_loss(logits, targets)

        total_loss = bce + focal + dice
        return total_loss