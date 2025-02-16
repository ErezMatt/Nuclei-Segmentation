import torch

def accuracy(pred, target):
    correct = (pred == target).sum()
    samples = torch.numel(pred)

    return correct/samples

def dice_coef(pred, target, epsilon=1e-8):
    """Computes the Dice Coefficient for binary segmentation"""
    intersection = (pred * target).sum()

    return (2 * intersection + epsilon) / ((pred + target).sum() + epsilon)

def binary_iou(pred, target, epsilon=1e-8):
    """Computes the Intersection over Union (IoU) for binary segmentation"""
    intersection = (pred * target).sum()
    union = (pred + target).sum() - intersection
    
    return (intersection + epsilon) / (union + epsilon)