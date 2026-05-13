import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Calcula el Dice Loss para segmentaciones binarias.
        logits: [B, 1, H, W] - Predicciones crudas de la red
        targets: [B, 1, H, W] - Etiquetas binarias (0 o 1)
        """
        probs = torch.sigmoid(logits)
        
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice
