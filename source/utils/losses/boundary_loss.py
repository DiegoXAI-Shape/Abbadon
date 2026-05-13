import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import distance_transform_edt

class BoundaryLoss(nn.Module):
    def __init__(self):
        super(BoundaryLoss, self).__init__()

    def forward(self, pred, sdf_target):
        """
        pred: Tensor con logits (sin sigmoide, ej: salida de ConvNeXt)
        sdf_target: Tensor del SDF Pre-calculado por Asincronismo de CPU [B, 1, H, W]
        """
        # 1. Transformar logits a probabilidades (0 a 1).
        probs = torch.sigmoid(pred)
            
        # 2. La magia matemática: Ya no bloqueamos la GPU buscando distancias scipy
        # Simplemente castigamos los Falsos Positivos lejos del borde, 
        # y premiamos el area interior con un SDF negativo.
        loss = probs * sdf_target
        
        return torch.mean(loss)