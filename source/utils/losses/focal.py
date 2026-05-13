import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        # alpha puede ser una lista de pesos, ej: [1.0, 0.5, 2.0]
        self.alpha = torch.tensor(alpha) if alpha is not None else None
        self.gamma = gamma

    def forward(self, inputs, targets):
        # inputs: Predicciones crudas [B, C, H, W]
        # targets: Máscaras reales [B, H, W]

        # 1. Calculamos la Cross-Entropy normal
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 2. Sacamos pt (la probabilidad de la clase correcta) usando matemáticas inversas
        pt = torch.exp(-ce_loss) 
        
        # 3. Aplicamos el factor gamma: (1 - pt)^gamma * CE
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        # 4. Aplicamos sus pesos alpha si los configuró
        if self.alpha is not None:
            self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets] # Le pone el peso exacto a cada píxel según su clase
            focal_loss = alpha_t * focal_loss

        return focal_loss.mean()