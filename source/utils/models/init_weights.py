import torch.nn as nn

def init_weights(m):
    """
    Inicializa los pesos de las capas convolucionales y lineales usando
    Kaiming Normal (He Initialization), ideal para activaciones ReLU/GELU.
    También inicializa los BatchNorms y LayerNorms con pesos en 1 y bias en 0.
    """
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
            
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
            
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
