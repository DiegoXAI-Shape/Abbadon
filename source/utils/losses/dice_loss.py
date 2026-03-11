import torch
from torch import nn
from torch.nn import functional as F

class GeneralizedDiceLossFN(nn.Module):
    def __init__(self, epsilon, target_classes):
        super(GeneralizedDiceLossFN, self).__init__()
        self.epsilon:float = epsilon
        self.target_classes:int = target_classes

    def forward(self, inTensor:torch.Tensor, target:torch.Tensor):

        """
        Attributes:
            - Input: Es el vector de entrada donde contiene los logits de la predicción, con shape = [Batch, N_clases, H, W]
            - Target: Es el tensor objetivo, donde se tiene que acercar, con shape = [Batch, H, W]
        """
        input = F.softmax(input=inTensor, dim = 1)
        input = input.view(input.size(0), input.size(1), -1).float()

        target = F.one_hot(target, self.target_classes).permute(0, 3, 1, 2).float()

        volumenes = torch.sum(target, dim = (2, 3))
        w_c = 1 / (volumenes ** 2 + self.epsilon)

        target = target.view(target.size(0), target.size(1), -1).float()

        interseccion = torch.sum(input * target, dim = 2)
        union = torch.sum(input + target, dim = 2)

        numerador = torch.sum(w_c * interseccion, dim = 1)
        denominador = torch.sum(w_c * union, dim = 1)

        out = 1 - (2 * (numerador / (denominador + self.epsilon)))
        
        return out.mean()
