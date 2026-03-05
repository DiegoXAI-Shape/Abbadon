import torch
from torch import nn

try:
    from .blocks import BloqueResidual, UpSampling, AttentionGates, EncoderBlockT
except ImportError:
    from blocks import BloqueResidual, UpSampling, AttentionGates, EncoderBlockT


class Daowa_maad(nn.Module):
    def __init__(self, num_clases = 3):
        super(Daowa_maad, self).__init__()

        self.in_channels = 64
        self.downsample = None
        
        # > Bajadaaaaaaaaaaaaaaaaaaa

        #No es tan agresiva la entrada a la ResU-Net
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride = 1, padding = 1, bias = False), #3, 64, 128, 128 -> 64, ...
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(inplace=True)
        )

        self.layer1 = self._hacer_capaXD(64, 3, 1)

        self.layer2 = self._hacer_capaXD(128, 3, 2)

        self.layer3 = self._hacer_capaXD(256, 3, 2)

        self.layer4 = self._hacer_capaXD(512, 3, 2)
        
        self.up1 = UpSampling(512, 256)

        self.attgate1 = AttentionGates(512, 256, 256)
        
        self.up2 = UpSampling(256, 128)

        self.attgate2 = AttentionGates(256, 128, 128)

        self.up3 = UpSampling(128, 64)
        
        self.attgate3 = AttentionGates(128, 64, 64)

        self.head = nn.Conv2d(64, num_clases, kernel_size=1)
            
    def _hacer_capaXD(self, out_channels, bloques, stride = 1):

        #El downsamples broooooooo
        if (stride != 1) or (self.in_channels != out_channels):
            self.downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride = stride, bias = False),
                nn.BatchNorm2d(out_channels, affine=True)
            )
        
        layers = []

        layers.append(BloqueResidual(self.in_channels, out_channels, stride, self.downsample))

        self.in_channels = out_channels

        for _ in range(1, bloques):
            layers.append(BloqueResidual(self.in_channels, out_channels, stride = 1))
        
        return nn.Sequential(*layers)
         
    def forward(self, x):
        x0 = self.stem(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1) 
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x3_filtrada = self.attgate1(x4, x3)
        x = self.up1(x4, x3_filtrada)
        
        x2_filtrada = self.attgate2(x, x2)
        x = self.up2(x, x2_filtrada)

        x1_filtrada = self.attgate3(x, x1)
        x = self.up3(x, x1_filtrada)

        logits = self.head(x)
        return logits


class TransformerDaowa_maad(nn.Module):
    def __init__(self, num_clases = 3):
        super(TransformerDaowa_maad, self).__init__()

        self.in_channels = 64
        self.downsample = None
        
        # > Bajadaaaaaaaaaaaaaaaaaaa

        #No es tan agresiva la entrada a la ResU-Net
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride = 1, padding = 1, bias = False), #3, 64, 128, 128 -> 64, ...
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(inplace=True)
        )

        self.layer1 = self._hacer_capaXD(64, 3, 1)

        self.layer2 = self._hacer_capaXD(128, 3, 2)

        self.layer3 = self._hacer_capaXD(256, 3, 2)

        self.layer4 = self._hacer_capaXD(512, 3, 2)
        
        self.up1 = UpSampling(512, 256)

        self.attgate1 = AttentionGates(512, 256, 256)
        
        self.up2 = UpSampling(256, 128)

        self.attgate2 = AttentionGates(256, 128, 128)

        self.up3 = UpSampling(128, 64)
        
        self.attgate3 = AttentionGates(128, 64, 64)
        
        self.Transformer = self.MakeLayerTransformer(6)

        self.patch_size = 24 * 24

        self.pos_embedding = nn.Parameter(torch.rand(1, self.patch_size, 512))

        self.head = nn.Conv2d(64, num_clases, kernel_size=1)

    def MakeLayerTransformer(self, n:int):
        if isinstance(n, int):
            transforms_sublayers = []
            for i in range(1, n + 1):
                transforms_sublayers.append(EncoderBlockT())
        
        return nn.Sequential(*transforms_sublayers)
            
    def _hacer_capaXD(self, out_channels, bloques, stride = 1):

        #El downsamples broooooooo
        if (stride != 1) or (self.in_channels != out_channels):
            self.downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride = stride, bias = False),
                nn.BatchNorm2d(out_channels, affine=True)
            )
        
        layers = []

        layers.append(BloqueResidual(self.in_channels, out_channels, stride, self.downsample))

        self.in_channels = out_channels

        for _ in range(1, bloques):
            layers.append(BloqueResidual(self.in_channels, out_channels, stride = 1))
        
        return nn.Sequential(*layers)
         
    def forward(self, x):
        x0 = self.stem(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1) 
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        #Primero para empezar, los transformers coloquialmente te piden la forma [Batch, Sequence / Patches, Embedding Dimension]
        #Pero aquí está el detalle dirian en mi rancho HJAAASHFAH, la red me da: [Batch, Filtros, H, W], por lo que debo reducir la dimensionalidad de la posición 2 en adelante
        x = x4.flatten(2)

        #Pero esto aquí me daría: [Batch, Filtros, El aplanao'], pero yo ocupo [Batch, sequence / el aplanao', Embeddings o lo que es mejor dicho filtros]
        x = x.permute(0, 2, 1)
        x = x + self.pos_embedding

        x_transformer = self.Transformer(x)

        #Una vez siendo procesado por el transformer, ocupamos que la red nos devuelva lo 2D para ir a la subida de la U Net
        x_transformer = x_transformer.permute(0, 2, 1)
        x_bottleneck = x_transformer.reshape(x_transformer.shape[0], 512, 24, 24)

        x3_filtrada = self.attgate1(x_bottleneck, x3)
        x = self.up1(x_bottleneck, x3_filtrada)
        
        x2_filtrada = self.attgate2(x, x2)
        x = self.up2(x, x2_filtrada)

        x1_filtrada = self.attgate3(x, x1)
        x = self.up3(x, x1_filtrada)

        logits = self.head(x)
        return logits
