import torch
from torch import nn

try:
    from .blocks import BloqueResidual
except ImportError:
    from blocks import BloqueResidual

class AttentionGate(nn.Module):
    """
    Oráculo de Atención (Mendicant Bias): 
    Este módulo recibe la Imagen RGB y la Máscara generada por Daowa-maad.
    Aprende a generar un Mapa de Corrección (Correction Map) para sumar o restar confianza a la máscara.
    """
    def __init__(self, in_channels=4): # 3 (RGB) + 1 (Mask) = 4
        super(AttentionGate, self).__init__()
        
        # Bloque ligero para no sobreajustar ni agregar mucho costo computacional
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
            # Capa final que extrae 1 solo canal (el mapa de corrección)
            nn.Conv2d(16, 1, kernel_size=3, padding=1, bias=True)
        )
        
        # ¡Defensa A: Inicialización Cero (SkipInit)!
        # Ponemos los pesos y sesgos de la ÚLTIMA capa convolucional exactamente en 0.
        # Así, al inicio del entrenamiento, el bloque escupe puros 0s, por lo que 
        # Mendicant Bias empieza confiando al 100% en la máscara original de Daowa.
        nn.init.zeros_(self.conv_block[-1].weight)
        nn.init.zeros_(self.conv_block[-1].bias)
        
    def forward(self, rgb, mask):
        # 1. Concatenar (Batch, 4 canales, Height, Width)
        x = torch.cat([rgb, mask], dim=1)
        
        # 2. Extraer corrección pura (arrancará escupiendo ceros absolutos)
        correction_raw = self.conv_block(x)
        
        # 3. Limitar a un rango matemático de -1 a +1 (Restar o Sumar píxeles)
        correction_map = torch.tanh(correction_raw)
        
        # 4. Suma Residual (La magia ocurre aquí)
        refined_mask = torch.clamp(mask + correction_map, min=0.0, max=1.0)
        
        # Retornamos la máscara refinada (para que ConvNeXt la use)
        # Y el mapa de corrección (para que tu código de entrenamiento lo penalice con L2 si abusa de él)
        return refined_mask, correction_map

import timm

class MendicantBias_ConvNeXt(nn.Module):
    """
    Mendicant Bias (Clasificador Final Perro/Gato).
    Usa ConvNeXtV2 Atto modificado para aceptar 4 canales (3 RGB + 1 Máscara de Atención).
    Incluye el AttentionGate internamente para manejar la lógica limpia.
    """
    def __init__(self, pretrained=True):
        super(MendicantBias_ConvNeXt, self).__init__()
        
        # 1. Instanciamos nuestro Gate mágico
        self.attention_gate = AttentionGate(in_channels=4)
        
        # 2. Cargamos ConvNeXtV2 Atto (el modelo base de clasificación)
        # num_classes=2 (0=Perro, 1=Gato, o viceversa según tu dataset)
        self.classifier = timm.create_model('convnextv2_atto', pretrained=pretrained, num_classes=2)
        
        # 3. Adaptación del Stem (la primera capa) para que trague 4 canales en vez de 3
        # El stem de convnextv2 es un Sequential, donde el índice 0 es la Convolución inicial.
        old_conv = self.classifier.stem[0]
        
        new_conv = nn.Conv2d(
            in_channels=4, 
            out_channels=old_conv.out_channels, 
            kernel_size=old_conv.kernel_size, 
            stride=old_conv.stride, 
            padding=old_conv.padding, 
            bias=(old_conv.bias is not None)
        )
        
        # Copiamos los pesos preentrenados de ImageNet para los canales RGB (los primeros 3)
        new_conv.weight.data[:, :3, :, :] = old_conv.weight.data
        
        # ¡CORRECCIÓN CRÍTICA! (La Trampa del Cero)
        # Si inicializamos el 4to canal en 0.0 absoluto, ConvNeXt (por sus LayerNorms)
        # se queda atascado en un punto de silla (Saddle Point) y no aprende, dando 50% de accuracy.
        # Solución: Inicializar el 4to canal con el promedio de los pesos RGB.
        # Así la red percibe la máscara como si fuera una "imagen en escala de grises" desde el paso 1.
        rgb_mean_weights = old_conv.weight.data.mean(dim=1, keepdim=True)
        new_conv.weight.data[:, 3:4, :, :] = rgb_mean_weights
        
        if old_conv.bias is not None:
            new_conv.bias.data = old_conv.bias.data
            
        # Reemplazamos la capa vieja por la nueva
        self.classifier.stem[0] = new_conv

        # Variable para almacenar el mapa de corrección actual y poder penalizarlo en el Loss
        self._last_correction_map = None
        
    def forward(self, rgb, daowa_mask, drop_rgb_prob=0.0):
        """
        rgb: Tensor [Batch, 3, H, W]
        daowa_mask: Tensor [Batch, 1, H, W] (Salida Sigmoide cruda del Oráculo)
        drop_rgb_prob: Probabilidad de apagar los colores (solo durante entrenamiento)
        """
        # 1. El Gate refina la máscara usando el RGB y la máscara original
        refined_mask, self._last_correction_map = self.attention_gate(rgb, daowa_mask)
        
        # 2. Defensa C (La Prueba de Fuego): Dropout de Canal RGB
        # Si estamos entrenando y cae en la probabilidad (ej. 15%), apagamos los colores.
        if self.training and drop_rgb_prob > 0.0:
            if torch.rand(1).item() < drop_rgb_prob:
                # Multiplicamos el RGB por 0, forzando a ConvNeXt a clasificar SOLO usando la forma de la máscara
                rgb = rgb * 0.0
                
        # 3. Ensamblamos el tensor de 4 canales
        x_4d = torch.cat([rgb, refined_mask], dim=1)
        
        # 4. Clasificación Final
        logits = self.classifier(x_4d)
        
        return logits
        
    def get_gate_regularization_loss(self):
        """
        Devuelve el L2 (Norma) del mapa de corrección para sumarlo al Loss final.
        Esto penaliza a la red por desobedecer al Oráculo de Daowa-maad.
        """
        if self._last_correction_map is None:
            return 0.0
        # Promedio del cuadrado de los valores de corrección (MSE vs 0)
        return torch.mean(self._last_correction_map ** 2)




class Mendicant_Biasv3(nn.Module):
    def __init__(self):
        super(Mendicant_Biasv3, self).__init__()
        # Conv = (input - kernel + 2 * padding) / stride

        self.in_channels = 64

        #Primer reducción agresiva de ResNet
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=(2, 2), bias=False),
            nn.BatchNorm2d(num_features=64, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)   
        )

        #Bloques residuales
        self.layer1 = self._make_layer(out_channels=64, blocks = 2, stride = 1)

        self.layer2 = self._make_layer(out_channels=128, blocks = 2, stride = 2)
        
        self.layer3 = self._make_layer(out_channels=256, blocks = 2, stride = 2)

        self.layer4 = self._make_layer(out_channels=512, blocks = 2, stride = 2)

        #Promedio
        self.avg = nn.AdaptiveAvgPool2d((1, 1))

        #Capas de clasificación
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=256), 
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.35),
            nn.Linear(in_features=256, out_features=2),
        )

    
    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None

        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )

        layers = []

        layers.append(BloqueResidual(self.in_channels, out_channels, stride, downsample))

        self.in_channels = out_channels

        for _ in range(1, blocks):
            layers.append(BloqueResidual(self.in_channels, out_channels))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.stem(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avg(x)
        x = self.classifier(x)
        return x
