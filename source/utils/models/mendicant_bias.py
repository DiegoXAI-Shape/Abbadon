from torch import nn

try:
    from .blocks import BloqueResidual
except ImportError:
    from blocks import BloqueResidual


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
