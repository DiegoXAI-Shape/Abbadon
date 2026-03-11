import torch
from torch import nn


def conv3x3(in_channels, out_channels, stride = 1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride = stride, padding = 1, bias = False)


class BloqueResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample=None):
        super(BloqueResidual, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels, stride = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class UpSampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSampling, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias = False)
        self.bn = nn.BatchNorm2d(out_channels, affine=True)
        self.downsample_res = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        self.refine = BloqueResidual(out_channels * 2, out_channels, downsample=self.downsample_res)

    def forward(self, x, skip_connection):
        x = self.bn(self.up(x))
        x = torch.cat([x, skip_connection], dim = 1)
        x = self.refine(x)
        return x


class AttentionGates(nn.Module):
    def __init__(self, f_g, f_l, F_int):
        super(AttentionGates, self).__init__()

        self.w_x = nn.Sequential(
            nn.Conv2d(in_channels=f_l, out_channels=F_int, kernel_size = 3, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(num_features=F_int, affine=True)
        )

        self.w_g = nn.Sequential(
            nn.Conv2d(in_channels=f_g, out_channels=F_int, kernel_size = 1, stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d(num_features=F_int, affine = True)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(in_channels=F_int, out_channels = 1, kernel_size = 1, stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d(num_features=1, affine = True)
        )

        self.relu = torch.nn.ReLU(inplace = True)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, gate, skip_connection):
        g1 = self.w_g(gate)
        x1 = self.w_x(skip_connection)

        if g1.shape[2:] != x1.shape[2:]:
            g1 = torch.nn.functional.interpolate(g1, size = x1.shape[2:], mode='NEAREST')

        psi = self.relu(g1 + x1)
        pre_sigmoid = self.psi(psi)
        probs = self.sigmoid(pre_sigmoid)
        out = torch.nn.functional.interpolate(probs, size = skip_connection.shape[2:], mode = 'bilinear', align_corners=True)
        return out * skip_connection


class EncoderBlockT(nn.Module):
    def __init__(self):
        super(EncoderBlockT, self).__init__()
        self.MHA = nn.MultiheadAttention(embed_dim = 512, num_heads = 8, dropout=0.2, bias = True, batch_first=True)
        self.layernorm = nn.LayerNorm(normalized_shape=512)
        self.feedforward = nn.Sequential(
            nn.Linear(512, 2048),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(2048, 512),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        norm_x = self.layernorm(x)
        attn_out, _ = self.MHA(norm_x, norm_x, norm_x)
        x = x + attn_out

        norm_x = self.layernorm(x)
        mlp_out = self.feedforward(norm_x)
        x = x + mlp_out

        return x


#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#                                                              &
#  Codificación de un Transformer encoder para contexto global $
#                                                              &
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


class EncoderTrans(nn.Module):
    def __init__(self, in_embed_dim, _num_heads, _bias, _batch_first):

        """
        Attributes:
            - in_embed_dim: Dimensiones de entrada, es decir, los canales de la imagen en el bottleneck
            - _num_heads: Número de cabezas de atención múltiple
            - _bias: Booleano para saber si se aplica o no un sesgo al módulo MultiheadAttention
            - _batch_first: Booleano para decirle al módulo: 'Oye, el primer elemento del tensor es el batch'
        """

        super(EncoderTrans, self).__init__()
        self.mha = nn.MultiheadAttention(
                in_embed_dim,
                _num_heads, 
                dropout = 0.2, 
                bias = _bias, 
                batch_first=_batch_first
            )
        self.norm1 = nn.RMSNorm(in_embed_dim)
        self.norm2 = nn.RMSNorm(in_embed_dim)
        self.ffn= nn.Sequential(
            nn.Linear(in_features=in_embed_dim, out_features=in_embed_dim * 4),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=in_embed_dim * 4, out_features=in_embed_dim),
            nn.Dropout(0.2)
        )
    
    def forward(self, x):
        norm_x = self.norm1(x)
        out_mha, _ = self.mha(norm_x, norm_x, norm_x)
        x = out_mha + x

        norm_x = self.norm2(x)
        out_ffn = self.ffn(norm_x)
        x = x + out_ffn

        return x
