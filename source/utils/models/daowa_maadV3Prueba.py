#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#
# Arquitectura Daowa_maad sin transformers
# después se le agregará el transformer
# para comparar el rendimiento
#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

import torch
from torch import nn
import torch.nn.functional as F

try:
    from .blocks import UpSampling, AttentionGates, EncoderTrans
except ImportError:
    from blocks import UpSampling, AttentionGates, EncoderTrans

import timm

class Daowa_maadPrueba(nn.Module):
    def __init__(self, num_clases = 3):
        super(Daowa_maadPrueba, self).__init__()

        self.encoder = timm.create_model(
            'convnextv2_tiny.fcmae_ft_in22k_in1k',
            pretrained=True,
            features_only=True,
            out_indices=[0, 1, 2, 3]
        ) # Salida de 768 filtros o canales

        old_conv = self.encoder.stem_0
        new_conv = nn.Conv2d(4, old_conv.out_channels, kernel_size=old_conv.kernel_size, stride=old_conv.stride, padding=old_conv.padding, bias=old_conv.bias is not None)

        with torch.no_grad():
            new_conv.weight.data[:, :3, :, :] = old_conv.weight.data
            new_conv.weight.data[:, 3:, :, :] = old_conv.weight.data[:, 0:1, :, :]

        self.encoder.stem_0 = new_conv


        self.att_gate1 = AttentionGates(768, 384, 384)
        self.att_gate2 = AttentionGates(384, 192, 192)
        self.att_gate3 = AttentionGates(192, 96, 96)

        self.up1 = UpSampling(768, 384)
        self.up2 = UpSampling(384, 192)
        self.up3 = UpSampling(192, 96)

        self.head = nn.Conv2d(96, num_clases, kernel_size=1)


    def forward(self, x):
        input_size = x.shape[2:]

        x1, x2, x3, x4 = self.encoder(x)
        # x1=[96,48,48], x2=[192,24,24], x3=[384,12,12], x4=[768,6,6]

        x3_filtrada = self.att_gate1(x4, x3)
        x = self.up1(x4, x3_filtrada)
        
        x2_filtrada = self.att_gate2(x, x2)
        x = self.up2(x, x2_filtrada)

        x1_filtrada = self.att_gate3(x, x1)
        x = self.up3(x, x1_filtrada)

        logits = self.head(x)
        logits = F.interpolate(logits, size=input_size, mode='bilinear', align_corners=True)
        return logits

class Daowa_maadPrueba2(nn.Module):
    def __init__(self, num_clases):
        super(Daowa_maadPrueba2, self).__init__()

        self.encoder = timm.create_model(
            'convnextv2_tiny.fcmae_ft_in22k_in1k',
            pretrained=True,
            features_only=True,
            out_indices=[0, 1, 2, 3]
        ) # Salida de 768 filtros o canales

        old_conv = self.encoder.stem_0
        new_conv = nn.Conv2d(4, old_conv.out_channels, kernel_size=old_conv.kernel_size, stride=old_conv.stride, padding=old_conv.padding, bias=old_conv.bias is not None)

        with torch.no_grad():
            new_conv.weight.data[:, :3, :, :] = old_conv.weight.data
            new_conv.weight.data[:, 3:, :, :] = old_conv.weight.data[:, 0:1, :, :]

        self.encoder.stem_0 = new_conv

        self.transformer_encoder = EncoderTrans(in_embed_dim=768, _num_heads=12, _bias=True, _batch_first=True)
        self.pos_embedding = nn.Parameter(torch.randn(1, 36, 768))

        self.att_gate1 = AttentionGates(768, 384, 384)
        self.att_gate2 = AttentionGates(384, 192, 192)
        self.att_gate3 = AttentionGates(192, 96, 96)

        self.up1 = UpSampling(768, 384)
        self.up2 = UpSampling(384, 192)
        self.up3 = UpSampling(192, 96)

        self.head = nn.Conv2d(96, num_clases, kernel_size=1)
    
    def interpolate_pos_embed(self, x):
        B, N, C = x.shape
        origin_N = self.pos_embedding.shape[1]
        if N == origin_N:
            return self.pos_embedding
        
        orig_size = int(origin_N ** 0.5)
        new_size = int(N ** 0.5)
        
        pos_embed = self.pos_embedding.view(1, orig_size, orig_size, C).permute(0, 3, 1, 2) # [B, 6, 6, C] -> [B, C, 6, 6]
        pos_embed = F.interpolate(pos_embed, size=(new_size, new_size), mode='bicubic', align_corners=False) # [B, C, new_size, new_size]
        pos_embed = pos_embed.flatten(2).permute(0, 2, 1) # [B, new_size*new_size, C]
        return pos_embed
        

    def forward(self, x):
        input_size = x.shape[2:]

        x1, x2, x3, x4 = self.encoder(x)
        # x1=[96,48,48], x2=[192,24,24], x3=[384,12,12], x4=[768,6,6]

        B, C, H, W = x4.shape

        x4 = x4.view(B, C, -1).permute(0, 2, 1)
        x4 = self.transformer_encoder(x4 + self.interpolate_pos_embed(x4))
        x4 = x4.permute(0, 2, 1).view(B, C, H, W)

        x3_filtrada = self.att_gate1(x4, x3)
        x = self.up1(x4, x3_filtrada)
        
        x2_filtrada = self.att_gate2(x, x2)
        x = self.up2(x, x2_filtrada)

        x1_filtrada = self.att_gate3(x, x1)
        x = self.up3(x, x1_filtrada)

        logits = self.head(x)
        logits = F.interpolate(logits, size=input_size, mode='bilinear', align_corners=True)
        return logits
