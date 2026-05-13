"""
adversarial_loss.py
===================
Loss combinada para adversarial fine-tuning de segmentación binaria.

Combina:
    - BinaryDiceLoss  : penaliza superposición incorrecta (FP y FN)
    - BoundaryLoss    : penaliza predicciones lejos del borde real (usa SDF)

En imágenes negativas (máscara = todo ceros):
    - El SDF precalculado es todo UNOS  → BoundaryLoss castiga CADA píxel predicho
    - El Dice penaliza automáticamente toda predicción positiva (intersection → 0)

No hay BCEWithLogitsLoss ni KL Divergence en este módulo.
"""

import torch
import torch.nn as nn


class AdversarialSegLoss(nn.Module):
    """
    Loss combinada Dice + Boundary para adversarial fine-tuning.

    Args:
        w_dice     : Peso de la BinaryDiceLoss  (default 1.0)
        w_boundary : Peso de la BoundaryLoss    (default 1.0)
        smooth     : Suavizado numérico del Dice (default 1e-5)

    Uso:
        loss_fn = AdversarialSegLoss(w_dice=1.0, w_boundary=1.0)
        loss = loss_fn(logits, mask_gt, sdf_target)
    """

    def __init__(self, w_dice: float = 1.0, w_boundary: float = 1.0, smooth: float = 1e-5):
        super().__init__()
        self.w_dice     = w_dice
        self.w_boundary = w_boundary
        self.smooth     = smooth

    def _dice_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Dice Loss binario.
        logits  : [B, 1, H, W] — salida cruda de la red (sin sigmoide)
        targets : [B, 1, H, W] — máscara ground truth (0.0 ó 1.0)

        Para imágenes negativas:
            targets es todo ceros → intersection = 0
            → dice = smooth / (probs.sum() + smooth) → cercano a 0
            → DiceLoss = 1 - dice → cercano a 1.0  (máximo castigo)
        """
        probs      = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        tgt_flat   = targets.view(-1)

        intersection = (probs_flat * tgt_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs_flat.sum() + tgt_flat.sum() + self.smooth
        )
        return 1.0 - dice

    def _boundary_loss(self, logits: torch.Tensor, sdf_target: torch.Tensor) -> torch.Tensor:
        """
        Boundary Loss (multiplicación punto a punto).
        logits     : [B, 1, H, W]
        sdf_target : [B, 1, H, W] — SDF precalculado, normalizado [-1, 1]
                     Para negativos, sdf_target = todo UNOS → castigo total.

        La red es castigada proporcionalmente a cuán lejos del borde real predice.
        """
        probs = torch.sigmoid(logits)
        return torch.mean(probs * sdf_target)

    def forward(
        self,
        logits: torch.Tensor,
        mask_gt: torch.Tensor,
        sdf_target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            logits     : [B, 1, H, W]
            mask_gt    : [B, 1, H, W] float32 (0 ó 1)
            sdf_target : [B, 1, H, W] float32

        Returns:
            total_loss : escalar
            desglose   : dict con 'dice' y 'boundary' para logging en TensorBoard
        """
        l_dice     = self._dice_loss(logits, mask_gt)
        l_boundary = self._boundary_loss(logits, sdf_target)

        total = self.w_dice * l_dice + self.w_boundary * l_boundary

        return total, {"dice": l_dice.detach(), "boundary": l_boundary.detach()}


class BurnInAdversarialLoss(AdversarialSegLoss):
    """
    Variante con Burn-In dinámico de pesos.

    Durante las primeras épocas el Dice domina (para no destruir lo aprendido).
    Conforme avanza el entrenamiento, el Boundary Loss gana importancia para
    afinar contornos y penalizar predicciones espurias en imágenes negativas.

    w_dice     : inicio → 1.0, fin → 0.5
    w_boundary : inicio → 0.1, fin → 1.0
    """

    def step_epoch(self, epoch: int, total_epochs: int):
        """Llamar al inicio de cada época para actualizar los pesos dinámicamente."""
        progreso       = (epoch - 1) / max(total_epochs - 1, 1)
        self.w_dice     = 1.0 * (1 - progreso) + 0.5 * progreso
        self.w_boundary = 0.1 * (1 - progreso) + 1.0 * progreso
