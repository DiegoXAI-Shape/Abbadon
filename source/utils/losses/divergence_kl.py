import torch
import torch.nn as nn
import torch.nn.functional as F

class DivergenceKL(nn.Module):
    def __init__(self, temperature=2.0, alpha=1.0):
        """
        temperature (T): Controla qué tanto se "suavizan" las probabilidades. 
                        T alto = más suave (el modelo aprende estructuras finas).
        alpha: Factor de peso para esta pérdida específica.
        """
        super(DivergenceKL, self).__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, student_logits, teacher_logits):
        """
        student_logits: [B, 1, H, W] - Salida cruda de tu red.
        teacher_logits: [B, 1, H, W] - Logits guardados de SAM.
        """
        T = self.temperature
        
        # 1. Estudiante: Convertimos el logit único a una distribución de 2 clases (Fondo vs Mascota)
        # Usamos logsigmoid para estabilidad numérica (log-probabilities)
        student_log_p1 = F.logsigmoid(student_logits / T)
        student_log_p0 = F.logsigmoid(-student_logits / T) 
        student_log_dist = torch.cat([student_log_p0, student_log_p1], dim=1) # [B, 2, H, W]
        
        # 2. Maestro SAM: Mismo proceso pero para obtener probabilidades puras
        teacher_p1 = torch.sigmoid(teacher_logits / T)
        teacher_p0 = 1.0 - teacher_p1
        teacher_dist = torch.cat([teacher_p0, teacher_p1], dim=1) # [B, 2, H, W]
        
        # 3. Cálculo de KL
        # IMPORTANTE: Cambiamos 'batchmean' a 'mean' para segmentación.
        # 'mean' divide por (B * 2 * H * W), manteniendo la pérdida en la misma escala que BCE y Dice.
        kl_loss = F.kl_div(student_log_dist, teacher_dist, reduction='mean')
        
        # 4. Escalamiento estándar de Distillation:
        # Se multiplica por T^2 para que la escala de los gradientes no cambie al variar T.
        # Se multiplica por alpha para el peso relativo.
        return self.alpha * kl_loss * (T ** 2)
