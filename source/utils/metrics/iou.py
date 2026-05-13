#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#                                                              &
#  Codificación de la métrica de precisión IoU                 $
#                                                              &
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


import torch
from typing import Any

def IoU_global(intersection:torch.Tensor, union:torch.Tensor, smooth = 1e-6):
    
    IoU_por_clase:torch.Tensor = (intersection + smooth) / (union + smooth)
    
    IoU_global:torch.Tensor = torch.mean(IoU_por_clase)

    return IoU_global, IoU_por_clase

def meanIoU(y_pred:torch.Tensor, y_true:torch.Tensor, num_classes:int, smooth:float = 1e-6):

    y_pred = torch.argmax(y_pred, dim = 1)

    IoU_clase:list[Any] = []

    for i in range(0, num_classes):
        pred_i = (y_pred == i).to(torch.float32)
        true_i = (y_true == i).to(torch.float32)

        interseccion = torch.sum(pred_i * true_i)
        union = torch.sum(pred_i) + torch.sum(true_i) - interseccion #PRINCIPIO DE INCLUSIÓN-EXCLUSIÓN

        iou = (interseccion + smooth) / (union + smooth)
        IoU_clase.append(iou)
    
    if len(IoU_clase) == 0:
        return torch.tensor(0.0, device=y_pred.device)
    
    return torch.mean(torch.stack(IoU_clase))

def get_intersections_and_unions(y_pred:torch.Tensor, y_true:torch.Tensor, class_id:int | Any):

    y_pred = (y_pred == class_id).to(torch.float32)
    y_true = (y_true == class_id).to(torch.float32)

    intersection = torch.sum(y_pred * y_true)
    union = torch.sum(y_pred) + torch.sum(y_true) - intersection

    return  (intersection, union)