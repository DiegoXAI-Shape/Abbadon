import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def evaluate_on_unseen_data(model, dataloader, device, save_dir="source/logs/evaluations"):
    """
    Evalúa el modelo en un dataset jamás visto (como dev_dataloader o test_dataloader).
    Calcula métricas binarias estrictas y genera imágenes comparativas para el 'ojo humano'.
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    total_iou = 0.0
    total_dice = 0.0
    num_batches = 0
    
    print(f"\n[Evaluación] Iniciando prueba con datos no vistos...")
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Evaluando")):
            # Algunas veces el batch trae (image, mask_human, mask_SAM) u otras variables
            # Intentaremos desempaquetar las primeras dos que son las más importantes
            image = batch[0].to(device)
            mask_human = batch[1].to(device) # Shape esperado: (B, 1, H, W)
            
            # Predicción pura
            with torch.amp.autocast_mode.autocast(device_type="cuda" if device.type == 'cuda' else 'cpu', dtype=torch.bfloat16):
                logits = model(image)
                
            # Convertir logits a binario (1 o 0)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            # Asegurar que máscaras tengan misma dimensión
            if mask_human.ndim == 3: # (B, H, W)
                mask_human = mask_human.unsqueeze(1)
                
            # Cálculo de Métricas Binarias (Perro = 1, Fondo = 0)
            intersection = torch.sum(preds * mask_human)
            union = torch.sum(preds) + torch.sum(mask_human) - intersection
            
            batch_iou = (intersection + 1e-6) / (union + 1e-6)
            batch_dice = (2. * intersection + 1e-6) / (torch.sum(preds) + torch.sum(mask_human) + 1e-6)
            
            total_iou += batch_iou.item()
            total_dice += batch_dice.item()
            num_batches += 1
            
            # Guardamos lotes aleatorios/primeros de prueba para validación visual "Ojo Humano"
            if i < 4:
                _save_visual_comparison(image, mask_human, preds, i, save_dir)
                
    # Resultados finales
    mean_iou = total_iou / num_batches
    mean_dice = total_dice / num_batches
    
    print("-" * 50)
    print(f"RESULTADOS FINALES EN DATOS NUNCA VISTOS:")
    print(f"👉 IoU (Intersección sobre Unión): {mean_iou * 100:.2f}%")
    print(f"👉 F1 / Dice Score:              {mean_dice * 100:.2f}%")
    print("-" * 50)
    print(f"Visualizaciones guardadas en: {save_dir}")
    
    return mean_iou, mean_dice

def _save_visual_comparison(image_tensor, target_tensor, pred_tensor, batch_idx, save_dir):
    """
    Función interna para graficar y guardar: Imagen Normal | Ground Truth | Predicción Modelo.
    """
    # Tomamos la primera imagen del batch
    img = image_tensor[0].cpu() # (C, H, W)
    target = target_tensor[0].cpu().squeeze() # (H, W)
    pred = pred_tensor[0].cpu().squeeze() # (H, W)
    
    # Para visualización gráfica cruda simplemente hacemos un min-max en la imagen
    # En caso de que haya estado normalizada
    img_show = img.clone()
    img_show = (img_show - img_show.min()) / (img_show.max() - img_show.min() + 1e-6)
    img_show = img_show.permute(1, 2, 0).numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_show)
    axes[0].set_title('Imagen de Entrada')
    axes[0].axis('off')
    
    axes[1].imshow(target.numpy(), cmap='gray')
    axes[1].set_title('Ground Truth (Lo que debe ser)')
    axes[1].axis('off')
    
    axes[2].imshow(pred.numpy(), cmap='gray')
    axes[2].set_title('Predicción (Evaluando Boundary)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'eval_visual_ojo_humano_lote_{batch_idx}.png'))
    plt.close()
