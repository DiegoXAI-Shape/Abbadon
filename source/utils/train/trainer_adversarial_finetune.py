import os
import sys
import torch
import datetime
import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter                        
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, MofNCompleteColumn

dia = datetime.date.today()
current_dir = os.path.dirname(os.path.abspath(__file__))

# Subimos 2 niveles desde 'train' -> 'utils' -> 'source'
source_path = os.path.abspath(os.path.join(current_dir, "..", ".."))
if source_path not in sys.path:
    sys.path.append(source_path)
    
metrics_dir = os.path.join(source_path, "utils", "metrics")
if metrics_dir not in sys.path:
    sys.path.append(metrics_dir)

os.chdir(source_path)

from utils.models.daowa_maadV3_rc3 import Daowa_maadPrueba
from utils.losses.binary_dice_loss import BinaryDiceLoss
from utils.losses.boundary_loss import BoundaryLoss
from utils.train.distillation_dataset import CustomDistillationDataset, CustomAdversarialDataset, get_dual_dataloaders
from iou import get_intersections_and_unions, IoU_global

NUM_CLASSES = 1  # 0=Fondo, 1=Mascota (binario)

def train_model_adversarial(
    modelo, 
    loss_fn:list, 
    optimizador, 
    dataloaders:dict, 
    device_calc, 
    scheduler = None, 
    epochs:int = 50, 
    epsilon:float = 1e-8, 
    patience:int = 10
    ):

    os.makedirs('logs', exist_ok=True)
    os.makedirs('logs/tensorboard', exist_ok=True)
    os.makedirs(os.path.join(source_path, "Models"), exist_ok=True)
    writer = SummaryWriter(fr'logs/tensorboard/daowa_adversarial_finetune_{dia}')

    if modelo and isinstance(dataloaders, dict):
        
        modelo.to(device_calc)

        # Desempaquetado de Data e Instancias
        BCE_loss, Boundary_loss, Dice_loss = loss_fn
        
        train_dl_norm = dataloaders['train_norm']
        train_dl_adv = dataloaders['train_adv']
        val_dl_norm = dataloaders['val_norm']
        val_dl_adv = dataloaders['val_adv']
        
        # ─── Métricas para lógica de guardado mejorada ────────────────────
        best_adv_iou = 0.0           
        best_score = 0.0         
        epochs_sin_mejora = 0

        print(f"Iniciando FINE-TUNING ADVERSARIAL Paralelo ({NUM_CLASSES} clases: Fondo, Mascota)")

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            expand=True
        ) as progress:
            epoch_task = progress.add_task("[bold magenta]Progreso Total", total=epochs)

            for i in range(1, epochs + 1):
                progress.update(epoch_task, description=f"[bold magenta]Epoch {i}/{epochs}")
                
                # ─── Schedulling Dinámico de Pesos (Burn-in Lineal Propuesto) ───
                progreso = (i - 1) / (epochs - 1) if epochs > 1 else 1.0
                alpha = 1.0 - progreso
                
                # 1. BCE Loss: Empieza fuerte (1.0) y baja suavemente pero sin apagarse del todo (ej. 0.2)
                # Si cae a 0, la red pierde el contexto global de clasificación píxel a píxel.
                peso_bce = 1.0 * alpha + 0.2 * (1 - alpha)
                
                # 2. Dice y Boundary: Empiezan suaves (0.1) y terminan potentes (1.0) para afilar contornos duros
                peso_dice = 0.1 * alpha + 1.0 * (1 - alpha)
                peso_boundary = 0.1 * alpha + 1.0 * (1 - alpha)
                
                #Métricas Train
                train_loss_acc = 0.0
                t_corr_norm, t_tot_norm = 0, 0
                t_corr_adv, t_tot_adv = 0, 0

                modelo.train()
                
                # Vamos a iterar usando zip para someter la red a 1 Batch Normal y 1 Batch Adversarial de golpe
                max_batches = min(len(train_dl_norm), len(train_dl_adv))
                batch_task = progress.add_task(f"[cyan]Epoch {i} [Train]", total=max_batches)
                
                for (batch_norm, batch_adv) in zip(train_dl_norm, train_dl_adv):
                    
                    # Cargar Batch Normal
                    img_n, mask_n, sdf_n = batch_norm
                    img_n, mask_n, sdf_n = img_n.to(device_calc), mask_n.to(device_calc), sdf_n.to(device_calc)
                    
                    # Cargar Batch Adversarial
                    img_a, mask_a, sdf_a = batch_adv
                    img_a, mask_a, sdf_a = img_a.to(device_calc), mask_a.to(device_calc), sdf_a.to(device_calc)

                    optimizador.zero_grad()
                    
                    # --- FORWARD & BACKWARD NORMAL ---
                    with torch.amp.autocast_mode.autocast(device_type="cuda" if device_calc.type == 'cuda' else 'cpu', dtype=torch.bfloat16):
                        out_n = modelo(img_n) 
                        l_bce_n = BCE_loss(out_n, mask_n)
                        l_bnd_n = Boundary_loss(out_n, sdf_n)
                        l_dic_n = Dice_loss(out_n, mask_n) 
                        # Dividimos entre 2 geométricamente para promediar las gradientes finales equitativamente
                        loss_n = ((peso_bce * l_bce_n) + (peso_boundary * l_bnd_n) + (peso_dice * l_dic_n)) / 2.0
                        
                    # Libera el Grafo Híper Pesado de la VRAM INMEDIATAMENTE
                    loss_n.backward() 
                    
                    # --- FORWARD & BACKWARD ADVERSARIAL ---
                    with torch.amp.autocast_mode.autocast(device_type="cuda" if device_calc.type == 'cuda' else 'cpu', dtype=torch.bfloat16):
                        out_a = modelo(img_a) 
                        l_bce_a = BCE_loss(out_a, mask_a)
                        l_bnd_a = Boundary_loss(out_a, sdf_a)
                        l_dic_a = Dice_loss(out_a, mask_a) 
                        loss_a = ((peso_bce * l_bce_a) + (peso_boundary * l_bnd_a) + (peso_dice * l_dic_a)) / 2.0
                        
                    # Acumula matemáticamente a las gradientes anteriores (Y libera VRAM de nuevo)
                    loss_a.backward() 
                    
                    # Promedio Total solo para sumarlo a tu métrica en pantalla
                    total_loss = loss_n.detach() + loss_a.detach()

                    # Gradient Clipping & Norms (Estabilidad vital en adversarial)
                    total_norm = 0.0
                    for p in modelo.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.detach().data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** 0.5
                    torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)

                    optimizador.step()

                    train_loss_acc += total_loss.item()
                    
                    # Track de Accuracy de Píxeles independiente
                    preds_n = (out_n > 0.0).float()
                    preds_a = (out_a > 0.0).float()
                    
                    t_corr_norm += (preds_n == mask_n).sum().item()
                    t_tot_norm += mask_n.numel() 
                    t_corr_adv += (preds_a == mask_a).sum().item()
                    t_tot_adv += mask_a.numel() 
                    
                    progress.update(batch_task, advance=1, description=f"[cyan]Epoch {i} [Train] Loss: {total_loss.item():.4f}")

                progress.remove_task(batch_task)
                train_loss = train_loss_acc / max_batches
                tr_acc_n = (t_corr_norm / t_tot_norm) * 100
                tr_acc_a = (t_corr_adv / t_tot_adv) * 100
                gap_t = tr_acc_n - tr_acc_a

                # ─── VALIDACIÓN DUÁL ───────────────────────────────────────────
                val_loss_acc = 0.0
                v_corr_norm, v_tot_norm = 0, 0
                v_corr_adv, v_tot_adv = 0, 0

                int_n = torch.zeros(NUM_CLASSES, device=device_calc)
                uni_n = torch.zeros(NUM_CLASSES, device=device_calc)
                int_a = torch.zeros(NUM_CLASSES, device=device_calc)
                uni_a = torch.zeros(NUM_CLASSES, device=device_calc)

                val_task = progress.add_task(f"[yellow]Epoch {i} [Val]", total=max_batches)
                modelo.eval()
                
                with torch.no_grad():
                    for (batch_norm, batch_adv) in zip(val_dl_norm, val_dl_adv):
                        # Normal
                        in_n, m_n, sdf_n = batch_norm
                        in_n, m_n, sdf_n = in_n.to(device_calc), m_n.to(device_calc), sdf_n.to(device_calc)
                        
                        # Adversarial
                        in_a, m_a, sdf_a = batch_adv
                        in_a, m_a, sdf_a = in_a.to(device_calc), m_a.to(device_calc), sdf_a.to(device_calc)
                        
                        with torch.amp.autocast_mode.autocast(device_type="cuda" if device_calc.type == 'cuda' else 'cpu', dtype=torch.bfloat16):
                            pred_n = modelo(in_n)
                            l_bce_n = BCE_loss(pred_n, m_n)
                            l_bnd_n = Boundary_loss(pred_n, sdf_n)
                            l_dic_n = Dice_loss(pred_n, m_n)
                            loss_n = (peso_bce * l_bce_n) + (peso_boundary * l_bnd_n) + (peso_dice * l_dic_n)

                            pred_a = modelo(in_a)
                            l_bce_a = BCE_loss(pred_a, m_a)
                            l_bnd_a = Boundary_loss(pred_a, sdf_a)
                            l_dic_a = Dice_loss(pred_a, m_a)
                            loss_a = (peso_bce * l_bce_a) + (peso_boundary * l_bnd_a) + (peso_dice * l_dic_a)
                            
                            val_loss = (loss_n + loss_a) / 2.0

                        val_loss_acc += val_loss.item()
                        _y_pred_n = (pred_n > 0.0).float()
                        _y_pred_a = (pred_a > 0.0).float()

                        # IoU Independiente
                        for cls_id in range(NUM_CLASSES):
                            interN, unionN = get_intersections_and_unions(_y_pred_n, m_n, cls_id)
                            int_n[cls_id] += interN
                            uni_n[cls_id] += unionN
                            
                            interA, unionA = get_intersections_and_unions(_y_pred_a, m_a, cls_id)
                            int_a[cls_id] += interA
                            uni_a[cls_id] += unionA

                        v_corr_norm += (_y_pred_n == m_n).sum().item()
                        v_tot_norm += m_n.numel()
                        
                        v_corr_adv += (_y_pred_a == m_a).sum().item()
                        v_tot_adv += m_a.numel()

                        progress.update(val_task, advance=1)
                
                progress.remove_task(val_task)

                avg_val_loss = val_loss_acc / max_batches
                vl_acc_n = (v_corr_norm / v_tot_norm) * 100
                vl_acc_a = (v_corr_adv / v_tot_adv) * 100
                gap_v = vl_acc_n - vl_acc_a
                
                iou_norm_global, iou_norm_clases = IoU_global(int_n, uni_n)
                iou_adv_global, iou_adv_clases = IoU_global(int_a, uni_a)

                # ─── TensorBoard ──────────────────────────────────────────
                writer.add_scalar('Gradients/total_norm', total_norm, i)
                writer.add_scalars('Loss', { 'train': train_loss, 'val': avg_val_loss }, i)       
                
                # Trazado crítico de las Accuracies Compitiendo
                writer.add_scalars('Pixel Accuracy/Train', { 'Normal': tr_acc_n, 'Adversarial': tr_acc_a }, i)             
                writer.add_scalars('Pixel Accuracy/Validation', { 'Normal': vl_acc_n, 'Adversarial': vl_acc_a }, i)             
                
                writer.add_scalars('mIoU Global', { 'Normal': iou_norm_global.item(), 'Adversarial': iou_adv_global.item() }, i)            
                
                lrs = { f'Group_{j}': group['lr'] for j, group in enumerate(optimizador.param_groups) }
                writer.add_scalars('Learning Rate', lrs, i)
                writer.add_scalars('Loss Weights', { 'bce': peso_bce, 'boundary': peso_boundary, 'dice': peso_dice}, i) 

                if scheduler:
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(avg_val_loss)
                    else:
                        scheduler.step()

                # ─── LÓGICA DE GUARDADO ──────────────────────────
                # Usaremos el IoU_Adversarial como brújula principal para guardar, pero castigaremos si 
                # la red sufre Catastrophic Forgetting (si Normal IoU se desploma).
                score = (0.7 * iou_adv_global.item()) + (0.3 * iou_norm_global.item())

                guardado = False
                model_dir = os.path.join(source_path, "Models")

                if iou_adv_global.item() > best_adv_iou:
                    torch.save(modelo.state_dict(), os.path.join(model_dir, f'BestIoU_Adv_{dia}_{iou_adv_global.item():.2f}.pth'))
                    best_adv_iou = iou_adv_global.item()
                    epochs_sin_mejora = 0
                    guardado = True
                    progress.console.print(f"[bold green]✓ Mejor Defensa IoU Adversatial: {best_adv_iou:.4f}[/]")

                if score > best_score:
                    torch.save(modelo.state_dict(), os.path.join(model_dir, f'BestScore_Global_{dia}_{score:.2f}.pth'))
                    best_score = score
                    epochs_sin_mejora = 0
                    guardado = True
                    progress.console.print(f"[bold blue]✓ Mejor Core Resistencia: {score:.4f} [/]")

                if not guardado:
                    epochs_sin_mejora += 1

                torch.save(modelo.state_dict(), os.path.join(model_dir, f'Last_FineTuned_{dia}.pth'))

                if epochs_sin_mejora >= patience:
                    progress.console.print(f"\n[bold red]--- Parada anticipada: {patience} epochs sin mejora ---\n")
                    break

                progress.console.print(f"[bold yellow]Epoch {i}:[/] Train Loss = [bold]{train_loss:.4f}[/]; Val Loss = [bold]{avg_val_loss:.4f}[/]")
                progress.console.print(f"   [white]-> Accs ->[/] Val_N: {vl_acc_n:.2f}% | Val_A: {vl_acc_a:.2f}% | [magenta]GAP_Val: {gap_v:.2f}%[/] | [magenta]GAP_Train: {gap_t:.2f}%[/]")
                progress.console.print(f"   [white]-> IoUs ->[/] M_Norm: {iou_norm_global.item():.4f} | M_Adv: {iou_adv_global.item():.4f}")

                progress.advance(epoch_task)

    writer.close()
    return "Entrenamiento completado"

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando aceleración en: {device}")
    
    # 1. Configurar directorios de Datos
    base_data_path = os.path.join(source_path, "data", "oxford")
    imgs_dir = os.path.join(base_data_path, "images")
    human_masks_dir = os.path.join(base_data_path, "masks")
    sam_masks_dir = os.path.join(base_data_path, "masks_SAM") # Usado solo para no romper DistillationDataset
    csv_labels_path = os.path.join(source_path, "labels", "dataset_generated.csv")
    
    ade20k_dir = os.path.join(source_path, "data", "ADEChallengeData2016")
    ade20k_csv = os.path.join(source_path, "labels", "ade20k_adversarial_train.csv")
    
    # 2. Cargar Datasets Duales Puros (Sin pseudo_gold) encapsulados modularmente
    dataloaders = get_dual_dataloaders(
        base_data_path=base_data_path,
        csv_labels_path=csv_labels_path,
        ade20k_dir=ade20k_dir,
        ade20k_csv=ade20k_csv,
        batch_size=4,
        img_size=(384, 384)
    )
    
    # 3. Preparar Entorno de Entrenamiento
    modelo = Daowa_maadPrueba(in_channels=3, out_channels=1) 
    
    # Cargando PRE-TRAINED Pesos (El modelo que APRENDIO de SAM previamente)
    # NOTA: Descomenta esto y propociona la ruta real si tu inicio no es desde cero.
    # modelo.load_state_dict(torch.load(os.path.join(source_path, "Models", "tu_modelo_sam.pth"), map_location=device))
    
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=1e-4, weight_decay=1e-5)
    
    loss_fn = [
        torch.nn.BCEWithLogitsLoss(),
        BoundaryLoss(),       # Tuya custom
        BinaryDiceLoss(),     # Tuya custom
    ]
    
    print("Iniciando Escenario de Defensa Adversarial Dual...")
    train_model(
        modelo=modelo,
        loss_fn=loss_fn,
        optimizador=optimizador,
        dataloaders=dataloaders,
        device_calc=device,
        epochs=50,
        patience=12
    )

if __name__ == "__main__":
    main()
