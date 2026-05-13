import torch
from torch.utils.tensorboard import SummaryWriter
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, MofNCompleteColumn
import pandas as pd
import os
import datetime
import sys

current_dir = os.path.abspath(os.path.dirname(__file__))
metrics_dir = os.path.join(current_dir, "..", "metrics")

sys.path.append(metrics_dir)

from iou import get_intersections_and_unions, IoU_global

def train_model(modelo, loss_fn, optimizador, dataloaders:list, device_calc, scheduler = None, epochs:int = 20, epsilon = 1e-8, patience:int = 5 ):

    dia = datetime.date.today()
    writer = SummaryWriter(fr'logs/tensorboard/daowa_maad_v3_{dia}')

    if modelo and isinstance(dataloaders, list):
        
        modelo.to(device_calc)

        #Desempaquetado
        cross_fn, dice_fn = loss_fn
        train_dl, val_dl, _ = dataloaders
        
        #Métricas
        best_iou = 0.0
        best_val_loss = float('inf')
        epochs_sin_mejora = 0

        print("Iniciando entrenamiento")

        # Usamos rich para un progreso más estético y detallado
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            expand=True
        ) as progress:
            epoch_task = progress.add_task("[green]Progreso Total", total=epochs)

            for i in range(1, epochs + 1):
                progress.update(epoch_task, description=f"[green]Epoch {i}/{epochs}")
                
                if 0 < i <= 5:
                    peso_dice = 0
                    peso_cross = 1

                    encoder_params = modelo.encoder.parameters()
                    for param in encoder_params:
                        param.requires_grad = False

                elif 5 < i <= 10: 
                    peso_dice = 0.5
                    peso_cross = 0.5

                    encoder_params = modelo.encoder.parameters()
                    for param in encoder_params:
                        param.requires_grad = True
                else:
                    peso_dice = 0.8
                    peso_cross = 0.2
                
                #Métricas
                train_loss_acc = 0.0
                train_correct_pixels = 0
                train_total_pixels = 0
                intersections = torch.zeros(3, device=device_calc)
                unions = torch.zeros(3, device=device_calc)

                #Modelo en modo de entrenamiento wachin
                modelo.train()
                
                # Barra para los Batches (Train)
                batch_task = progress.add_task(f"[cyan]Epoch {i} [Train]", total=len(train_dl))
                
                for image, mask in train_dl:
                    image, mask = image.to(device_calc), mask.to(device_calc)

                    optimizador.zero_grad()
                    with torch.amp.autocast_mode.autocast(device_type="cuda", dtype=torch.bfloat16):
                        output = modelo(image) 
                    
                        loss_cross = cross_fn(output, mask)
                        loss_dice = dice_fn(output, mask) 
                        total_loss = (peso_cross * loss_cross + peso_dice * loss_dice)
                    
                    total_loss.backward()

                    total_norm = 0.0
                    for p in modelo.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.detach().data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** 0.5

                    optimizador.step()

                    current_loss = total_loss.item()
                    train_loss_acc += current_loss
                    
                    # Precisión de Píxeles (Train)
                    _, preds = torch.max(output, 1) 
                    train_correct_pixels += (preds == mask).sum().item()
                    train_total_pixels += mask.numel() 
                    
                    progress.update(batch_task, advance=1, description=f"[cyan]Epoch {i} [Train] Loss: {current_loss:.4f}")

                progress.remove_task(batch_task)
                train_loss = train_loss_acc / len(train_dl)
                train_acc = (train_correct_pixels / train_total_pixels) * 100

                #Ya nomás pa' evaluar
                actual_valLoss = 0.0
                val_correct_pixels = 0
                val_total_pixels = 0

                modelo.eval()
                with torch.no_grad():
                    for image, mask in val_dl:
                        image = image.to(device_calc)
                        mask = mask.to(device_calc)
                        
                        predict = modelo(image)
                        _y_predicts = torch.argmax(predict, dim = 1)

                        inter_0, union_0 = get_intersections_and_unions(_y_predicts, mask, 0)
                        inter_1, union_1 = get_intersections_and_unions(_y_predicts, mask, 1)
                        inter_2, union_2 = get_intersections_and_unions(_y_predicts, mask, 2)

                        inters = torch.stack([inter_0, inter_1, inter_2])
                        unis = torch.stack([union_0, union_1, union_2])

                        intersections += inters
                        unions += unis

                        cross_loss = cross_fn(predict, mask)
                        dice_loss = dice_fn(predict, mask)
                        val_loss = (cross_loss + dice_loss)

                        actual_valLoss += val_loss.item()
                        _, predicts = torch.max(predict, 1) 
                        val_correct_pixels += (predicts == mask).sum().item()
                        val_total_pixels += mask.numel()

                avg_val_loss = actual_valLoss / len(val_dl)
                val_acc = (val_correct_pixels / val_total_pixels) * 100
                mIoU, IoU_por_clase = IoU_global(intersections, unions)
                IoU_por_clase = IoU_por_clase.cpu().numpy().tolist()
                
                all_weights = torch.cat([p.view(-1) for p in modelo.parameters() if p.requires_grad]).detach()

                mean = torch.mean(all_weights)
                std = torch.std(all_weights)
                kurtosis = (torch.mean(((all_weights - mean) / (std + epsilon)) ** 4)) - 3
                skewness = torch.mean(((all_weights - mean) / (std + epsilon)) ** 3)

                # TensorBoard
                writer.add_scalar('Weights/Kurtosis', kurtosis.item(), i)
                writer.add_scalar('Weights/Skewness', skewness.item(), i)
                writer.add_scalar('Gradients/total_norm', total_norm, i)
                writer.add_scalars('Loss', { 'train': train_loss, 'val': avg_val_loss }, i)       
                writer.add_scalars('Pixel Accuracy', { 'train': train_acc, 'val': val_acc }, i)             
                writer.add_scalar('mIoU Global', mIoU, i)            
                writer.add_scalars('IoU por Clase', { 'Mascota': IoU_por_clase[0], 'Fondo': IoU_por_clase[1], 'Borde': IoU_por_clase[2] }, i)        
                
                # Log multiple learning rates if they exist
                lrs = { f'Group_{j}': group['lr'] for j, group in enumerate(optimizador.param_groups) }
                writer.add_scalars('Learning Rate', lrs, i)
                   
                writer.add_scalars('Loss Weights', { 'cross': peso_cross, 'dice': peso_dice }, i) 

                if scheduler:
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(avg_val_loss)
                    else:
                        scheduler.step()

                mejoro_iou = mIoU > best_iou
                if mejoro_iou:
                    torch.save(modelo.state_dict(), fr'../Models/ModeloPrueba{dia}.pth')
                    best_iou = mIoU
                    best_val_loss = avg_val_loss
                    epochs_sin_mejora = 0
                    progress.console.print(f"[bold green]---- Nuevo mejor modelo con IoU Global: {best_iou:.4f} ---")

                elif mIoU > (best_iou - 0.01) and avg_val_loss < best_val_loss:
                    torch.save(modelo.state_dict(), fr'../Models/ModeloPrueba_BestLoss_{dia}.pth')
                    best_val_loss = avg_val_loss
                    epochs_sin_mejora = 0
                    progress.console.print(f"[bold blue]--> [🛡️] Modelo Más Estable (Bajo Loss): ValLoss: {avg_val_loss:.4f}")
                else:
                    epochs_sin_mejora += 1
                
                if epochs_sin_mejora >= patience:
                    progress.console.print(f"\n[bold red]--- Se detuvo el entrenamiento por falta de mejora en {patience} epochs ---\n")
                    break

                fila_csv = {
                    "train_loss":train_loss,
                    "train_acc":train_acc, 
                    "val_loss":avg_val_loss,
                    "val_acc":val_acc,
                    "val_iou Global":mIoU,
                    "val_iou Clases":IoU_por_clase
                    }
                
                df = pd.DataFrame([fila_csv])
                df.to_csv(fr'logs/training_history{dia}.csv',
                            mode='a', 
                            header=not os.path.exists(fr'logs/training_history{dia}.csv'),
                            index=False)

                progress.console.print(f"[bold yellow]Epoch {i}:[/] Train Loss = [bold]{train_loss:.4f}[/]; Precision = [bold]{train_acc:.2f}%[/]; Val Loss = [bold]{avg_val_loss:.4f}[/], Val Acc = [bold]{val_acc:.2f}%[/], mIoU = [bold]{mIoU:.4f}[/]")
                progress.console.print(f"   [dim]LRs: {', '.join([f'{group['lr']:.2e}' for group in optimizador.param_groups])} | IoU por Clase: Mascota={IoU_por_clase[0]:.4f}, Fondo={IoU_por_clase[1]:.4f}, Borde={IoU_por_clase[2]:.4f}[/]")

                progress.advance(epoch_task)

    writer.close()
    return "Entrenamiento completado"
