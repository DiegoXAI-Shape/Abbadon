import os
import sys
import torch
import datetime
import pandas as pd
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

from utils.models.daowa_maadV3_rc3 import Daowa_maadPrueba
from utils.losses.binary_dice_loss import BinaryDiceLoss
from utils.losses.divergence_kl import DivergenceKL
from utils.train.distillation_dataset import CustomDistillationDataset
from iou import get_intersections_and_unions, IoU_global

NUM_CLASSES = 1  # 0=Fondo, 1=Mascota (binario)

def train_model(
    modelo, 
    loss_fn:list, 
    optimizador, 
    dataloaders:list, 
    device_calc, 
    scheduler = None, 
    epochs:int = 50, 
    epsilon:float = 1e-8, 
    patience:int = 10
    ):

    def get_kl_weight(epoch):
        return max(0.01, 1.0 * (0.7 ** (epoch - 1)))
    
    os.makedirs('logs', exist_ok=True)
    os.makedirs('logs/tensorboard', exist_ok=True)
    os.makedirs(os.path.join(source_path, "Models"), exist_ok=True)
    writer = SummaryWriter(fr'logs/tensorboard/daowa_destilado_{dia}_binary')

    if modelo and isinstance(dataloaders, list):
        
        modelo.to(device_calc)

        #Desempaquetado de Data e Instancias
        BCE_loss, Boundary_loss, Dice_loss, KLD_loss = loss_fn
        train_dl, val_dl, _ = dataloaders
        
        # ─── Métricas para lógica de guardado mejorada ────────────────────
        best_iou = 0.0           
        best_score = 0.0         
        best_val_loss = float('inf')
        epochs_sin_mejora = 0

        print(f"Iniciando entrenamiento Distilado ({NUM_CLASSES} clases: Fondo, Mascota)")

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
                
                # ─── Schedulling Dinámico de Pesos (Fórmula de Interpolación) ───
                # progreso va de 0.0 (Época 1) hasta 1.0 (Última Época)
                progreso = (i - 1) / (epochs - 1) if epochs > 1 else 1.0
                
                # alpha va de 1.0 (Inicio) disminuyendo hasta 0.0 (Final)
                alpha = 1.0 - progreso
                
                # Aplicamos la Ecuación Lineal: Y = Inicio * (alpha) + Final * (1 - alpha)
                # 1. SAM KL (Profesor): Empieza en 1.0, termina en 0.1 (Nunca se apaga del todo)
                peso_kl = 1.0 * alpha + 0.1 * (1 - alpha)
                
                # 2. BCE Loss: Empieza en 1.0 y baja a 0.0 (Misma lógica que tu Phase 3 antigua)
                peso_bce = 1.0 * alpha + 0.0 * (1 - alpha)
                
                # 3. Dice y Boundary: Empiezan apagados (0.0) y terminan en 1.0 para afilar contornos
                peso_dice = 0.0 * alpha + 1.0 * (1 - alpha)
                peso_boundary = 0.0 * alpha + 1.0 * (1 - alpha)
                
                #Métricas
                train_loss_acc = 0.0
                train_correct_pixels = 0
                train_total_pixels = 0
                intersections = torch.zeros(NUM_CLASSES, device=device_calc)
                unions = torch.zeros(NUM_CLASSES, device=device_calc)

                modelo.train()
                
                batch_task = progress.add_task(f"[cyan]Epoch {i} [Train]", total=len(train_dl))
                
                for image, mask_human, mask_SAM in train_dl:
                    image, mask_human, mask_SAM = image.to(device_calc), mask_human.to(device_calc), mask_SAM.to(device_calc)

                    optimizador.zero_grad()
                    with torch.amp.autocast_mode.autocast(device_type="cuda" if device_calc.type == 'cuda' else 'cpu', dtype=torch.bfloat16):
                        output = modelo(image) 

                        loss_bce = BCE_loss(output, mask_human)
                        loss_boundary = Boundary_loss(output, mask_human)
                        loss_dice = Dice_loss(output, mask_human) 
                        loss_kld = KLD_loss(output, mask_SAM)
                        
                        total_loss = (peso_bce * loss_bce) + (peso_boundary * loss_boundary) + (peso_dice * loss_dice) + (peso_kl * loss_kld)
                    
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
                    
                    preds = (output > 0.0).float()
                    train_correct_pixels += (preds == mask_human).sum().item()
                    train_total_pixels += mask_human.numel() 
                    
                    progress.update(batch_task, advance=1, description=f"[cyan]Epoch {i} [Train] Loss: {current_loss:.4f}")

                progress.remove_task(batch_task)
                train_loss = train_loss_acc / len(train_dl)
                train_acc = (train_correct_pixels / train_total_pixels) * 100

                # ─── VALIDACIÓN ───────────────────────────────────────────
                actual_valLoss = 0.0
                val_correct_pixels = 0
                val_total_pixels = 0

                val_task = progress.add_task(f"[yellow]Epoch {i} [Val]", total=len(val_dl))

                modelo.eval()
                with torch.no_grad():
                    for image, mask_human, mask_SAM in val_dl:
                        image = image.to(device_calc)
                        mask_human = mask_human.to(device_calc)
                        mask_SAM = mask_SAM.to(device_calc)
                        
                        with torch.amp.autocast_mode.autocast(device_type="cuda" if device_calc.type == 'cuda' else 'cpu', dtype=torch.bfloat16):
                            predict = modelo(image)

                            v_loss_bce = BCE_loss(predict, mask_human)
                            v_loss_boundary = Boundary_loss(predict, mask_human)
                            v_loss_dice = Dice_loss(predict, mask_human)
                            v_loss_kld = KLD_loss(predict, mask_SAM)
                            val_loss = (peso_bce * v_loss_bce) + (peso_boundary * v_loss_boundary) + (peso_dice * v_loss_dice) + (peso_kl * v_loss_kld)

                        actual_valLoss += val_loss.item()

                        # En Segmentación Binaria output = 1 channel. Softmax no sirve, usamos Logits > 0 = Clase 1
                        _y_predicts = (predict > 0.0).float()

                        # IoU para 2 clases: Fondo(0) y Mascota(1)
                        for cls_id in range(NUM_CLASSES):
                            inter, union = get_intersections_and_unions(_y_predicts, mask_human, cls_id)
                            intersections[cls_id] += inter
                            unions[cls_id] += union

                        val_correct_pixels += (_y_predicts == mask_human).sum().item()
                        val_total_pixels += mask_human.numel()

                        progress.update(val_task, advance=1)
                
                progress.remove_task(val_task)

                avg_val_loss = actual_valLoss / len(val_dl)
                val_acc = (val_correct_pixels / val_total_pixels) * 100
                mIoU, IoU_por_clase = IoU_global(intersections, unions)
                IoU_por_clase = IoU_por_clase.cpu().numpy().tolist()
                
                all_weights = torch.cat([p.view(-1) for p in modelo.parameters() if p.requires_grad]).detach()

                mean = torch.mean(all_weights)
                std = torch.std(all_weights)
                kurtosis = (torch.mean(((all_weights - mean) / (std + epsilon)) ** 4)) - 3
                skewness = torch.mean(((all_weights - mean) / (std + epsilon)) ** 3)

                # ─── TensorBoard ──────────────────────────────────────────
                writer.add_scalar('Weights/Kurtosis', kurtosis.item(), i)
                writer.add_scalar('Weights/Skewness', skewness.item(), i)
                writer.add_scalar('Gradients/total_norm', total_norm, i)
                writer.add_scalars('Loss', { 'train': train_loss, 'val': avg_val_loss }, i)       
                writer.add_scalars('Pixel Accuracy', { 'train': train_acc, 'val': val_acc }, i)             
                writer.add_scalar('mIoU Global', mIoU.item(), i)            
                writer.add_scalars('IoU por Clase', {
                    'Fondo': IoU_por_clase[0],
                    'Mascota': IoU_por_clase[1]
                }, i)        
                
                lrs = { f'Group_{j}': group['lr'] for j, group in enumerate(optimizador.param_groups) }
                writer.add_scalars('Learning Rate', lrs, i)
                writer.add_scalars('Loss Weights', { 'bce': peso_bce, 'boundary': peso_boundary, 'dice': peso_dice, 'kl': peso_kl }, i) 

                if scheduler:
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(avg_val_loss)
                    else:
                        scheduler.step()

                # ─── LÓGICA DE GUARDADO MEJORADA ──────────────────────────
                loss_normalizado = min(avg_val_loss / 2.0, 1.0)
                score = 0.7 * mIoU.item() + 0.3 * (1.0 - loss_normalizado)

                guardado = False
                
                model_dir = os.path.join(source_path, "Models")

                # Checkpoint 1: Mejor IoU absoluto
                if mIoU.item() > best_iou:
                    torch.save(modelo.state_dict(), os.path.join(model_dir, f'BestIoU_{dia}_binary.pth'))
                    best_iou = mIoU.item()
                    epochs_sin_mejora = 0
                    guardado = True
                    progress.console.print(f"[bold green]✓ Mejor IoU: {best_iou:.4f}[/]")

                # Checkpoint 2: Mejor score compuesto (IoU + estabilidad)
                if score > best_score:
                    torch.save(modelo.state_dict(), os.path.join(model_dir, f'BestScore_{dia}_binary.pth'))
                    best_score = score
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                    epochs_sin_mejora = 0
                    guardado = True
                    progress.console.print(f"[bold blue]✓ Mejor Score: {score:.4f} (IoU={mIoU.item():.4f}, ValLoss={avg_val_loss:.4f})[/]")

                if not guardado:
                    epochs_sin_mejora += 1

                # Checkpoint 3: Siempre guardar el último (para poder resumir)
                torch.save(modelo.state_dict(), os.path.join(model_dir, f'Last_{dia}_binary.pth'))

                if epochs_sin_mejora >= patience:
                    progress.console.print(f"\n[bold red]--- Parada anticipada: {patience} epochs sin mejora ---\n")
                    break

                # ─── CSV LOG ──────────────────────────────────────────────
                fila_csv = {
                    "train_loss":train_loss,
                    "train_acc":train_acc, 
                    "val_loss":avg_val_loss,
                    "val_acc":val_acc,
                    "val_iou Global":mIoU.item(),
                    "val_iou Clases":IoU_por_clase,
                    "score":score
                    }
                
                df = pd.DataFrame([fila_csv])
                csv_path = fr'logs/training_history{dia}_binary.csv'
                df.to_csv(csv_path,
                            mode='a', 
                            header=not os.path.exists(csv_path),
                            index=False)

                progress.console.print(f"[bold yellow]Epoch {i}:[/] Train Loss = [bold]{train_loss:.4f}[/]; Acc = [bold]{train_acc:.2f}%[/]; Val Loss = [bold]{avg_val_loss:.4f}[/], Val Acc = [bold]{val_acc:.2f}%[/], mIoU = [bold]{mIoU.item():.4f}[/], Score = [bold]{score:.4f}[/]")
                
                lrs_str = ', '.join([f"{group['lr']:.2e}" for group in optimizador.param_groups])
                progress.console.print(f"   [dim]LRs: {lrs_str} | IoU: Fondo={IoU_por_clase[0]:.4f}, Mascota={IoU_por_clase[1]:.4f}[/]")

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
    sam_masks_dir = os.path.join(base_data_path, "masks_SAM")
    csv_labels_path = os.path.join(source_path, "labels", "dataset_generated.csv")
    
    # 2. Cargar Dataset y dividir en Train / Val
    df_labels = pd.read_csv(csv_labels_path)
    dataset = CustomDistillationDataset(df=df_labels, imgs_dir=imgs_dir, human_masks_dir=human_masks_dir, sam_logits_dir=sam_masks_dir, img_size=(256, 256))
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)
    
    dataloaders = [train_loader, val_loader, None]
    
    modelo = Daowa_maadPrueba(in_channels=3, out_channels=1) 
    
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=1e-4, weight_decay=1e-5)
    
    loss_fn = [
        torch.nn.BCEWithLogitsLoss(),
        BinaryDiceLoss(),
        DivergenceKL(temperature=2.0)
    ]
    
    print("Iniciando bucle de entrenamiento destilado...")
    train_model(
        modelo=modelo,
        loss_fn=loss_fn,
        optimizador=optimizador,
        dataloaders=dataloaders,
        device_calc=device,
        epochs=50,
        patience=10
    )

if __name__ == "__main__":
    main()
