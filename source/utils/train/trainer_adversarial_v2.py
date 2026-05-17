"""
trainer_adversarial_v2.py
=========================
Trainer limpio para adversarial fine-tuning con un único DataLoader.

Diferencias con trainer_adversarial_finetune.py (versión anterior):
    - Un solo DataLoader con NegativeAwareBatchSampler (no zip de dos loaders)
    - El flag `es_adversarial` por muestra separa el logging sin doble forward
    - Loss combinada AdversarialSegLoss / BurnInAdversarialLoss (Dice + Boundary)
    - Sin BCE ni KL Divergence
    - Gradient Clipping + AMP (bfloat16)
    - Early stopping por score ponderado (70% IoU_adv + 30% IoU_pos)
"""

import os
import sys
import csv
import datetime
import random
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from rich.progress import (
    Progress, TextColumn, BarColumn,
    TaskProgressColumn, TimeRemainingColumn, MofNCompleteColumn,
)

# ─── Resolución de paths ─────────────────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
source_path = os.path.abspath(os.path.join(current_dir, "..", ".."))
if source_path not in sys.path:
    sys.path.append(source_path)

metrics_dir = os.path.join(source_path, "utils", "metrics")
if metrics_dir not in sys.path:
    sys.path.append(metrics_dir)

os.chdir(source_path)

from utils.losses.adversarial_loss import BurnInAdversarialLoss
from iou import get_intersections_and_unions, IoU_global   # tu módulo existente


# ─── Función principal de entrenamiento ─────────────────────────────────────

def train_model(
    modelo,
    train_loader,
    val_loader,
    optimizador,
    device,
    epochs: int = 50,
    patience: int = 10,
    scheduler=None,
    w_dice: float = 1.0,
    w_boundary: float = 0.1,
    accum_steps: int = 4,       # Gradient accumulation (simula batch_size × accum_steps)
):
    """
    Loop de entrenamiento adversarial unificado.

    Args:
        modelo        : Tu red Daowa_maadPrueba (o cualquier nn.Module compatible).
        train_loader  : DataLoader con NegativeAwareBatchSampler.
        val_loader    : DataLoader de validación.
        optimizador   : torch.optim (AdamW recomendado).
        device        : torch.device
        epochs        : Número de épocas máximo.
        patience      : Early stopping.
        scheduler     : LRScheduler opcional.
        w_dice        : Peso inicial del Dice Loss.
        w_boundary    : Peso inicial del Boundary Loss (crece con BurnIn).
        accum_steps   : Cuantos batches acumular antes de hacer optimizer.step().
                        Batch real en GPU = batch_size. Batch efectivo = batch_size × accum_steps.
                        Con batch_size=16 y accum_steps=4 → efectivo de 64.
    """
    dia = datetime.date.today()

    os.makedirs("logs/tensorboard", exist_ok=True)
    os.makedirs(os.path.join(source_path, "Models"), exist_ok=True)
    writer = SummaryWriter(f"logs/tensorboard/adv_v2_{dia}_2")

    loss_fn = BurnInAdversarialLoss(w_dice=w_dice, w_boundary=w_boundary)
    modelo.to(device)

    # GradScaler para AMP correcto en float16/bfloat16
    use_cuda = device.type == "cuda"
    scaler   = torch.amp.GradScaler(device="cuda") if use_cuda else None
    amp_ctx  = lambda: torch.amp.autocast(device_type="cuda" if use_cuda else "cpu", dtype=torch.bfloat16)

    NUM_CLASSES  = 1
    best_adv_iou = 0.0
    best_score   = 0.0
    no_mejora    = 0

    # ── Historial de métricas por época ─────────────────────────────────────
    os.makedirs("logs", exist_ok=True)
    history_path = os.path.join(source_path, "logs", f"history_adv_{dia}_2.csv")
    history_fields = [
        "epoch", "train_loss", "val_loss",
        "dice_loss", "boundary_loss", "w_dice", "w_boundary",
        "acc_pos", "acc_neg", "iou_pos", "iou_neg",
        "score", "grad_norm", "lr",
    ]
    with open(history_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=history_fields).writeheader()
    print(f"[trainer] 💾 Métricas por época → {history_path}")

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        expand=True,
    ) as progress:
        epoch_task = progress.add_task("[bold magenta]Entrenamiento Total", total=epochs)

        for epoch in range(1, epochs + 1):
            progress.update(epoch_task, description=f"[bold magenta]Epoch {epoch}/{epochs}")

            # Actualizar pesos dinámicos (Burn-In)
            loss_fn.step_epoch(epoch, epochs)

            # ── TRAIN ─────────────────────────────────────────────────────────
            modelo.train()
            acc_loss    = 0.0
            acc_dice    = 0.0
            acc_boundary= 0.0
            corr_pos, tot_pos = 0, 0
            corr_neg, tot_neg = 0, 0
            total_norm_acc = 0.0

            batch_task = progress.add_task(
                f"[cyan]Epoch {epoch} [Train]", total=len(train_loader)
            )

            for step, batch in enumerate(train_loader):
                imgs, masks, sdfs, flags_adv = batch
                imgs  = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                sdfs  = sdfs.to(device, non_blocking=True)

                with amp_ctx():
                    logits = modelo(imgs)
                    # Dividimos la loss por accum_steps para que la escala sea correcta
                    loss, detalle = loss_fn(logits, masks, sdfs)
                    loss_scaled   = loss / accum_steps

                if scaler:
                    scaler.scale(loss_scaled).backward()
                else:
                    loss_scaled.backward()

                # Solo actualizamos pesos cada accum_steps batches
                if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                    if scaler:
                        scaler.unscale_(optimizador)
                        grad_norm = torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
                        scaler.step(optimizador)
                        scaler.update()
                    else:
                        grad_norm = torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
                        optimizador.step()
                    optimizador.zero_grad(set_to_none=True)   # set_to_none libera VRAM inmediatamente
                    total_norm_acc += grad_norm.item()

                acc_loss     += loss.detach().item()
                acc_dice     += detalle["dice"].item()
                acc_boundary += detalle["boundary"].item()

                # ── Accuracy (sin guardar grafo) ──────────────────────────────
                with torch.no_grad():
                    preds = (logits.detach() > 0.0).float()
                    if not isinstance(flags_adv, torch.Tensor):
                        flags_adv = torch.tensor(flags_adv, dtype=torch.bool)
                    flags_adv = flags_adv.to(device)

                    if (~flags_adv).any():
                        corr_pos += (preds[~flags_adv] == masks[~flags_adv]).sum().item()
                        tot_pos  += masks[~flags_adv].numel()
                    if flags_adv.any():
                        corr_neg += (preds[flags_adv] == masks[flags_adv]).sum().item()
                        tot_neg  += masks[flags_adv].numel()

                # Liberar referencias explícitamente para no acumular en VRAM
                del imgs, masks, sdfs, logits, loss, loss_scaled

                progress.update(
                    batch_task, advance=1,
                    description=f"[cyan]Epoch {epoch} Loss: {acc_loss / (step + 1):.4f}"
                )

            progress.remove_task(batch_task)

            n_batches    = len(train_loader)
            avg_loss     = acc_loss     / n_batches
            avg_dice     = acc_dice     / n_batches
            avg_boundary = acc_boundary / n_batches
            avg_norm     = total_norm_acc / n_batches
            acc_pos = (corr_pos / tot_pos * 100) if tot_pos > 0 else 0.0
            acc_neg = (corr_neg / tot_neg * 100) if tot_neg > 0 else 0.0

            # ── VAL ───────────────────────────────────────────────────────────
            modelo.eval()
            val_loss_acc = 0.0
            int_pos = torch.zeros(NUM_CLASSES, device=device)
            uni_pos = torch.zeros(NUM_CLASSES, device=device)
            int_neg = torch.zeros(NUM_CLASSES, device=device)
            uni_neg = torch.zeros(NUM_CLASSES, device=device)

            val_task = progress.add_task(
                f"[yellow]Epoch {epoch} [Val]", total=len(val_loader)
            )

            with torch.no_grad():
                for batch in val_loader:
                    imgs, masks, sdfs, flags_adv = batch
                    imgs  = imgs.to(device)
                    masks = masks.to(device)
                    sdfs  = sdfs.to(device)

                    with torch.amp.autocast_mode.autocast(
                        device_type="cuda" if device.type == "cuda" else "cpu",
                        dtype=torch.bfloat16,
                    ):
                        logits     = modelo(imgs)
                        val_loss, _ = loss_fn(logits, masks, sdfs)

                    val_loss_acc += val_loss.item()
                    preds = (logits > 0.0).float()

                    if not isinstance(flags_adv, torch.Tensor):
                        flags_adv = torch.tensor(flags_adv, dtype=torch.bool)
                    flags_adv = flags_adv.to(device)

                    # IoU separado por positivos y negativos
                    for cls_id in range(NUM_CLASSES):
                        if (~flags_adv).any():
                            i, u = get_intersections_and_unions(
                                preds[~flags_adv], masks[~flags_adv], cls_id
                            )
                            int_pos[cls_id] += i
                            uni_pos[cls_id] += u
                        if flags_adv.any():
                            i, u = get_intersections_and_unions(
                                preds[flags_adv], masks[flags_adv], cls_id
                            )
                            int_neg[cls_id] += i
                            uni_neg[cls_id] += u

                    progress.update(val_task, advance=1)

            progress.remove_task(val_task)

            avg_val_loss = val_loss_acc / max(len(val_loader), 1)
            iou_pos_global, _ = IoU_global(int_pos, uni_pos)
            iou_neg_global, _ = IoU_global(int_neg, uni_neg)

            iou_pos_val = iou_pos_global.item()
            iou_neg_val = iou_neg_global.item()
            score       = 0.7 * iou_neg_val + 0.3 * iou_pos_val
            lr_actual   = optimizador.param_groups[0]["lr"]

            # ── TensorBoard ───────────────────────────────────────────────────
            writer.add_scalars("Loss", {"train": avg_loss, "val": avg_val_loss}, epoch)
            writer.add_scalars("Loss/Componentes_Train", {"dice": avg_dice, "boundary": avg_boundary}, epoch)
            writer.add_scalars("Loss/Weights", {"w_dice": loss_fn.w_dice, "w_boundary": loss_fn.w_boundary}, epoch)
            writer.add_scalars("Pixel_Acc/Train", {"positivos": acc_pos, "negativos": acc_neg}, epoch)
            writer.add_scalars("mIoU", {"positivos": iou_pos_val, "negativos": iou_neg_val}, epoch)
            writer.add_scalar("Gradients/norm", avg_norm, epoch)
            writer.add_scalars(
                "LR",
                {f"group_{j}": g["lr"] for j, g in enumerate(optimizador.param_groups)},
                epoch,
            )

            # ── Guardar fila en CSV ────────────────────────────────────────────
            with open(history_path, "a", newline="") as f:
                writer_csv = csv.DictWriter(f, fieldnames=history_fields)
                writer_csv.writerow({
                    "epoch"        : epoch,
                    "train_loss"   : f"{avg_loss:.6f}",
                    "val_loss"     : f"{avg_val_loss:.6f}",
                    "dice_loss"    : f"{avg_dice:.6f}",
                    "boundary_loss": f"{avg_boundary:.6f}",
                    "w_dice"       : f"{loss_fn.w_dice:.4f}",
                    "w_boundary"   : f"{loss_fn.w_boundary:.4f}",
                    "acc_pos"      : f"{acc_pos:.2f}",
                    "acc_neg"      : f"{acc_neg:.2f}",
                    "iou_pos"      : f"{iou_pos_val:.6f}",
                    "iou_neg"      : f"{iou_neg_val:.6f}",
                    "score"        : f"{score:.6f}",
                    "grad_norm"    : f"{avg_norm:.4f}",
                    "lr"           : f"{lr_actual:.2e}",
                })

            # ── Scheduler ─────────────────────────────────────────────────────
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(avg_val_loss)
                else:
                    scheduler.step()

            # ── Guardado de modelos ───────────────────────────────────────────
            model_dir = os.path.join(source_path, "Models")
            guardado  = False

            if iou_neg_val > best_adv_iou:
                path = os.path.join(model_dir, f"BestAdv_IoU_{dia}_{iou_neg_val:.3f}_2.pth")
                torch.save(modelo.state_dict(), path)
                best_adv_iou = iou_neg_val
                no_mejora    = 0
                guardado     = True
                progress.console.print(f"[bold green]✓ Nuevo mejor IoU adversarial: {best_adv_iou:.4f}[/]")

            if score > best_score:
                path = os.path.join(model_dir, f"BestScore_Adv_{dia}_{score:.3f}_2.pth")
                torch.save(modelo.state_dict(), path)
                best_score = score
                no_mejora  = 0
                guardado   = True
                progress.console.print(f"[bold blue]✓ Nuevo mejor score global: {best_score:.4f}[/]")

            # Siempre guardamos el último checkpoint
            torch.save(modelo.state_dict(), os.path.join(model_dir, f"Last_Adv_{dia}_2.pth"))

            if not guardado:
                no_mejora += 1

            progress.console.print(
                f"[bold yellow]Epoch {epoch}:[/] "
                f"TrainLoss={avg_loss:.4f} ValLoss={avg_val_loss:.4f} | "
                f"Acc_Pos={acc_pos:.1f}% Acc_Neg={acc_neg:.1f}% | "
                f"IoU_Pos={iou_pos_val:.4f} IoU_Neg={iou_neg_val:.4f}"
            )
            progress.console.print(
                f"   Pesos → Dice={loss_fn.w_dice:.2f} Boundary={loss_fn.w_boundary:.2f}"
            )

            if no_mejora >= patience:
                progress.console.print(
                    f"\n[bold red]Early stopping: {patience} epochs sin mejora.[/]\n"
                )
                break

            progress.advance(epoch_task)

    writer.close()
    print(f"[trainer] ✅ Historial completo en: {history_path}")
    return history_path


# ─── main() de referencia ────────────────────────────────────────────────────

def main():
    """
    Ejemplo mínimo de uso. Ajusta las rutas y parámetros a tu entorno.
    """
    import pandas as pd
    from torch.utils.data import DataLoader
    from utils.models.daowa_maadV3_rc3 import Daowa_maadPrueba
    from utils.train.adversarial_dataset import AdversarialPetDataset, get_adversarial_dataloader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando: {device}")

    # ── Rutas ────────────────────────────────────────────────────────────────
    oxford_dir  = os.path.join(source_path, "data", "oxford")
    ade20k_dir  = os.path.join(source_path, "data", "ADEChallengeData2016")
    csv_oxford  = os.path.join(source_path, "labels", "dataset_generated.csv")
    csv_ade20k  = os.path.join(source_path, "labels", "ade20k_adversarial_train.csv")

    df_oxford = pd.read_csv(csv_oxford)
    df_ade20k = pd.read_csv(csv_ade20k)

    # Filtrar pseudo-gold si existe la columna
    if "is_gold" in df_oxford.columns:
        df_oxford = df_oxford[df_oxford["is_gold"] == False]

    # Split 80/20
    train_oxford = df_oxford.sample(frac=0.8, random_state=42)
    val_oxford   = df_oxford.drop(train_oxford.index)
    train_ade20k = df_ade20k.sample(frac=0.8, random_state=42)
    val_ade20k   = df_ade20k.drop(train_ade20k.index)

    # ── DataLoaders ──────────────────────────────────────────────────────────
    train_loader, _ = get_adversarial_dataloader(
        df_oxford=train_oxford,
        df_ade20k=train_ade20k,
        oxford_dir=oxford_dir,
        ade20k_dir=ade20k_dir,
        batch_size=64,
        num_pos_per_batch=54,
        img_size=(384, 384),
        num_workers=4,
    )

    # Validación: sin sampler especial, mezclamos todo y el flag ya viene en el batch
    val_dataset = AdversarialPetDataset(
        df_oxford=val_oxford,
        df_ade20k=val_ade20k,
        oxford_dir=oxford_dir,
        ade20k_dir=ade20k_dir,
        img_size=(384, 384),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    # ── Modelo ───────────────────────────────────────────────────────────────
    modelo = Daowa_maadPrueba(num_clases=1)

    # Cargar pesos pre-entrenados si existen (descomenta y ajusta la ruta)
    # modelo.load_state_dict(torch.load("Models/tu_modelo_previo.pth", map_location=device))

    # ── Optimizador y Scheduler ──────────────────────────────────────────────
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=5e-5, weight_decay=1e-5)
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(optimizador, T_max=50, eta_min=1e-6)

    # ── Entrenamiento ────────────────────────────────────────────────────────
    train_model(
        modelo=modelo,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizador=optimizador,
        device=device,
        epochs=50,
        patience=12,
        scheduler=scheduler,
    )


if __name__ == "__main__":
    main()


# ─── evaluate_model() ─────────────────────────────────────────────────────────

def evaluate_model(modelo, val_loader, device, threshold: float = 0.0):
    """
    Eval completa sobre un DataLoader (val o test).
    Devuelve un dict con todas las métricas separadas por positivos y negativos.

    Args:
        modelo      : Red en cualquier estado (la pone en eval internamente).
        val_loader  : DataLoader que devuelve (img, mask, sdf, is_adv).
        device      : torch.device
        threshold   : Umbral sobre el logit (default 0.0 = sigmoid > 0.5)

    Returns:
        dict con claves: val_loss, iou_pos, iou_neg, acc_pos, acc_neg, score
    """
    from utils.losses.adversarial_loss import BurnInAdversarialLoss

    loss_fn  = BurnInAdversarialLoss()
    modelo.to(device)
    modelo.eval()

    use_cuda = device.type == "cuda"
    amp_ctx  = lambda: torch.amp.autocast(
        device_type="cuda" if use_cuda else "cpu", dtype=torch.bfloat16
    )

    NUM_CLASSES  = 1
    val_loss_acc = 0.0
    int_pos = torch.zeros(NUM_CLASSES, device=device)
    uni_pos = torch.zeros(NUM_CLASSES, device=device)
    int_neg = torch.zeros(NUM_CLASSES, device=device)
    uni_neg = torch.zeros(NUM_CLASSES, device=device)
    corr_pos, tot_pos = 0, 0
    corr_neg, tot_neg = 0, 0

    print(f"Evaluando sobre {len(val_loader)} batches...")

    with torch.no_grad():
        for batch in val_loader:
            imgs, masks, sdfs, flags_adv = batch
            imgs  = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            sdfs  = sdfs.to(device, non_blocking=True)

            with amp_ctx():
                logits  = modelo(imgs)
                loss, _ = loss_fn(logits, masks, sdfs)

            val_loss_acc += loss.item()
            preds = (logits > threshold).float()

            if not isinstance(flags_adv, torch.Tensor):
                flags_adv = torch.tensor(flags_adv, dtype=torch.bool)
            flags_adv = flags_adv.to(device)

            for cls_id in range(NUM_CLASSES):
                if (~flags_adv).any():
                    i, u = get_intersections_and_unions(
                        preds[~flags_adv], masks[~flags_adv], cls_id
                    )
                    int_pos[cls_id] += i
                    uni_pos[cls_id] += u
                    corr_pos += (preds[~flags_adv] == masks[~flags_adv]).sum().item()
                    tot_pos  += masks[~flags_adv].numel()
                if flags_adv.any():
                    i, u = get_intersections_and_unions(
                        preds[flags_adv], masks[flags_adv], cls_id
                    )
                    int_neg[cls_id] += i
                    uni_neg[cls_id] += u
                    corr_neg += (preds[flags_adv] == masks[flags_adv]).sum().item()
                    tot_neg  += masks[flags_adv].numel()

    iou_pos, _ = IoU_global(int_pos, uni_pos)
    iou_neg, _ = IoU_global(int_neg, uni_neg)

    metrics = {
        "val_loss" : round(val_loss_acc / max(len(val_loader), 1), 6),
        "iou_pos"  : round(iou_pos.item(), 4),
        "iou_neg"  : round(iou_neg.item(), 4),
        "acc_pos"  : round(corr_pos / tot_pos * 100, 2) if tot_pos > 0 else 0.0,
        "acc_neg"  : round(corr_neg / tot_neg * 100, 2) if tot_neg > 0 else 0.0,
        "score"    : round(0.7 * iou_neg.item() + 0.3 * iou_pos.item(), 4),
    }

    print(
        f"\u2705 Evaluación completa:\n"
        f"   Val Loss : {metrics['val_loss']:.4f}\n"
        f"   IoU Pos  : {metrics['iou_pos']:.4f}   Acc Pos: {metrics['acc_pos']:.1f}%\n"
        f"   IoU Neg  : {metrics['iou_neg']:.4f}   Acc Neg: {metrics['acc_neg']:.1f}%\n"
        f"   Score    : {metrics['score']:.4f}  (0.7×IoU_Neg + 0.3×IoU_Pos)"
    )
    return metrics


# ─── predict_random_samples() ─────────────────────────────────────────────────────────

def predict_random_samples(
    modelo,
    loader,
    device,
    n_samples: int = 8,
    threshold: float = 0.0,
    titulo: str = "Validación Visual Aleatoria",
):
    """
    Muestra una cuadrícula de predicciones sobre muestras aleatorias del loader.

    Por cada muestra muestra 3 columnas:
        Imagen desnormalizada | Máscara GT | Predicción

    Args:
        modelo     : Red (se pone en eval internamente).
        loader     : Cualquier DataLoader adversarial (val, test, train).
        device     : torch.device
        n_samples  : Cuántas muestras mostrar.
        threshold  : Umbral sobre logit (default 0.0).
        titulo     : Título del plot.

    Returns:
        fig  : Figura de matplotlib.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
    IMAGENET_STD  = np.array([0.229, 0.224, 0.225])

    modelo.to(device)
    modelo.eval()

    # Tomar un batch aleatorio del loader
    batch_list = list(loader)
    batch = random.choice(batch_list)
    imgs, masks, sdfs, flags_adv = batch

    n = min(n_samples, imgs.shape[0])
    indices = random.sample(range(imgs.shape[0]), n)

    imgs_sel  = imgs[indices].to(device)
    masks_sel = masks[indices]
    flags_sel = (
        flags_adv[indices]
        if isinstance(flags_adv, torch.Tensor)
        else torch.tensor([flags_adv[i] for i in indices], dtype=torch.bool)
    )

    use_cuda = device.type == "cuda"
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda" if use_cuda else "cpu", dtype=torch.bfloat16):
            logits = modelo(imgs_sel)
        preds = (logits > threshold).float().cpu()

    imgs_cpu = imgs_sel.cpu().float()

    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(titulo, fontsize=14, fontweight="bold")

    for row in range(n):
        img_np = imgs_cpu[row].permute(1, 2, 0).numpy()
        img_np = (img_np * IMAGENET_STD + IMAGENET_MEAN).clip(0, 1)

        gt_np   = masks_sel[row].squeeze().numpy()
        pred_np = preds[row].squeeze().numpy()
        is_adv  = bool(flags_sel[row].item())
        label   = "👎 Negativo (ADE20K)" if is_adv else "🐾 Positivo (Oxford)"
        match   = np.mean(pred_np == gt_np) * 100

        axes[row, 0].imshow(img_np)
        axes[row, 0].set_title(f"{label}\nPixel Acc: {match:.1f}%", fontsize=9)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(gt_np, cmap="RdYlGn", vmin=0, vmax=1)
        axes[row, 1].set_title("Ground Truth", fontsize=9)
        axes[row, 1].axis("off")

        axes[row, 2].imshow(pred_np, cmap="RdYlGn", vmin=0, vmax=1)
        axes[row, 2].set_title("Predicción", fontsize=9)
        axes[row, 2].axis("off")

    verde = mpatches.Patch(color="green", label="Mascota (1)")
    rojo  = mpatches.Patch(color="red",   label="Fondo  (0)")
    fig.legend(handles=[verde, rojo], loc="lower center", ncol=2, fontsize=9)
    plt.tight_layout()
    plt.show()
    return fig
