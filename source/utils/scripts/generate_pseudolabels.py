import torch
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader

current_dir = os.getcwd()
utils_models_dir = os.path.join(current_dir, "..", "..", 'utils', 'models')
models_dir = os.path.join(current_dir, "..", "..", "..", 'models')
sys.path.append(utils_models_dir)

from daowa_maadV3Prueba import Daowa_maadPrueba
from datasets import get_unlabeled_dataloader

# ─────────────────────────────────────────────
#  Rutas de salida de pseudo-etiquetas
# ─────────────────────────────────────────────
SAVE_DIR_IMAGES = r"C:\Users\PC\Desktop\Abbadon\source\data\image\images"
SAVE_DIR_MASKS  = r"C:\Users\PC\Desktop\Abbadon\source\data\image\masks"
os.makedirs(SAVE_DIR_IMAGES, exist_ok=True)
os.makedirs(SAVE_DIR_MASKS,  exist_ok=True)

# Paleta de colores por clase (Blanco=Mascota, Negro=Fondo, Rojo=Borde)
PALETTE = {
    0: (255,   255,   255),    # Mascota  → blanco
    1: (0,   0, 0),    # Fondo    → negro
    2: (255, 0,   0),    # Borde    → rojo
}

# ─────────────────────────────────────────────
#  Utilidades
# ─────────────────────────────────────────────

def mask_to_color(mask_np):
    h, w = mask_np.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in PALETTE.items():
        color_img[mask_np == cls_id] = color
    return color_img

def denormalize(tensor_img):
    # Solo desnormalizamos los primeros 3 canales (RGB)
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img_rgb = tensor_img[:3, :, :].permute(1, 2, 0).numpy()
    img_rgb = (img_rgb * std + mean).clip(0, 1)
    return (img_rgb * 255).astype(np.uint8)

# ─────────────────────────────────────────────
#  Interfaz de selección granular
# ─────────────────────────────────────────────

def show_batch_and_select(batch, batch_idx):
    """
    Muestra un lote pequeño de imágenes para mejor visibilidad.
    Teclas:
      1-4 → Alternar selección de la imagen (o 1-N según el lote)
      S / Enter → Confirmar y guardar seleccionados
      N / Espacio → Descartar todo el batch
      Q / Esc → Salir
    """
    n = len(batch)
    selected = [True] * n # Por defecto todas seleccionadas

    # Aumentamos el tamaño significativamente (ancho 12, alto 4 por par de imágenes)
    fig, axes = plt.subplots(n, 2, figsize=(12, n * 4.5))
    fig.suptitle(
        f"Lote {batch_idx + 1} | [1-{n}] Alternar Selección | [S/Enter] Guardar Seleccionados",
        fontsize=14, fontweight='bold', y=0.98
    )

    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    # Guardamos referencias a los rectángulos de selección
    rects = []

    for i, (img_t, mask_np, conf, filename) in enumerate(batch):
        img_rgb = denormalize(img_t)
        mask_rgb = mask_to_color(mask_np)

        axes[i][0].imshow(img_rgb)
        axes[i][0].set_title(f"[{i+1}] Imagen: {filename} (conf={conf:.3f})", fontsize=10)
        axes[i][0].axis('off')

        axes[i][1].imshow(mask_rgb)
        axes[i][1].set_title("Máscara Predicha", fontsize=10)
        axes[i][1].axis('off')

        # Rectángulo indicador de selección (Overlay sobre la imagen)
        rect = mpatches.Rectangle((0,0), 1, 1, transform=axes[i][0].transAxes, 
                                  color='lime', alpha=0.3, visible=True)
        axes[i][0].add_patch(rect)
        rects.append(rect)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    result = {'action': 'none', 'selected_indices': []}

    def on_key(event):
        key = event.key
        if key in [str(i) for i in range(1, n + 1)]:
            idx = int(key) - 1
            selected[idx] = not selected[idx]
            rects[idx].set_visible(selected[idx])
            rects[idx].set_color('lime' if selected[idx] else 'red')
            fig.canvas.draw_idle()
        
        elif key.lower() in ('s', 'enter'):
            result['action'] = 'save'
            result['selected_indices'] = [i for i, s in enumerate(selected) if s]
            plt.close(fig)
        
        elif key in ('n', ' '):
            result['action'] = 'discard'
            plt.close(fig)
            
        elif key.lower() in ('q', 'escape'):
            result['action'] = 'quit'
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show(block=True)
    return result

# ─────────────────────────────────────────────
#  Lógica de Pseudo-Labeling
# ─────────────────────────────────────────────

def run_pseudo_labeling(model, dataloader, device, threshold=0.95, batch_display=4):
    model.eval()
    pending = []
    saved_tot = 0
    batch_idx = 0

    with torch.no_grad():
        for imgs, filenames in dataloader:
            preds = model(imgs.to(device))
            probs = torch.softmax(preds, dim=1)
            max_probs, labels = torch.max(probs, dim=1)

            for b in range(imgs.size(0)):
                conf = max_probs[b].mean().item()
                if conf > threshold:
                    pending.append((imgs[b].cpu(), labels[b].cpu().numpy(), conf, filenames[b]))

                if len(pending) >= batch_display:
                    res = show_batch_and_select(pending, batch_idx)
                    batch_idx += 1

                    if res['action'] == 'save':
                        to_save = [pending[i] for i in res['selected_indices']]
                        saved_tot += _save_samples(to_save)
                        print(f"[✓] Guardadas {len(to_save)} imágenes del batch {batch_idx}")
                    elif res['action'] == 'quit':
                        print("[!] Proceso cancelado.")
                        return
                    
                    pending.clear()

    if pending:
        res = show_batch_and_select(pending, batch_idx)
        if res['action'] == 'save':
            to_save = [pending[i] for i in res['selected_indices']]
            saved_tot += _save_samples(to_save)
        print(f"[✓] Fin del proceso. Total: {saved_tot} guardadas.")

def _save_samples(samples):
    count = 0
    for img_t, mask_np, _, original_name in samples:
        base_name = os.path.splitext(original_name)[0]
        full_name = f"pseudo_{base_name}"

        # Guardar Imagen RGB
        img_rgb = denormalize(img_t)
        Image.fromarray(img_rgb).save(os.path.join(SAVE_DIR_IMAGES, full_name + "_dog.jpg"))

        # Guardar Máscara Coloreada (Mascota=Blanco, Fondo=Negro, Borde=Rojo)
        mask_rgb = mask_to_color(mask_np)
        Image.fromarray(mask_rgb).save(os.path.join(SAVE_DIR_MASKS, full_name + "_mask.png"))
        count += 1
    return count

# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Daowa_maadPrueba(num_clases=3).to(device)
    model.load_state_dict(torch.load(os.path.join(models_dir, 'ModeloPrueba2026-03-12_2.pth'), map_location=device), strict=False)
    
    # RUTA A TU CARPETA DE IMÁGENES NUEVAS (SIN ETIQUETAR)
    NEW_IMAGES_PATH = r"C:\Users\PC\Desktop\Abbadon\source\data\PetImages\Dog" # <--- CAMBIA ESTO
    
    if not os.path.exists(NEW_IMAGES_PATH):
        os.makedirs(NEW_IMAGES_PATH, exist_ok=True)
        print(f"[!] Carpeta {NEW_IMAGES_PATH} creada. Pon tus imágenes ahí y vuelve a ejecutar.")
        return

    dataloader = get_unlabeled_dataloader(NEW_IMAGES_PATH, batch_size=16, shape_img=(384, 384))
    
    if len(dataloader.dataset) == 0:
        print(f"[!] No hay imágenes en {NEW_IMAGES_PATH}.")
        return

    print(f"[i] Iniciando pseudo-labeling sobre {len(dataloader.dataset)} imágenes...")
    run_pseudo_labeling(model, dataloader, device, threshold=0.95, batch_display=4)

if __name__ == "__main__":
    main()