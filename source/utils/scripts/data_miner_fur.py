import os
import hashlib
from PIL import Image
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import sys

# Importar YOLO y SAM (Asegúrate de tener ultralytics instalado)
try:
    from ultralytics import YOLO
except ImportError:
    print("Por favor instala ultralytics: pip install ultralytics")
    sys.exit(1)

# Importar SAM loader desde la carpeta del proyecto
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..', '..'))
sys.path.append(os.path.join(project_root, 'source', 'utils', 'models'))

from sam_loader import load_sam

def get_image_hash(filepath):
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def is_valid_image(filepath):
    try:
        img = Image.open(filepath)
        img.verify()
        img = Image.open(filepath).convert('RGB')
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        if img_cv is None or img_cv.size == 0:
            return False, None
        h, w = img_cv.shape[:2]
        if h < 100 or w < 100:
            return False, None
        return True, img_cv
    except Exception:
        return False, None

def process_and_segment_images(target_dir, masks_dir):
    print(f"\n🧠 Cargando modelos YOLO y SAM...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # YOLOv8 nano es súper rápido y ligero
    yolo_model = YOLO('yolov8n.pt')
    sam_predictor = load_sam(model_type="tiny", device=device)
    
    print(f"\n🧹 Procesando imágenes en {target_dir}...")
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Carpeta creada. Por favor pon tus imágenes ahí.")
        return []
        
    os.makedirs(masks_dir, exist_ok=True)

    results_data = []
    hash_set = set()
    files = [f for f in os.listdir(target_dir) if os.path.isfile(os.path.join(target_dir, f))]
    
    if not files:
        print("La carpeta está vacía.")
        return []

    for filename in tqdm(files, desc="Analizando y segmentando"):
        filepath = os.path.join(target_dir, filename)
        
        is_valid, img_cv = is_valid_image(filepath)
        if not is_valid:
            os.remove(filepath)
            continue
            
        img_hash = get_image_hash(filepath)
        if img_hash in hash_set:
            os.remove(filepath)
            continue
        hash_set.add(img_hash)
        
        # Renombrar a estandar
        new_filename = f"hard_neg_{img_hash[:8]}.jpg"
        new_filepath = os.path.join(target_dir, new_filename)
        
        if filename != new_filename:
            cv2.imwrite(new_filepath, img_cv)
            os.remove(filepath)
            filepath = new_filepath
            
        # Detectar con YOLO
        # Clases COCO: 15 es gato (cat), 16 es perro (dog)
        yolo_results = yolo_model(img_cv, classes=[15, 16], verbose=False)
        boxes = yolo_results[0].boxes
        
        has_pet = False
        mask_path = ""
        is_adv = 1 # Por defecto es pura textura (sin mascota)
        
        if len(boxes) > 0:
            # Encontramos un perro/gato!
            has_pet = True
            is_adv = 0 # Ahora es un positivo para el dataloader
            
            # Tomar la primera mascota detectada
            box = boxes[0].xyxy[0].cpu().numpy() # [x1, y1, x2, y2]
            center_x = int((box[0] + box[2]) / 2)
            center_y = int((box[1] + box[3]) / 2)
            
            # Pasar a SAM
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            sam_predictor.set_image(img_rgb)
            
            input_point = np.array([[center_x, center_y]])
            input_label = np.array([1]) # Foreground
            
            masks, scores, _ = sam_predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True
            )
            
            best_mask = masks[np.argmax(scores)]
            
            # Guardar máscara (Formato Oxford: 0=Mascota, 1=Fondo)
            mask_uint8 = np.ones_like(best_mask, dtype=np.uint8)  # Llenar de 1s (Fondo)
            mask_uint8[best_mask > 0] = 0                         # Poner 0 donde hay mascota
            
            mask_filename = new_filename.replace('.jpg', '.png')
            mask_path = os.path.join(masks_dir, mask_filename)
            cv2.imwrite(mask_path, mask_uint8)
            
            print(f"\n[!] Mascota detectada en {new_filename}. Máscara generada con SAM.")
        else:
            print(f"\n[-] Pura textura en {new_filename}. Se va a 'tirar a lion' (is_adv=1).")
            
        results_data.append({
            'image': new_filepath,
            'mask': mask_path,
            'is_adv': is_adv
        })

    return results_data

def generate_csv(results_data, csv_output_path, base_dir):
    if not results_data:
        return
        
    df = pd.DataFrame(results_data)
    
    # Hacer rutas relativas para el dataloader
    df['image'] = df['image'].apply(lambda p: os.path.relpath(p, base_dir).replace('\\', '/'))
    df['mask'] = df['mask'].apply(lambda p: os.path.relpath(p, base_dir).replace('\\', '/') if p else "")
    
    df.to_csv(csv_output_path, index=False)
    print(f"\n✨ CSV guardado con {len(df)} filas en {csv_output_path}")

if __name__ == "__main__":
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..', '..'))
    
    output_dir = os.path.join(project_root, "source", "data", "hard_negatives", "fur_coats")
    masks_dir = os.path.join(project_root, "source", "data", "hard_negatives", "masks")
    csv_path = os.path.join(project_root, "source", "labels", "hard_negatives_fur.csv")
    
    # 1. Procesa, pasa por YOLO y SAM
    results = process_and_segment_images(output_dir, masks_dir)
    
    # 2. Genera el CSV
    generate_csv(results, csv_path, project_root)
