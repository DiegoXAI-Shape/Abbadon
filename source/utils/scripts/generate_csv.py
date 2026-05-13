"""
Genera un CSV con el formato file,mask,is_gold leyendo las carpetas de imágenes y máscaras.
  is_gold=True  -> Pseudo-etiqueta generada por el modelo (empieza con "pseudo_")
  is_gold=False -> Imagen del dataset original con máscara manual
Salida: C:/Users/PC/Desktop/Abbadon prueba SAM/source/labels/dataset_generated.csv
"""
import os
import csv

IMAGES_DIR = r"C:\Users\PC\Desktop\Abbadon prueba SAM\source\data\oxford\images"
MASKS_DIR  = r"C:\Users\PC\Desktop\Abbadon prueba SAM\source\data\oxford\masks"
MASKS_SAM_DIR = r"C:\Users\PC\Desktop\Abbadon prueba SAM\source\data\oxford\masks_SAM"
OUTPUT_CSV = r"C:\Users\PC\Desktop\Abbadon prueba SAM\source\labels\dataset_generated.csv"

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}
KNOWN_SUFFIXES = ['_dog', '_cat', '_pet']

def find_mask(img_stem, masks_dir):
    """Busca la máscara correspondiente a una imagen dada por su stem."""
    candidates = [img_stem]
    for suffix in KNOWN_SUFFIXES:
        if img_stem.endswith(suffix):
            candidates.append(img_stem[: -len(suffix)])

    for stem in candidates:
        for ext in ('.png', '.jpg', '.jpeg', '.npy'):
            # Buscar sin sufijo
            path = os.path.join(masks_dir, stem + ext)
            if os.path.exists(path):
                return stem + ext
            # Buscar con sufijo _mask
            path_mask = os.path.join(masks_dir, stem + '_mask' + ext)
            if os.path.exists(path_mask):
                return stem + '_mask' + ext
            # Buscar con sufijo _SAM
            path_sam = os.path.join(masks_dir, stem + '_SAM' + ext)
            if os.path.exists(path_sam):
                return stem + '_SAM' + ext
    return None

rows = []
gold_count = 0
normal_count = 0

for fname in sorted(os.listdir(IMAGES_DIR)):
    ext = os.path.splitext(fname)[1].lower()
    if ext not in IMAGE_EXTS:
        continue

    stem = os.path.splitext(fname)[0]
    
    # 1. VERIFICACIÓN CRÍTICA: ¿Existe máscara de SAM?
    mask_sam = find_mask(stem, MASKS_SAM_DIR)
    if mask_sam is None:
        print(f"[!] Sin máscara de SAM (Descartada del CSV): {fname}")
        continue

    # 2. Verificación de máscara humana (opcional según su lógica, pero necesaria para la ruta)
    mask_human = find_mask(stem, MASKS_DIR)
    
    # Si llegó aquí, TIENE máscara de SAM. Asignamos la máscara final que se guardará.
    # Prioridad a la humana si existe, de lo contrario usamos la de SAM.
    final_mask = mask_human if mask_human else mask_sam

    # Las pseudo-etiquetas generadas por el modelo tienen prefijo "pseudo_"
    is_gold = fname.startswith("pseudo_")
    if is_gold:
        gold_count += 1
    else:
        normal_count += 1

    rows.append({'file': fname, 'mask': final_mask, 'is_gold': is_gold})

with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['file', 'mask', 'is_gold'])
    writer.writeheader()
    writer.writerows(rows)

print(f"[✓] CSV generado: {OUTPUT_CSV}")
print(f"[i] Total de pares: {len(rows)}")
print(f"    ├── Dataset original (is_gold=False): {normal_count}")
print(f"    └── Pseudo-etiquetas (is_gold=True) : {gold_count}")