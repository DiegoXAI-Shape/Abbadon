"""
generate_person_negatives_csv.py
================================
Genera un CSV de "hard negatives de persona" desde ADE20K.

El problema a resolver:
    El modelo confunde personas con abrigos de piel real con mascotas, porque
    nunca vio ejemplos de "pelaje real + silueta humana = NO mascota".

Estrategia:
    Buscamos imágenes de ADE20K que contengan personas (clase 13) con alguna
    textura trampa (ropa, telas, pieles) en la misma escena. Esas imágenes
    se añaden al dataset adversarial con máscara = todo ceros.

    Clases ADE20K relevantes (IDs en las anotaciones PNG, 1-indexed):
        13  : person / people
        92  : apparel / clothing
        75  : cloth / fabric
        27  : clothes
        95  : coat / jacket
        100 : fur coat (si existe en el split)

    Adicionalmente, se incluyen escenas con persona + cualquiera de las
    clases trampa originales (sofás, tapetes) para maximizar la diversidad.

Ejecutar UNA VEZ antes de entrenar:
    python utils/scripts/generate_person_negatives_csv.py

Output:
    labels/ade20k_person_negatives.csv
    Columnas: image, annotation
"""

import os
import sys
import numpy as np
from PIL import Image
import pandas as pd

# ─── Paths por defecto ────────────────────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
source_dir  = os.path.abspath(os.path.join(current_dir, "..", ".."))

ADE20K_DIR  = os.path.join(source_dir, "data", "ADEChallengeData2016")
LABELS_DIR  = os.path.join(source_dir, "labels")
OUTPUT_CSV  = os.path.join(LABELS_DIR, "ade20k_person_negatives.csv")

# ─── Clases de interés ────────────────────────────────────────────────────────

# Clase obligatoria: debe estar en la imagen
CLASE_PERSONA = {13}

# Clases que aumentan relevancia (persona + alguna de estas = hard negative ideal)
# Ropa, telas, abrigos, pieles, accesorios de tela
CLASES_ROPA_Y_TELAS = {27, 75, 92, 95, 100, 131, 138, 145}

# Clases trampa originales (sofás, camas, etc.) — ya las tienes, pero person+sofa
# también es un hard negative útil (persona sentada en sofá peludo)
CLASES_TRAMPA_ORIGINALES = {4, 8, 19, 23, 28, 40, 57, 68, 75, 131}

# Unión: cualquier textura trampa que pueda co-ocurrir con persona
CLASES_CONTEXTO = CLASES_ROPA_Y_TELAS | CLASES_TRAMPA_ORIGINALES


def generar_csv_personas(
    base_dir: str = ADE20K_DIR,
    split_name: str = "training",
    output_csv: str = OUTPUT_CSV,
    requerir_contexto: bool = False,
):
    """
    Escanea ADE20K y guarda imágenes con personas.

    Args:
        base_dir          : Raíz de ADEChallengeData2016.
        split_name        : 'training' o 'validation'.
        output_csv        : Ruta de salida del CSV.
        requerir_contexto : Si True, solo guarda si además de persona hay
                            alguna clase de ropa/tela/trampa en la misma imagen.
                            Si False, toda imagen con persona sirve.
    """
    img_dir_abs = os.path.join(base_dir, "images", split_name)
    ann_dir_rel = os.path.join("annotations", split_name)

    if not os.path.exists(img_dir_abs):
        print(f"[ERROR] No se encontró: {img_dir_abs}")
        return

    archivos = [f for f in os.listdir(img_dir_abs) if f.endswith(".jpg")]
    total    = len(archivos)
    print(f"Escaneando {total} imágenes en '{split_name}' buscando personas...")
    if requerir_contexto:
        print("  Modo estricto: persona + textura trampa/ropa en la misma imagen.")
    else:
        print("  Modo amplio: cualquier imagen que contenga una persona.")

    encontrados = []

    for idx, filename in enumerate(archivos):
        ann_filename = filename.replace(".jpg", ".png")
        ann_path_rel = os.path.join(ann_dir_rel, ann_filename).replace("\\", "/")
        ann_path_abs = os.path.join(base_dir, ann_path_rel)

        if not os.path.exists(ann_path_abs):
            continue

        with Image.open(ann_path_abs) as mask_img:
            mascara = np.array(mask_img)
            clases  = set(np.unique(mascara).tolist())

        # Filtro 1: debe haber una persona
        if not clases.intersection(CLASE_PERSONA):
            continue

        # Filtro 2 (opcional): debe haber además alguna textura de contexto
        if requerir_contexto and not clases.intersection(CLASES_CONTEXTO):
            continue

        encontrados.append({
            "image"      : filename,
            "annotation" : ann_filename,
        })

        if (idx + 1) % 2000 == 0:
            print(f"  → {idx + 1}/{total} | Encontradas: {len(encontrados)}")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df = pd.DataFrame(encontrados)
    df.to_csv(output_csv, index=False)

    print(
        f"\n✅ CSV generado: {os.path.basename(output_csv)}\n"
        f"   {len(df)} imágenes de personas encontradas en ADE20K '{split_name}'."
    )
    return df


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ade20k_dir",  default=ADE20K_DIR)
    parser.add_argument("--split",       default="training")
    parser.add_argument("--output",      default=OUTPUT_CSV)
    parser.add_argument(
        "--strict", action="store_true",
        help="Solo incluir si también hay tela/ropa/trampa en la escena."
    )
    args = parser.parse_args()

    generar_csv_personas(
        base_dir=args.ade20k_dir,
        split_name=args.split,
        output_csv=args.output,
        requerir_contexto=args.strict,
    )


if __name__ == "__main__":
    main()
