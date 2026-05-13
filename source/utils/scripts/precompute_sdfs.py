"""
precompute_sdfs.py
==================
Pre-computa los Signed Distance Fields de TODAS las máscaras de Oxford
y los guarda en disco como archivos .npy.

Ejecutar UNA SOLA VEZ antes de entrenar:
    python utils/scripts/precompute_sdfs.py

Los SDFs se guardan en:
    data/oxford/sdfs/<basename>.npy

Cada archivo tiene shape (H, W) float32 normalizado en [-1, 1].
El DataLoader los carga directamente en __getitem__, sin bloquear la GPU.
"""

import os
import sys
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from concurrent.futures import ProcessPoolExecutor, as_completed

# ─── Paths ───────────────────────────────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
source_dir  = os.path.abspath(os.path.join(current_dir, "..", ".."))

MASKS_DIR   = os.path.join(source_dir, "data", "oxford", "masks")
SDF_OUT_DIR = os.path.join(source_dir, "data", "oxford", "sdfs")

# Mapeo de etiquetas Oxford: 0=Mascota, 2=Borde→Mascota, 1=Fondo
OXFORD_MAP = {0: 1.0, 2: 1.0, 1: 0.0}


# ─── Función de cómputo (ejecutada por worker) ───────────────────────────────

def _compute_and_save(mask_path: str, out_path: str) -> str:
    """Calcula el SDF de una máscara y lo guarda. Devuelve out_path si OK."""
    if os.path.exists(out_path):
        return f"[SKIP] {os.path.basename(out_path)}"

    raw  = np.array(Image.open(mask_path), dtype=np.int64)
    mask = np.zeros_like(raw, dtype=np.float32)
    for src, dst in OXFORD_MAP.items():
        mask[raw == src] = dst

    posmask = mask > 0.5

    if posmask.any():
        negmask   = ~posmask
        dist_out  = distance_transform_edt(negmask)
        dist_in   = distance_transform_edt(posmask)
        sdf_map   = dist_out - dist_in
        max_dist  = np.max(np.abs(sdf_map))
        if max_dist > 0:
            sdf_map /= max_dist
        sdf = sdf_map.astype(np.float32)
    else:
        # Máscara vacía → castigo máximo uniforme
        sdf = np.ones_like(mask, dtype=np.float32)

    np.save(out_path, sdf)
    return f"[OK]   {os.path.basename(out_path)}"


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(masks_dir: str = MASKS_DIR, sdf_dir: str = SDF_OUT_DIR, num_workers: int = 8):
    os.makedirs(sdf_dir, exist_ok=True)

    archivos = [f for f in os.listdir(masks_dir) if f.endswith(".png")]
    total    = len(archivos)
    print(f"Pre-computando SDFs para {total} máscaras de Oxford...")
    print(f"Destino: {sdf_dir}\n")

    tareas = []
    for fname in archivos:
        basename  = os.path.splitext(fname)[0]
        mask_path = os.path.join(masks_dir, fname)
        out_path  = os.path.join(sdf_dir, basename + ".npy")
        tareas.append((mask_path, out_path))

    completados = 0
    skipped     = 0
    errores     = 0

    # ProcessPoolExecutor para paralelizar los cálculos de Scipy en CPU
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futuros = {executor.submit(_compute_and_save, mp, op): (mp, op) for mp, op in tareas}

        for futuro in as_completed(futuros):
            try:
                resultado = futuro.result()
                if resultado.startswith("[SKIP]"):
                    skipped += 1
                else:
                    completados += 1
                # Feedback cada 500
                procesados = completados + skipped + errores
                if procesados % 500 == 0:
                    print(f"  {procesados}/{total} — OK: {completados} | Skip: {skipped} | Err: {errores}")
            except Exception as e:
                errores += 1
                print(f"  [ERROR] {futuros[futuro][0]}: {e}")

    print(f"\n✅ Finalizado — Calculados: {completados} | Ya existían: {skipped} | Errores: {errores}")


if __name__ == "__main__":
    # Opcionalmente recibe la ruta de máscaras como argumento
    masks_arg = sys.argv[1] if len(sys.argv) > 1 else MASKS_DIR
    sdf_arg   = sys.argv[2] if len(sys.argv) > 2 else SDF_OUT_DIR
    main(masks_dir=masks_arg, sdf_dir=sdf_arg)
