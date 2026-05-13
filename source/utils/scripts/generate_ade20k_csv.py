import os
import pandas as pd
import numpy as np
from PIL import Image

def generar_csv(base_dir, split_name, output_csv):
    """
    Genera un CSV con rutas relativas a las carpetas de imágenes y anotaciones 
    de ADE20K (ej. images/training/xxx.jpg).
    """
    img_dir_rel = os.path.join("images", split_name)
    ann_dir_rel = os.path.join("annotations", split_name)
    
    img_dir_abs = os.path.join(base_dir, img_dir_rel)
    
    if not os.path.exists(img_dir_abs):
        print(f"Directorio no encontrado: {img_dir_abs}")
        return
        
    archivos = []
    
    for filename in os.listdir(img_dir_abs):
        if filename.endswith(".jpg"):
            ann_filename = filename.replace(".jpg", ".png")
            
            # Se usa '/' para mantener compatibilidad universal de rutas relativas
            img_path_rel = os.path.join(img_dir_rel, filename).replace('\\', '/')
            ann_path_rel = os.path.join(ann_dir_rel, ann_filename).replace('\\', '/')
            
            # Verificamos que la máscara físicamente exista antes de anotarla
            if os.path.exists(os.path.join(base_dir, ann_path_rel)):
                archivos.append({
                    "image": filename,
                    "annotation": ann_filename
                })
                
    df = pd.DataFrame(archivos)
    df.to_csv(output_csv, index=False)
    print(f"Generado {os.path.basename(output_csv)} con {len(df)} filas.")


def generar_csv_adversarial(base_dir, split_name, output_csv):
    """
    Escanea archivos PNG para buscar texturas adversariales de ADE20K.
    Solo guarda en el CSV si la máscara contiene Sillones, Alfombras, Camas, etc.
    """
    img_dir_rel = os.path.join("images", split_name)
    ann_dir_rel = os.path.join("annotations", split_name)
    
    img_dir_abs = os.path.join(base_dir, img_dir_rel)
    
    if not os.path.exists(img_dir_abs):
        print(f"Directorio no encontrado: {img_dir_abs}")
        return
        
    # Las clases trampa de ADE20K según la documentación
    clases_trampa = {4, 8, 19, 23, 28, 40, 57, 68, 75, 131}
    archivos = []
    
    lista_archivos = [f for f in os.listdir(img_dir_abs) if f.endswith(".jpg")]
    total = len(lista_archivos)
    print(f"Escaneando {total} imágenes en {split_name} (esto tomará ~2 minutos)...")
    
    for idx, filename in enumerate(lista_archivos):
        ann_filename = filename.replace(".jpg", ".png")
        ann_path_rel = os.path.join(ann_dir_rel, ann_filename).replace('\\', '/')
        ann_path_abs = os.path.join(base_dir, ann_path_rel)
        
        if os.path.exists(ann_path_abs):
            # Procesamiento Matemático Rápido (Solo abrimos la máscara)
            with Image.open(ann_path_abs) as mask_img:
                mascara_numpy = np.array(mask_img)
                # Obtenemos TODOS los IDs únicos que contiene la casa/escena
                clases_en_imagen = set(np.unique(mascara_numpy))
                
                # Si intersecciona con nuestra lista de texturas peligrosas, aprueba.
                if clases_en_imagen.intersection(clases_trampa):
                    archivos.append({
                        "image": filename,
                        "annotation": ann_filename
                    })
        
        # Feedback visual cada 2000 imágenes
        if (idx+1) % 2000 == 0:
            print(f"  -> Progreso: {idx+1}/{total} | Trampas Encontradas hasta ahora: {len(archivos)}")
                
    df = pd.DataFrame(archivos)
    df.to_csv(output_csv, index=False)
    print(f"Finalizado. Extracto adversarial guardado en {os.path.basename(output_csv)} con {len(df)} filas.")


def main():
    # Basado en la ruta que vi que pusiste en el view_ade20k.py
    base_dir = r"C:\Users\PC\Desktop\Abbadon prueba SAM\source\data\ADEChallengeData2016"
    out_dir = r"C:\Users\PC\Desktop\Abbadon prueba SAM\source\labels"
    
    print("Iniciando escaneo de dataset ADE20K...")
    
    generar_csv(
        base_dir, 
        "training", 
        os.path.join(out_dir, "ade20k_train.csv")
    )
    
    # NUEVO: Generar CSV de Entrenamiento Adversarial
    generar_csv_adversarial(
        base_dir, 
        "training", 
        os.path.join(out_dir, "ade20k_adversarial_train.csv")
    )

if __name__ == "__main__":
    main()
