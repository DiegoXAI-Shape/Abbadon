import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def main():
    base_dir = r"C:\Users\PC\Desktop\Abbadon prueba SAM\source\data\ADEChallengeData2016"
    img_dir = os.path.join(base_dir, "images", "training")
    ann_dir = os.path.join(base_dir, "annotations", "training")
    
    if not os.path.exists(img_dir):
        print("No se encontró el dataset en:", img_dir)
        return

    # Las clases trampa de ADE20K según la documentación
    # 4: floor, 8: bed, 19: curtain, 23: sofa, 28: carpet/rug, 
    # 40: cushion, 57: pillow, 68: mat, 75: cloth, 131: blanket
    clases_trampa = {4, 8, 19, 23, 28, 40, 57, 68, 75, 131}
    
    todas_las_imagenes = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
    random.shuffle(todas_las_imagenes)
    
    print(f"Buscando aleatoriamente en tu base local de {len(todas_las_imagenes)} imágenes...")
    
    indices_validos = []
    
    # Escaneamos imágenes al azar hasta toparnos con 4 que nos sirvan
    for img_filename in todas_las_imagenes:
        ann_filename = img_filename.replace('.jpg', '.png')
        ann_path = os.path.join(ann_dir, ann_filename)
        
        if not os.path.exists(ann_path): continue
            
        with Image.open(ann_path) as mask_img:
            mascara_numpy = np.array(mask_img)
            clases_en_imagen = set(np.unique(mascara_numpy))
            
            # Vemos si esta imagen tiene alguna de nuestras trampas texturales
            interseccion = clases_en_imagen.intersection(clases_trampa)
            if interseccion:
                img_path = os.path.join(img_dir, img_filename)
                indices_validos.append((img_path, ann_path, list(interseccion)))
                
        # Cuando tengamos 4, nos detenemos para no tardar tanto
        if len(indices_validos) == 4:
            break
            
    if not indices_validos:
        print("No se encontraron imágenes (Raro).")
        return
        
    print(f"¡Se encontraron 4 imágenes locales!")
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("ADE20K Local: Tus Hard Negatives (Sofás, Cortinas, Camas, Cobijas)", fontsize=18)
    
    for i, (img_path, ann_path, clases_encontradas) in enumerate(indices_validos):
        img = np.array(Image.open(img_path))
        mask = np.array(Image.open(ann_path))
        
        # Filtramos la máscara para que SOLO brille donde exista la clase adversaria
        mascara_trampas_solo = np.isin(mask, list(clases_trampa)).astype(np.float32)
        
        # Original
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"{os.path.basename(img_path)}\nIDs presentes: {clases_encontradas}", fontsize=10)
        axes[0, i].axis('off')
        
        # Segmentación Aislada
        axes[1, i].imshow(mascara_trampas_solo, cmap='magma')
        axes[1, i].set_title("Máscara Enemiga Aislada (1 = Trampa)")
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
