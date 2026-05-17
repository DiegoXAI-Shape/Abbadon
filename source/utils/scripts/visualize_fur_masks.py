import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def visualize_generated_masks():
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..', '..'))
    
    csv_path = os.path.join(project_root, "source", "labels", "hard_negatives_fur.csv")
    
    if not os.path.exists(csv_path):
        print(f"Error: No se encontró el CSV en {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    # Filtrar solo los que tienen mascota (es decir, que SAM generó una máscara)
    df_positives = df[df['is_adv'] == 0]
    
    if df_positives.empty:
        print("No se encontraron imágenes con mascotas (is_adv=0) en el CSV.")
        return
        
    print(f"Se encontraron {len(df_positives)} imágenes con mascotas generadas por SAM.")
    
    # Calcular cuántas filas necesitamos para la cuadrícula (máximo 5 para no saturar la pantalla)
    num_samples = min(len(df_positives), 5)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4 * num_samples))
    
    # Si solo hay una imagen, axes no es un arreglo 2D
    if num_samples == 1:
        axes = [axes]
        
    for i in range(num_samples):
        row = df_positives.iloc[i]
        
        # Las rutas en el CSV son relativas al project_root o source (depende de cómo se guardó)
        # En nuestro script pusimos 'source/...' así que unimos con project_root
        img_path = os.path.join(project_root, row['image'])
        mask_path = os.path.join(project_root, row['mask'])
        
        try:
            img = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')
            
            # Crear una superposición (Overlay) para ver qué tan bien segmentó
            img_np = np.array(img)
            mask_np = np.array(mask)
            
            overlay = img_np.copy()
            # Pintar de verde donde la máscara sea 0 (Mascota en formato Oxford)
            overlay[mask_np == 0] = [0, 255, 0] 
            
            axes[i][0].imshow(img)
            axes[i][0].set_title(f"Original: {os.path.basename(img_path)}")
            axes[i][0].axis('off')
            
            # Mostrar la máscara para humanos (0 se ve negro, 1 se ve casi negro, escalémoslo para visualizar)
            mask_vis = np.zeros_like(mask_np)
            mask_vis[mask_np == 0] = 255 # Blanco para mascota
            
            axes[i][1].imshow(mask_vis, cmap='gray')
            axes[i][1].set_title("Máscara Ground Truth (SAM)")
            axes[i][1].axis('off')
            
            axes[i][2].imshow(overlay)
            axes[i][2].set_title("Superposición (Verde = Mascota)")
            axes[i][2].axis('off')
            
        except Exception as e:
            print(f"Error procesando la imagen {img_path}: {e}")
            
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_generated_masks()
