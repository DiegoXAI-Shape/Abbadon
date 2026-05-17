import matplotlib.pyplot as plt
import torch
import cv2
import os
import numpy as np
from PIL import Image
from torchvision import transforms
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(current_dir, '..', 'models')

sys.path.append(dataset_dir)

def postprocessMask(out_network, kernel:int, umbral:float=0.5):
    """
    Args:
        - out_network: Salida de la red neuronal, en este caso Daowa-Maad. Con la forma: [Batch, Channels, Height, Weight]
        - kernel: El tamaño 'pincel'
        - umbral: El punto de corte para la probabilidad (default 0.5).
    """
    if torch.is_tensor(out_network) and (kernel % 2 != 0):
        # Primero debemos quitarle el espía al tensor
        out_network = out_network.detach()
        # Bajarlo de la CPU
        out_network = out_network.cpu()

        if out_network.shape[1] == 1:
            # Aplicar Sigmoide para obtener probabilidades reales
            probs = torch.sigmoid(out_network)
            
            # Usar el umbral calibrado (ej. 0.80)
            pred_mascota = (probs > umbral).squeeze() # [Height, Width]
            
            # Mapear a 0 (Mascota) y 1 (Fondo) para el visualizador
            indices = torch.where(pred_mascota, torch.tensor(0), torch.tensor(1))
            mascara_animal = pred_mascota.numpy()
        else:
            # Modelo antiguo de 3 canales
            out_network = torch.argmax(out_network, dim = 1)
            indices = out_network.squeeze(0)
            mascara_animal = (indices == 0).numpy()

        mascara_uint8 = (mascara_animal * 255).astype(np.uint8)

        # Removemos el post-proceso de CV2 para ver la predicción pura
        return indices.numpy()

def prediccionPrueba(modelo, dir_path:str, img_dir:str, device, umbral:float=0.5):
    img_path = os.path.join(dir_path, img_dir)
    img_open = Image.open(img_path).convert('RGB').resize((384, 384))
    
    tf = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])
    img = tf(img_open)
    
    img = img.to(device).unsqueeze(0)
    modelo = modelo.to(device)
    modelo.eval()
    
    with torch.no_grad():
        output = modelo(img)
    
    indices = postprocessMask(output, kernel=1, umbral=umbral)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    ax[0].imshow(img_open)
    ax[0].set_title("Input Original")
    ax[0].axis('off')
    
    from matplotlib.colors import ListedColormap
    cmap_binario = ListedColormap(['#1f77b4', '#2ca02c'])  # azul=mascota, verde=fondo
    cax = ax[1].imshow(indices, cmap=cmap_binario, vmin=0, vmax=1)
    ax[1].set_title(f"Predicción (Umbral={umbral:.2f})")
    ax[1].axis('off')
    cbar = plt.colorbar(cax, ax=ax[1], ticks=[0, 1], fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels(['Mascota (0)', 'Fondo (1)']) 
    
    plt.tight_layout()
    plt.show()

def comparar_modelos(modelo_base, modelo_adv, dir_path:str, img_dir:str, device, umbral:float=0.5):
    """
    Compara visualmente el modelo antes y después del entrenamiento adversarial.
    """
    img_path = os.path.join(dir_path, img_dir)
    img_open = Image.open(img_path).convert('RGB').resize((384, 384))
    
    tf = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])
    img = tf(img_open).to(device).unsqueeze(0)
    
    modelo_base = modelo_base.to(device)
    modelo_adv = modelo_adv.to(device)
    modelo_base.eval()
    modelo_adv.eval()
    
    with torch.no_grad():
        out_base = modelo_base(img)
        out_adv = modelo_adv(img)
    
    indices_base = postprocessMask(out_base, kernel=1, umbral=0.5) # El base lo dejamos en 0.5 para comparar justo
    indices_adv = postprocessMask(out_adv, kernel=1, umbral=umbral)

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    
    ax[0].imshow(img_open)
    ax[0].set_title("Input Original")
    ax[0].axis('off')
    
    from matplotlib.colors import ListedColormap
    cmap_binario = ListedColormap(['#1f77b4', '#2ca02c'])
    
    ax[1].imshow(indices_base, cmap=cmap_binario, vmin=0, vmax=1)
    ax[1].set_title("Modelo Base")
    ax[1].axis('off')
    
    ax[2].imshow(indices_adv, cmap=cmap_binario, vmin=0, vmax=1)
    ax[2].set_title("Modelo Adversarial")
    ax[2].axis('off')
    
    plt.tight_layout()
    plt.show()
