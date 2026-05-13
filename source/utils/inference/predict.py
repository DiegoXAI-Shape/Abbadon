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

def postprocessMask(out_network, kernel:int):
    """
    Args:
        - out_network: Salida de la red neuronal, en este caso Daowa-Maad. Con la forma: [Batch, Channels, Height, Weight]

        - kernel: El tamaño 'pincel' por así decirlo, siempre tiene que ser impar por mera geometría matemática, además de que si el número es muy alto, puedes perder detalles finos.
    """
    if torch.is_tensor(out_network) and (kernel % 2 != 0):
        #Primero tenemos que convertir esa máscara a un arreglo de NumPy para CV2

        #> Aquí primero debemos quitarle el espía al tensor, para que PyTorch deje de rastrearlo y esto se usa para el AutoGrad, pero en este caso no lo ocupamos
        out_network = out_network.detach()

        #AHORA debemos bajarlo de la CPU
        out_network = out_network.cpu()

        # Ahora debemos quitar los canales.
        # Si el modelo fue entrenado con 1 canal (BCEWithLogits), el output es simplemente Logits.
        if out_network.shape[1] == 1:
            # Logits > 0.0 equivale a probabilidad > 0.5 (Mascota)
            pred_mascota = (out_network > 0.0).squeeze() # [Height, Width]
            
            # Para mantener compatibilidad con tu código de visualización viejo (donde Mascota = 0, Fondo = 1):
            indices = torch.where(pred_mascota, torch.tensor(0), torch.tensor(1))
            mascara_animal = pred_mascota.numpy()
        else:
            # Modelo antiguo de 3 canales
            out_network = torch.argmax(out_network, dim = 1)
            indices = out_network.squeeze(0)
            mascara_animal = (indices == 0).numpy()

        mascara_uint8 = (mascara_animal * 255).astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, kernel))
        mask_limpia = cv2.morphologyEx(mascara_uint8, cv2.MORPH_CLOSE, kernel)

        return indices.numpy(), mask_limpia

def prediccionPrueba(modelo, dir_path:str, img_dir:str, device):
    img_path = os.path.join(dir_path, img_dir)
    img_open = Image.open(img_path).convert('RGB').resize((384, 384))
    
    # Canal Fourier (igual que en tu CusDataset)
    #img_fourier_np = np.array(img_open.convert('L'))
    #img_lowpass = get_fourier_lowpass(img_fourier_np, 50)
    #img_tensor_lowpass = torch.from_numpy(img_lowpass).unsqueeze(0).float()
    
    tf = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])
    img = tf(img_open)
    
    # Concatenar RGB (3) + Fourier (1) = 4 canales
    #img = torch.cat((img, img_tensor_lowpass), dim=0)

    img = img.to(device)
    img = img.unsqueeze(0)
    modelo = modelo.to(device)
    modelo.eval()
    with torch.no_grad():
        output = modelo(img)
    
    indices, mascara = postprocessMask(output, kernel = 15)

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    
    # A) Imagen Original
    ax[0].imshow(img_open)
    ax[0].set_title("Input Original")
    ax[0].axis('off')
    
    # B) Salida Cruda de la Red (Multiclase)
    from matplotlib.colors import ListedColormap
    cmap_binario = ListedColormap(['#1f77b4', '#2ca02c'])  # azul=mascota, verde=fondo
    cax = ax[1].imshow(indices, cmap=cmap_binario, vmin=0, vmax=1)
    ax[1].set_title("Red Neuronal (Binario)")
    ax[1].axis('off')
    cbar = plt.colorbar(cax, ax=ax[1], ticks=[0, 1], fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels(['Mascota (0)', 'Fondo (1)']) 
    
    # C) Salida Limpia (CV2)
    ax[2].imshow(mascara, cmap='gray')
    ax[2].set_title("Post-Proceso")
    ax[2].axis('off')
    
    plt.tight_layout()
    plt.show()