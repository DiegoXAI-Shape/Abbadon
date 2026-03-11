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

from datasets import get_fourier_lowpass

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
        print(out_network.device) #Saldrá device(type='cuda', index=0)

        out_network = out_network.cpu()
        print(out_network.device)

        #Ahora debemos quitar los canales, pues a ver... vamo' a hacerle un poco de ingeniería inversa pq hace 2 semanas no toco este código JAJAJ

        """
        Vale, ya entendí, devuelve 3 canales (son 3 laminas en un cubo, básicamente...) que son una para el fondo, el animal y el borde, así que podemos usar torch.argmax para colapsar columnas
        """
        out_network = torch.argmax(out_network, dim = 1)
        print(out_network.shape) #torch.Size([1, 192, 192])

        #Ahora usaremos squeeze para quitar el 1 al final y ahora si que funcione CV2
        indices = out_network.squeeze(0)

        mascara_animal = (indices == 0).numpy()
        mascara_uint8 = (mascara_animal * 255).astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, kernel))
        mask_limpia = cv2.morphologyEx(mascara_uint8, cv2.MORPH_OPEN, kernel)
        
        print(mask_limpia.size)

        return indices.numpy(), mask_limpia

def prediccionPrueba(modelo, dir_path:str, img_dir:str, device):
    img_path = os.path.join(dir_path, img_dir)
    img_open = Image.open(img_path).convert('RGB').resize((192, 192))
    
    # Canal Fourier (igual que en tu CusDataset)
    img_fourier_np = np.array(img_open.convert('L'))
    img_lowpass = get_fourier_lowpass(img_fourier_np, 50)
    img_tensor_lowpass = torch.from_numpy(img_lowpass).unsqueeze(0).float()
    
    tf = transforms.Compose([
        transforms.Resize((192, 192)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])
    img = tf(img_open)
    
    # Concatenar RGB (3) + Fourier (1) = 4 canales
    img = torch.cat((img, img_tensor_lowpass), dim=0)

    img = img.to(device)
    img = img.unsqueeze(0)
    modelo = modelo.to(device)
    modelo.eval()
    with torch.no_grad():
        output = modelo(img)
    
    indices, mascara = postprocessMask(output, kernel = 5)

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    
    # A) Imagen Original
    ax[0].imshow(img_open)
    ax[0].set_title("Input Original")
    ax[0].axis('off')
    
    # B) Salida Cruda de la Red (Multiclase)
    cax = ax[1].imshow(indices, cmap='jet', vmin=0, vmax=2)
    ax[1].set_title("Red Neuronal (Crudo)")
    ax[1].axis('off')
    cbar = plt.colorbar(cax, ax=ax[1], ticks=[0, 1, 2], fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels(['Mascota (0)', 'Fondo(1)', 'Borde (2)']) 
    
    # C) Salida Limpia (CV2)
    ax[2].imshow(mascara, cmap='gray')
    ax[2].set_title("Post-Proceso")
    ax[2].axis('off')
    
    plt.tight_layout()
    plt.show()

    return output