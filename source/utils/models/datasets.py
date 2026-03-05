import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


class CustomDS_Med(Dataset):
    def __init__(self, dataframe, images_dir, transform=None, target_transform=None): #Parámetros modificables
        self.labels = dataframe #Le pasamos el DataFrame que ya habíamos hecho sin el problema de Filename collision
        self.directory = images_dir #Ruta al directorio de imágenes
        if transform:
            self.transform_data = transform #Transformador de la imagen
        else:
            self.transform_data = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
            ])
        self.target_transform = target_transform #Transformador de etiquetas
        self.mapping = {"Cat": 0, "Dog": 1} #Mapeo de etiquetas

    def __len__(self):
        return len(self.labels) #Solamente obtiene la cantidad de elementos

    def __getitem__(self, idx):
        label_df = self.labels.iloc[idx, 1] #Obtiene el elemento de la fila (supongamos idx = 0) 0 y columna 1, lo que sería el elemento de la posición [0, 1] si vieramos al DataFrame como una matriz
        img_path = os.path.join(self.directory, self.labels.iloc[idx, 1], self.labels.iloc[idx, 0]) #Generamos el path de donde se encuentran las imágenes (Notese que solo accedo al elemento, porque ya no hay distinción entre imágenes hablando de las carpetas)
        img = Image.open(img_path).convert('RGB') #Abrimos la imagen con PIL para saber si la imagen no está corrupta o dañada y si esto es cierto, la convertimos a RGB para tener 3 canales en el tensor
        img_label = self.mapping[label_df] #Mapeamos la etiqueta, es decir, si la etiqueta es Cat, la convierte en 0 o si es Dog la convierte en 1
        if self.transform_data: #Revisamos si se pasaron transformadores para los datos
            img = self.transform_data(img)
        if self.target_transform: #Lo mismo, pero para etiquetado
            img_label = self.target_transform(img_label)
        return img, img_label, self.labels.iloc[idx, 0] #Retornamos la imagen y su etiqueta


class CustomDS(Dataset):
    def __init__(self, dataframe, img_dir, mask_dir, transformador = None):
        self._dataframe = dataframe
        self._img_dir = img_dir
        self._mask_dir = mask_dir
        self._transformador = transforms.Compose([
            transforms.Resize((192, 192)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self._dataframe)

    def __getitem__(self, idx):
        image_path = self._dataframe.iloc[idx, 0]
        mask_path = self._dataframe.iloc[idx, 1]
        img_path = os.path.join(self._img_dir, image_path)
        maskPath = os.path.join(self._mask_dir, mask_path)
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(maskPath)

        img = self._transformador(img)
        mask = transforms.functional.resize(mask, (192, 192), interpolation=InterpolationMode.NEAREST) 

        mask_np = np.array(mask)
        mask_tensor = torch.from_numpy(mask_np).long()
        mask_tensor = torch.clamp(mask_tensor, 0, 2)
        return img, mask_tensor
