import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

class MendicantDataset(Dataset):
    """
    Dataset para Mendicant Bias (Clasificación Binaria: Cat vs Dog).
    Carga imágenes desde el disco y aplica augmentations básicas.
    No interactúa con Daowa-maad aquí (eso ocurre en la GPU durante el bucle de entrenamiento).
    """
    def __init__(self, csv_file, img_dir, is_train=True):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.is_train = is_train
        
        # Mapeo de etiquetas: Gato=0, Perro=1
        self.label_map = {'Cat': 0, 'Dog': 1}
        
        # Filtramos posibles errores o archivos faltantes si es necesario
        # (Se asume que dataset_mendicantV3.csv ya está limpio)
        
        # Augmentations
        if self.is_train:
            self.transform = A.Compose([
                A.Resize(384, 384),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.3),
                # Normalización estándar de ImageNet (Esperada por ConvNeXt)
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(384, 384),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_name = row['filename']
        # Limpieza robusta del label para evitar el error de 50% de accuracy por desajuste de strings
        label_str = str(row['label']).capitalize().strip()
        
        # Rutas
        img_path = os.path.join(self.img_dir, label_str, img_name)
        mask_path = os.path.join("source/data/PetMasks", label_str, img_name.replace('.jpg', '.png'))
        
        # Leer imagen
        image = cv2.imread(img_path)
        if image is None:
            image = np.zeros((384, 384, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (384, 384)) # Igualar dimensiones con la máscara cacheada
            
        # Leer máscara cacheada
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros((384, 384), dtype=np.uint8)
            
        # Aplicar transformaciones SIMULTÁNEAS a imagen y máscara (¡El detalle brillante que mencionaste!)
        augmented = self.transform(image=image, mask=mask)
        image_tensor = augmented['image']
        
        # La máscara vuelve a ser un tensor de shape [1, 384, 384] y sus valores de vuelta a [0.0, 1.0]
        mask_tensor = augmented['mask'].unsqueeze(0).float() / 255.0
        
        # Convertir etiqueta a numérico
        label_idx = self.label_map.get(label_str, 0)
        label_tensor = torch.tensor(label_idx, dtype=torch.long)
        
        return image_tensor, mask_tensor, label_tensor

# Función helper para obtener los DataLoaders
def get_mendicant_dataloaders(csv_train, csv_val, img_dir, batch_size=16, num_workers=4):
    from torch.utils.data import DataLoader
    
    train_ds = MendicantDataset(csv_train, img_dir, is_train=True)
    val_ds = MendicantDataset(csv_val, img_dir, is_train=False)
    
    # IMPORTANTE: drop_last=True para el train_loader ayuda a evitar errores 
    # de BatchNorm si el último batch tiene tamaño 1.
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True, drop_last=True)
                              
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True)
                            
    return {"train": train_loader, "val": val_loader}
