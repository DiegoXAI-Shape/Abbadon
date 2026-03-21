import os
import torch
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode, functional
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2


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


#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#                                                            $
# Descomposición de funciones de Fourier para imágenes       $
#                                                            $
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


def get_fourier_lowpass(img_np, ratio = 0.40):
    """
    Aplica un filtro pasa-bajas en el dominio de Fourier a una imagen en escala de grises.

    Args:
        - img_np: Array NumPy 2D (escala de grises)
        - ratio: Porcentaje del tamaño de la imagen que se considera baja frecuencia

    Returns:
        Array NumPy normalizado a [0, 1]
    """
    h, w = img_np.shape
    freq_radius = int(min(h, w) * ratio)
    f = np.fft.fft2(img_np)
    fshift = np.fft.fftshift(f)

    mask = np.zeros((h, w), np.float32)
    cy, cx = h // 2, w // 2
    cv2.circle(mask, (cx, cy), freq_radius, 1, -1)

    fshift_filtered = fshift * mask
    img_lowpass = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift_filtered)))

    # Normalizamos a 0-1 para que coincida con ToTensor()
    return img_lowpass / 255.0


class CusDataset(Dataset):
    """
    Dataset con canal Fourier (4 canales: RGB + lowpass).
    """
    def __init__(self, dataframe, images_dir, masks_dir, images_transform=None, shape_img=(192, 192), is_train=True):
        self.df = dataframe
        self.img_dir = images_dir
        self.masks_dir = masks_dir
        self.shape = shape_img
        if images_transform:
            self.transformador = images_transform
        elif is_train:
            # Data augmentation solo para entrenamiento

            self.transformador = A.Compose([
                # 1. Geometría (Para que no importe dónde esté el objeto)
                A.Resize(height=shape_img[0], width=shape_img[1]),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=30, p=0.5),
                A.HorizontalFlip(p=0.5),

                # 2. Iluminación y Color (Para diferentes cámaras y climas)
                A.RandomBrightnessContrast(p=0.4),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
                # 3. Degradación de Imagen (Para que aprenda con "mala calidad")
                A.OneOf([
                    A.MotionBlur(p=0.2),
                    A.GaussianBlur(blur_limit=3, p=0.2),
                    A.GaussNoise(std_range=(0.1, 0.5), p=0.2),
                ], p=0.4),

                # 4. Oclusiones 
                A.CoarseDropout(
                    num_holes_range=(1, 8),
                    hole_height_range=(0.15, 0.3),
                    hole_width_range=(0.15, 0.3),
                    p=0.4
                ), 
                
                # 5. Normalización
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ], additional_targets={'fourier': 'mask'})
        else:
            # Sin augmentation para validación y test
            self.transformador = A.Compose([
                A.Resize(height=shape_img[0], width=shape_img[1]),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ], additional_targets={'fourier': 'image'})

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_filename = self.df.iloc[index, 0]
        mask_filename = self.df.iloc[index, 1]

        img_dir = os.path.join(self.img_dir, img_filename)
        mask_dir = os.path.join(self.masks_dir, mask_filename)

        img = cv2.imread(img_dir)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_dir, cv2.IMREAD_GRAYSCALE)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = get_fourier_lowpass(gray)

        #Data augmentation
        resultados = self.transformador(
            image = img,
            mask = mask,
            fourier = gray)
        img = resultados['image'] #Tensor [3, H, W]
        mask = resultados['mask']
        fourier = resultados['fourier']

        if fourier.ndim == 2:  # (H, W) → (1, H, W)
            fourier = fourier.unsqueeze(0)
        fourier = fourier.float()
        fourier = (fourier - fourier.mean()) / (fourier.std() + 1e-6)

        img = torch.cat((img, fourier), dim=0)
        mask = mask.long()
        mask = torch.clamp(mask, 0, 2)
        
        return img, mask


#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#                                                                                $
#   Generación de dataloaders con split train/val/test                           $
#                                                                                $
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


def get_dataloaders(batch_size=32, shuffle=True, num_workers=4, pin_memory=True, shape_img=(192, 192),
                    gold_weight=3.0):
    """
    Crea los dataloaders de entrenamiento, validación y test.
    
    Args:
        gold_weight: Peso de sobre-muestreo para las imágenes pseudo-etiquetadas (is_gold=True).
                     Por ejemplo, 3.0 = las imágenes de mundo real se samplearán 3x más que las normales.
    """
    df = pd.read_csv("labels/dataset_generated.csv")
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(val_df, test_size=0.1, random_state=42)

    train_ds = CusDataset(train_df.reset_index(drop=True), "data/image/images", "data/image/masks", shape_img=shape_img, is_train=True)
    val_ds   = CusDataset(val_df.reset_index(drop=True),   "data/image/images", "data/image/masks", shape_img=shape_img, is_train=False)
    test_ds  = CusDataset(test_df.reset_index(drop=True),  "data/image/images", "data/image/masks", shape_img=shape_img, is_train=False)

    # ── WeightedRandomSampler ─────────────────────────────────────────────
    # Peso 1.0 para imágenes del dataset original, gold_weight para pseudo-etiquetas
    if 'is_gold' in train_df.columns:
        weights = train_df['is_gold'].map({True: gold_weight, False: 1.0}).fillna(1.0).tolist()
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                                  num_workers=num_workers, pin_memory=pin_memory)
        print(f"[i] WeightedRandomSampler activo: gold_weight={gold_weight}x")
        gold_n = sum(1 for w in weights if w == gold_weight)
        print(f"    ├── Normales  : {len(weights) - gold_n}")
        print(f"    └── is_gold   : {gold_n}")
    else:
        # Fallback si el CSV no tiene la columna is_gold
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle,
                                  num_workers=num_workers, pin_memory=pin_memory)

    val_loader  = DataLoader(val_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return [train_loader, val_loader, test_loader]


class UnlabeledDataset(Dataset):
    """
    Dataset para imágenes sin etiquetas (para inferencia o pseudo-labeling).
    Genera automáticamente el canal Fourier.
    """
    def __init__(self, images_dir, shape_img=(384, 384), augment=False):
        self.img_dir = images_dir
        self.shape = shape_img
        self.augment = augment
        self.file_list = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Pipeline Base (Obligatorio)
        self.base_transform = A.Compose([
            A.Resize(height=shape_img[0], width=shape_img[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], additional_targets={'fourier': 'image'})

        # Pipeline de Aumento (Opcional, para testear robustez)
        self.aug_transform = None
        if augment:
            self.aug_transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Affine(shift_limit=0.1, scale_limit=0.15, rotate_limit=30, p=0.5),
            ], additional_targets={'fourier': 'image'})

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_filename = self.file_list[index]
        img_path = os.path.join(self.img_dir, img_filename)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Canal Fourier
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        fourier_np = get_fourier_lowpass(gray)

        # Aplicar aumento si está activado
        if self.augment and self.aug_transform:
            res_aug = self.aug_transform(image=img, fourier=fourier_np)
            img = res_aug['image']
            fourier_np = res_aug['fourier']

        # Transformaciones base (Resize, Normalize, ToTensor)
        resultados = self.base_transform(image=img, fourier=fourier_np)
        img_t = resultados['image']
        fourier_t = resultados['fourier']

        if fourier_t.ndim == 2:
            fourier_t = fourier_t.unsqueeze(0)
        fourier_t = fourier_t.float()
        fourier_t = (fourier_t - fourier_t.mean()) / (fourier_t.std() + 1e-6)

        # Concatenar canal Fourier (4 canales en total)
        full_img = torch.cat((img_t, fourier_t), dim=0)

        return full_img, img_filename


def get_unlabeled_dataloader(images_dir, batch_size=16, num_workers=4, shape_img=(384, 384), augment=False):
    ds = UnlabeledDataset(images_dir, shape_img=shape_img, augment=augment)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
