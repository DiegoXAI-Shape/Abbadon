import os
import torch
import sys
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms.functional as F
from scipy.ndimage import distance_transform_edt

def compute_sdf_single_mask(mask_tensor):
    """
    Calcula el SDF asíncronamente usando Scipy sobre un solo hilo num_worker.
    Previene el congelamiento masivo de la GPU.
    """
    mask_np = mask_tensor.squeeze(0).numpy()
    sdf = np.zeros_like(mask_np)
    posmask = mask_np > 0.5
    if posmask.any():
        negmask = ~posmask
        dist_out = distance_transform_edt(negmask)
        dist_in = distance_transform_edt(posmask)
        sdf_map = dist_out - dist_in
        max_dist = np.max(np.abs(sdf_map))
        if max_dist > 0:
            sdf_map = sdf_map / max_dist
        sdf = sdf_map
    else:
        sdf = np.ones_like(mask_np)
    return torch.from_numpy(sdf).unsqueeze(0).float()

current_dir = os.path.dirname(os.path.abspath(__file__))

source_dir = os.path.abspath(os.path.join(current_dir, "..", "..")) 
if source_dir not in sys.path:
    sys.path.append(source_dir)

from utils.models.copy_paste_augmentation import get_lsj_transform, get_augmentation_ade20k

class CustomDistillationDataset(Dataset):
    def __init__(self, df, imgs_dir, human_masks_dir, sam_logits_dir, transform=None, img_size=(256, 256)):
        self.df = df
        self.imgs_dir = imgs_dir
        self.human_masks_dir = human_masks_dir
        self.sam_logits_dir = sam_logits_dir
        self.img_size = img_size
        self.lsj_transformador = get_lsj_transform(img_size)
        
        if transform is None:
            # Transformador Básico/Estándar (Menos agresivo)
            self.transformador = A.Compose([
                A.RandomResizedCrop(
                    size=(img_size[0], img_size[1]),
                    scale=(0.5, 1.0),
                    ratio=(0.8, 1.2),
                    p=0.5
                ),
                A.Resize(height=img_size[0], width=img_size[1]),
                A.HorizontalFlip(p=0.5),
                A.Affine(
                    scale=(0.9, 1.1),
                    translate_percent=(-0.05, 0.05),
                    rotate=(-15, 15),
                    p=0.4
                ),
                A.RandomBrightnessContrast(p=0.7),
                A.HueSaturationValue(
                    hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3
                ),
                A.OneOf([
                    A.MotionBlur(p=0.2),
                    A.GaussianBlur(blur_limit=3, p=0.2),
                    A.GaussNoise(p=0.2), 
                ], p=0.3),
                A.CoarseDropout(
                    num_holes_range=(1, 5),
                    hole_height_range=(int(0.05 * img_size[0]), int(0.15 * img_size[0])),
                    hole_width_range=(int(0.05 * img_size[1]), int(0.15 * img_size[1])),
                    p=0.3
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        else:
            self.transformador = transform

    def __len__(self):
        return len(self.df)

    def _get_raw_data(self, idx):
        img_name = self.df.iloc[idx]['file']
        base_name = os.path.splitext(img_name)[0]
        
        # Paths
        img_path = os.path.join(self.imgs_dir, img_name)
        human_mask_path = os.path.join(self.human_masks_dir, base_name + ".png")
        # Descomentar cuando se tenga el dataset de logits de SAM
        # sam_logit_path = os.path.join(self.sam_logits_dir, base_name + "_SAM.npy") 
        
        # 1. Cargar la imagen original
        image = Image.open(img_path).convert("RGB")
        image_uint8 = np.array(image) # H, W, 3
        H_orig, W_orig = image_uint8.shape[:2]

        # 2. Cargar Máscara Humana real (Si existe)
        if os.path.exists(human_mask_path):
            human_mask = Image.open(human_mask_path)
            human_mask_np = np.array(human_mask, dtype=np.int64)
            
            binary_human_mask = np.zeros_like(human_mask_np, dtype=np.float32)
            binary_human_mask[human_mask_np == 0] = 1.0 # Mascota
            binary_human_mask[human_mask_np == 2] = 1.0 # Borde -> Mascota
            binary_human_mask[human_mask_np == 1] = 0.0 # Fondo
        else:
            # Es un Pseudo-label: Asignamos array temporal, lo llenaremos con la de SAM enseguida
            binary_human_mask = np.zeros((H_orig, W_orig), dtype=np.float32)
        
        # 3. Cargar Logits SAM (Por defecto SAM siempre escupe 256x256 logits)
        # Descomentar cuando se tenga el dataset de logits de SAM
        # sam_mask_np = np.load(sam_logit_path).astype(np.float32)
        
        # 3.1 ¡CORRECCIÓN CRÍTICA! Albumentations requiere que todas las entradas tengan el mismo (H, W) inicial.
        # Descomentar cuando se tenga el dataset de logits de SAM
        # if sam_mask_np.shape != (H_orig, W_orig):
        #     sam_tensor = torch.from_numpy(sam_mask_np).unsqueeze(0).unsqueeze(0) # [1, 1, 256, 256]
        #     # Interpolación bilineal para no dañar los logits continuos
        #     sam_tensor = torch.nn.functional.interpolate(sam_tensor, size=(H_orig, W_orig), mode='bilinear', align_corners=False)
        #     sam_mask_np = sam_tensor.squeeze().numpy()
            
        # 3.2 Completar Pseudo-label si falta máscara humana
        # Descomentar cuando se tenga el dataset de logits de SAM
        # if not os.path.exists(human_mask_path):
        #     binary_human_mask = (sam_mask_np > 0.0).astype(np.float32)

        return image_uint8, binary_human_mask

    def __getitem__(self, idx):
        # 1. Cargar A
        image_uint8_A, binary_human_mask_A = self._get_raw_data(idx)
        
        # 2. Lógica Probabilística de Copy-Paste
        # 'rand() < 0.6' significa 60% de probabilidad de caer en el Copy-Paste + LSJ Fuerte.
        apply_copy_paste = (np.random.rand() < 0.6) and self.lsj_transformador is not None

        if apply_copy_paste:
            # Escoger Imagen B al azar
            random_idx = np.random.randint(0, len(self.df))
            image_uint8_B, binary_human_mask_B = self._get_raw_data(random_idx)

            # Aumentar ambas de forma independiente usando LSJ Agresivo
            aug_A = self.lsj_transformador(image=image_uint8_A, mask=binary_human_mask_A)
            aug_B = self.lsj_transformador(image=image_uint8_B, mask=binary_human_mask_B)

            # Albumentations (ToTensorV2) ya arrojó [C, H, W] para las imágenes y [H, W] para las máscaras
            img_tensor_A = aug_A['image']
            mask_tensor_A = aug_A['mask']
            # sam_tensor_A = aug_A['masks'][0]

            img_tensor_B = aug_B['image']
            mask_tensor_B = aug_B['mask']
            #sam_tensor_B = aug_B['masks'][0]

            # Encontrar dónde existe la mascota B (umbral 0.5 ya que nuestra máscara es 0.0 o 1.0)
            mask_b_bool = mask_tensor_B > 0.5

            # Pegar Mascota B en Imagen A
            img_tensor_A[:, mask_b_bool] = img_tensor_B[:, mask_b_bool]
            mask_tensor_A[mask_b_bool] = 1.0
            # sam_tensor_A[mask_b_bool] = sam_tensor_B[mask_b_bool]

            image_tensor = img_tensor_A
            human_mask_tensor = mask_tensor_A.unsqueeze(0) # [1, H, W]
            # sam_mask_tensor = sam_tensor_A.unsqueeze(0)     # [1, H, W]

        else:
            # Flujo Normal sin Copy-Paste
            if self.transformador is not None:
                aug = self.transformador(
                    image=image_uint8_A, 
                    mask=binary_human_mask_A, 
                    # masks=[sam_mask_np_A] 
                )
                image_tensor = aug['image']
                human_mask_tensor = aug['mask'].unsqueeze(0)
                # sam_mask_tensor = aug['masks'][0].unsqueeze(0) 
            else:
                image_tensor = F.to_tensor(image_uint8_A)
                human_mask_tensor = torch.from_numpy(binary_human_mask_A).unsqueeze(0)
                # sam_mask_tensor = torch.from_numpy(sam_mask_np_A).unsqueeze(0)
                
                H_tgt, W_tgt = self.img_size
                image_tensor = torch.nn.functional.interpolate(image_tensor.unsqueeze(0), size=(H_tgt, W_tgt), mode='bilinear', align_corners=False).squeeze(0)
                human_mask_tensor = torch.nn.functional.interpolate(human_mask_tensor.unsqueeze(0), size=(H_tgt, W_tgt), mode='nearest').squeeze(0)

        sdf_tensor = compute_sdf_single_mask(human_mask_tensor)
        return image_tensor, human_mask_tensor, sdf_tensor

class CustomAdversarialDataset(Dataset):
    def __init__(self, df_oxford, df_ade20k, data_dir_oxford, data_dir_ade20k, transformador=None):
        """
        Dataset especializado en Adversarial Training.
        Por cada imagen cargada, recorta a la mascota de Oxford y la adhiere sin piedad
        sobre un fondo seleccionado de la "lista negra" de ADE20K (sillas, tapetes, etc)
        """
        self.df_oxford = df_oxford
        self.df_ade20k = df_ade20k
        
        self.img_dir_oxford = os.path.join(data_dir_oxford, "images")
        self.mask_dir_oxford = os.path.join(data_dir_oxford, "masks")
        self.data_dir_ade20k = data_dir_ade20k
        if transformador is None:
            self.transformador = get_lsj_transform()
        else:
            self.transformador = transformador

    def _get_raw_data(self, idx):
        # 1. Obteniendo el nombre del archivo iterado mediante Pandas (iloc)
        # La columna 0 es 'file' en tu dataset de Oxford
        img_name = str(self.df_oxford.iloc[idx, 0])
        base_name = img_name.split('.')[0]
        
        img_path = os.path.join(self.img_dir_oxford, img_name)
        mask_path = os.path.join(self.mask_dir_oxford, base_name + ".png")
        
        # 2. Cargar imagen original del perrito o michi
        image = Image.open(img_path).convert("RGB")
        image_uint8 = np.array(image)
        
        # 3. Mapeo y Destrucción Estilizada de las Etiquetas de Oxford 
        # Convertimos: [Animal, Fondo, Borde] -> Animal y Borde=1. Fondo=0.
        try:
            human_mask = Image.open(mask_path)
            human_mask_np = np.array(human_mask, dtype=np.float32)
        except Exception as e:
            # Cachar errores si falta una máscara del dataset
            print(f"Error cargando la máscara {mask_path}: {e}")
            human_mask_np = np.zeros(image_uint8.shape[:2], dtype=np.float32)
            
        binary_mask = np.zeros_like(human_mask_np, dtype=np.float32)
        # El usuario confirmó que sus máscaras usan 0: Mascota, 1: Fondo
        binary_mask[human_mask_np == 0] = 1.0 # Mascota original Pura
        binary_mask[human_mask_np == 2] = 1.0 # Borde se vuelve mascota
        binary_mask[human_mask_np == 1] = 0.0 # Todo el fondo restante a 0 absoluto.
        
        return image_uint8, binary_mask

    def __len__(self):
        return len(self.df_oxford)

    def __getitem__(self, idx):
        # A) Recuperamos al animal a recortar puramente en Numpy
        img_a_np, mask_a_np = self._get_raw_data(idx)
        
        # B y C) Copy-Paste Totalmente Modularizado y Matemático
        # Llamamos a nuestra función exportada pura en Numpy de copy_paste_augmentation.py
        img_sintetica_np = get_augmentation_ade20k(
            df_ade20k=self.df_ade20k, 
            ade20k_base_dir=self.data_dir_ade20k, 
            img_a_np=img_a_np, 
            mask_a_np=mask_a_np
        )
        
        # D) Entregar al Transformador (LSJ Post-Fusión)
        if self.transformador is not None:
            # LSJ se encargará de fundir el contraste y borrar el borde de recorte (Edge artifact)
            aug = self.transformador(image=img_sintetica_np, mask=mask_a_np, masks=[mask_a_np])
            image_tensor = aug['image']
            human_mask_tensor = aug['mask'].unsqueeze(0)
        else:
            image_tensor = F.to_tensor(img_sintetica_np)
            human_mask_tensor = torch.from_numpy(mask_a_np).unsqueeze(0)
            
            # Ajuste extra si no hay albumentations
            H_tgt, W_tgt = (384, 384) # asumiendo el standard de tus validaciones
            image_tensor = torch.nn.functional.interpolate(image_tensor.unsqueeze(0), size=(H_tgt, W_tgt), mode='bilinear', align_corners=False).squeeze(0)
            human_mask_tensor = torch.nn.functional.interpolate(human_mask_tensor.unsqueeze(0), size=(H_tgt, W_tgt), mode='nearest').squeeze(0)

        sdf_tensor = compute_sdf_single_mask(human_mask_tensor)
        return image_tensor, human_mask_tensor, sdf_tensor

class NegativeADE20KDataset(Dataset):
    def __init__(self, df_ade20k, ade20k_dir, transform=None, img_size=(384, 384)):
        """
        Dataset de "Hard Negatives" de ADE20K.
        Devuelve la imagen trampa (sofás, camas, tapetes) pero con una máscara
        y SDF completamente en CEROS, enseñándole a la red que NO hay mascota.
        """
        self.df = df_ade20k
        self.ade20k_dir = ade20k_dir
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def _get_raw_data(self, idx):
        # Asumiendo que tu df_ade20k tiene la ruta en la primera columna o en 'file'
        col_name = 'file' if 'file' in self.df.columns else self.df.columns[0]
        img_filename = str(self.df.iloc[idx][col_name])
        
        # Buscar la imagen en la carpeta (puede venir solo el nombre o la ruta parcial)
        img_path = os.path.join(self.ade20k_dir, img_filename)
        if not os.path.exists(img_path):
            img_path = os.path.join(self.ade20k_dir, "images", "training", os.path.basename(img_filename))
            
        try:
            image = Image.open(img_path).convert("RGB")
            image_uint8 = np.array(image)
        except Exception as e:
            print(f"Error cargando negativo {img_path}: {e}")
            # Imagen negra de seguridad
            image_uint8 = np.zeros((*self.img_size, 3), dtype=np.uint8)
            
        # Máscara 100% vacía (fondo puro, NO HAY mascota aquí)
        binary_mask = np.zeros(image_uint8.shape[:2], dtype=np.float32)
        return image_uint8, binary_mask

    def __getitem__(self, idx):
        image_uint8, binary_mask = self._get_raw_data(idx)
        
        if self.transform is not None:
            aug = self.transform(image=image_uint8, mask=binary_mask)
            image_tensor = aug['image']
            human_mask_tensor = aug['mask'].unsqueeze(0)
        else:
            image_tensor = F.to_tensor(image_uint8)
            human_mask_tensor = torch.from_numpy(binary_mask).unsqueeze(0)
            
            H_tgt, W_tgt = self.img_size
            image_tensor = torch.nn.functional.interpolate(image_tensor.unsqueeze(0), size=(H_tgt, W_tgt), mode='bilinear', align_corners=False).squeeze(0)
            human_mask_tensor = torch.nn.functional.interpolate(human_mask_tensor.unsqueeze(0), size=(H_tgt, W_tgt), mode='nearest').squeeze(0)
            
        # El compute_sdf_single_mask ya está programado sabiamente:
        # Si la máscara es puros ceros, devuelve un SDF de puros UNOS (castigo máximo)
        sdf_tensor = compute_sdf_single_mask(human_mask_tensor)
        
        return image_tensor, human_mask_tensor, sdf_tensor

def get_balanced_dataloaders(csv_labels_path:str, imgs_dir:str, human_masks_dir:str, sam_logits_dir:str, batch_size:int = 8, img_size:tuple = (256, 256), val_split:float = 0.1, test_split:float = 0.1):
    """
    Carga el dataset, realiza una partición estratificada guardando la "naturaleza" estándar (train/val/test),
    y devuelve los DataLoaders con un WeightedRandomSampler para forzar 50% gatos / 50% perros en CADA ÉPOCA.
    """
    df = pd.read_csv(csv_labels_path)
    
    # 1. Identificar si es gato (Mayúscula inicial) o perro (Minúscula inicial)
    df['is_cat'] = df['file'].str[0].str.isupper()
    
    # 2. Stratified Split Manual con Pandas (Entrenamiento)
    train_frac = 1.0 - val_split - test_split
    
    train_dfs = []
    for cat_flag in [True, False]:
        sub_df = df[df['is_cat'] == cat_flag]
        train_dfs.append(sub_df.sample(frac=train_frac, random_state=42))
    train_df = pd.concat(train_dfs)
    
    # El resto (Val + Test)
    rem_df = df.drop(train_df.index)
    
    # Partición estratificada del resto (Validación)
    val_rel_frac = val_split / (val_split + test_split) if (val_split + test_split) > 0 else 0
    val_dfs = []
    for cat_flag in [True, False]:
        sub_df = rem_df[rem_df['is_cat'] == cat_flag]
        val_dfs.append(sub_df.sample(frac=val_rel_frac, random_state=42))
    val_df = pd.concat(val_dfs)
    
    # Lo último que sobra es el Test (dev_test)
    test_df = rem_df.drop(val_df.index)
    
    # Reset indices
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    # 3. Función Interna para Crear Samplers (Balance 50/50 durante la carga)
    def create_sampler(dataframe):
        class_counts = dataframe['is_cat'].value_counts()
        weights = [1.0 / class_counts[val] for val in dataframe['is_cat']]
        return WeightedRandomSampler(
            weights=weights, 
            num_samples=len(weights),
            replacement=True
        )

    train_sampler = create_sampler(train_df)
    val_sampler = create_sampler(val_df)
    test_sampler = create_sampler(test_df)
    
    # 4. Creación de Datasets
    train_dataset = CustomDistillationDataset(
        df=train_df, imgs_dir=imgs_dir, human_masks_dir=human_masks_dir, 
        sam_logits_dir=sam_logits_dir, img_size=img_size, transform=None 
    )
    
    val_dataset = CustomDistillationDataset(
        df=val_df, imgs_dir=imgs_dir, human_masks_dir=human_masks_dir, 
        sam_logits_dir=sam_logits_dir, img_size=img_size, transform=None
    )
    
    test_dataset = CustomDistillationDataset(
        df=test_df, imgs_dir=imgs_dir, human_masks_dir=human_masks_dir, 
        sam_logits_dir=sam_logits_dir, img_size=img_size, transform=None
    )
    
    # 5. DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, num_workers=2)
    
    print(f"✅ DataLoaders Balanceados (50/50) Creados Exitosamente.")
    print(f"   - Entrenamiento: {len(train_df)} muestras (Gatos: {train_df['is_cat'].sum()})")
    print(f"   - Validación: {len(val_df)} muestras (Gatos: {val_df['is_cat'].sum()})")
    print(f"   - Dev/Test: {len(test_df)} muestras (Gatos: {test_df['is_cat'].sum()})")
    
    return train_loader, val_loader, test_loader

def get_dual_dataloaders(base_data_path:str, csv_labels_path:str, ade20k_dir:str, ade20k_csv:str, batch_size:int=4, img_size=(384,384)):
    """
    Construye el diccionario de DataLoaders paralelos Normales y Adversariales,
    asegurándose de utilizar EXCLUSIVAMENTE datos originales (Oxford), filtrando los seudoetiquetados (GOLD).
    """
    imgs_dir = os.path.join(base_data_path, "images")
    masks_dir = os.path.join(base_data_path, "masks")
    
    df_oxford = pd.read_csv(csv_labels_path)
    df_ade20k = pd.read_csv(ade20k_csv)
    
    # 1. Filtro estricto: Eliminamos los datos "GOLD" generados por inferencias anteriores
    # Accedemos a la columna Booleana que tienes en tu dataset_generated.csv
    if 'is_gold' in df_oxford.columns:
        df_oxford = df_oxford[df_oxford['is_gold'] == False]
    else:
        # Fallback de seguridad en caso de fallo de lectura
        df_oxford = df_oxford[~df_oxford['file'].str.startswith('pseudo_')]
        
    print(f"🚀 Creando DataLoaders Duales con {len(df_oxford)} imágenes base puras (sin Gold data)")
        
    # 2. Random Split Global 80/20
    train_size = int(0.8 * len(df_oxford))
    train_df = df_oxford.sample(n=train_size, random_state=42)
    val_df = df_oxford.drop(train_df.index)
    
    # 3. Datasets Normales
    train_dataset_norm = CustomDistillationDataset(df=train_df, imgs_dir=imgs_dir, human_masks_dir=masks_dir, sam_logits_dir=masks_dir, img_size=img_size)
    val_dataset_norm = CustomDistillationDataset(df=val_df, imgs_dir=imgs_dir, human_masks_dir=masks_dir, sam_logits_dir=masks_dir, img_size=img_size)
    
    # 4. Datasets Adversariales (Reaprovechamos las mismas particiones para alinear la validación global)
    train_dataset_adv = CustomAdversarialDataset(df_oxford=train_df, df_ade20k=df_ade20k, data_dir_oxford=base_data_path, data_dir_ade20k=ade20k_dir, transformador=None)
    val_dataset_adv = CustomAdversarialDataset(df_oxford=val_df, df_ade20k=df_ade20k, data_dir_oxford=base_data_path, data_dir_ade20k=ade20k_dir, transformador=None)
    
    # 5. DataLoaders
    # NOTA IMPORTANTE PARA EL USUARIO: Num_workers se ajusta aquí. Si peta en Windows pon worker=0, sino mantén 2.
    train_dl_norm = DataLoader(train_dataset_norm, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dl_norm = DataLoader(val_dataset_norm, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    train_dl_adv = DataLoader(train_dataset_adv, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dl_adv = DataLoader(val_dataset_adv, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return {
        'train_norm': train_dl_norm,
        'val_norm': val_dl_norm,
        'train_adv': train_dl_adv,
        'val_adv': val_dl_adv
    }