"""
adversarial_dataset.py
======================
Dataset unificado para adversarial fine-tuning.

Fuentes:
    - Positivos: Oxford-IIIT Pet (mascotas con máscara real)
    - Negativos: ADE20K filtrado (texturas trampa, máscara = todo ceros)

Devuelve por cada muestra: (imagen_tensor, mascara_tensor, sdf_tensor, es_adversarial)
    - imagen_tensor   : [3, H, W] float32 normalizado ImageNet
    - mascara_tensor  : [1, H, W] float32, 0.0 ó 1.0
    - sdf_tensor      : [1, H, W] float32 SDF pre-computado (cargado desde disco)
    - es_adversarial  : bool (True si es imagen negativa de ADE20K)

Optimizaciones vs versión anterior:
    - SDFs pre-computados en disco (.npy) → SIN scipy en el worker → sin tirones
    - SDF se pasa como máscara adicional a Albumentations → transforms consistentes
    - Para negativos: SDF = torch.ones() instantáneo, sin lectura a disco
    - Transforms separados: geométricos (imagen+máscara+SDF) / fotométricos (solo imagen)
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ─── Resolución de imports relativos ────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
source_dir  = os.path.abspath(os.path.join(current_dir, "..", ".."))
if source_dir not in sys.path:
    sys.path.append(source_dir)

# ─── Transformadores ─────────────────────────────────────────────────────────

def _build_geo_transform(img_size: tuple) -> A.Compose:
    """
    Transforms GEOMÉTRICOS: se aplican igual a imagen, máscara y SDF.
    El SDF viaja como máscara adicional en 'masks=[sdf]'.
    """
    H, W = img_size
    return A.Compose([
        A.RandomResizedCrop(size=(H, W), scale=(0.5, 1.0), ratio=(0.8, 1.2), p=0.5),
        A.Resize(height=H, width=W),
        A.HorizontalFlip(p=0.5),
        A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-15, 15), p=0.4),
    ])


def _build_photo_transform() -> A.Compose:
    """
    Transforms FOTOMÉTRICOS: solo afectan a la imagen RGB, no al SDF ni a la máscara.
    """
    return A.Compose([
        A.RandomBrightnessContrast(p=0.7),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
        A.OneOf([
            A.MotionBlur(p=0.3),
            A.GaussianBlur(blur_limit=3, p=0.3),
            A.GaussNoise(p=0.3),
        ], p=0.3),
        A.CoarseDropout(
            num_holes_range=(1, 4),
            hole_height_range=(8, 40),
            hole_width_range=(8, 40),
            p=0.25
        ),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


# ─── Dataset Unificado ───────────────────────────────────────────────────────

class AdversarialPetDataset(Dataset):
    """
    Dataset único que combina Oxford-IIIT Pet (positivos) y ADE20K (negativos).

    El flag `es_adversarial` por muestra permite distinguirlas dentro del loop
    de entrenamiento sin necesidad de dos DataLoaders separados.

    Args:
        df_oxford  : DataFrame con columna 'file' apuntando a imágenes de Oxford.
        df_ade20k  : DataFrame con columna 'file' apuntando a imágenes de ADE20K.
        oxford_dir : Directorio raíz de Oxford (debe contener 'images/' y 'masks/').
        ade20k_dir : Directorio raíz de ADE20K (debe contener 'images/training/').
        img_size   : Tupla (H, W) de salida.
        transform  : Transformador Albumentations. Si None, se usa el estándar.
    """

    # Mapeo de etiquetas Oxford: 0=Mascota, 1=Fondo, 2=Borde→Mascota
    _OXFORD_LABEL_MAP = {0: 1.0, 2: 1.0, 1: 0.0}

    def __init__(
        self,
        df_oxford,
        df_ade20k,
        oxford_dir: str,
        ade20k_dir: str,
        sdf_dir: str = None,           # Carpeta con los .npy pre-computados
        img_size: tuple = (384, 384),
    ):
        self.oxford_dir  = oxford_dir
        self.ade20k_dir  = ade20k_dir
        self.img_size    = img_size

        # Si no se especifica, buscamos en data/oxford/sdfs/ por defecto
        self.sdf_dir = sdf_dir or os.path.join(oxford_dir, "sdfs")
        self._sdf_cache_available = os.path.isdir(self.sdf_dir)

        if not self._sdf_cache_available:
            print(
                f"[AdversarialPetDataset] ⚠️  No se encontró carpeta de SDFs en '{self.sdf_dir}'.\n"
                f"   Ejecuta: python utils/scripts/precompute_sdfs.py\n"
                f"   Por ahora los SDFs se calcularán al vuelo (más lento)."
            )

        # Transforms separados: geométrico (imagen+máscara+SDF) y fotométrico (solo imagen)
        self._geo   = _build_geo_transform(img_size)
        self._photo = _build_photo_transform()

        # Indexamos positivos y negativos con su origen marcado
        df_oxford  = df_oxford.copy()
        df_ade20k  = df_ade20k.copy()
        df_oxford["_is_adv"]  = False
        df_ade20k["_is_adv"]  = True

        # Normalizamos el nombre de columna a 'file'
        if "file" not in df_oxford.columns:
            df_oxford = df_oxford.rename(columns={df_oxford.columns[0]: "file"})
        if "file" not in df_ade20k.columns:
            df_ade20k = df_ade20k.rename(columns={df_ade20k.columns[0]: "file"})

        import pandas as pd
        self.df = pd.concat([df_oxford, df_ade20k], ignore_index=True)

        # Guardamos los rangos de índice para el sampler externo
        self.num_positivos = len(df_oxford)
        self.num_negativos = len(df_ade20k)

    # ── Propiedades auxiliares para el Sampler ────────────────────────────────
    @property
    def positive_indices(self) -> list:
        return list(range(0, self.num_positivos))

    @property
    def negative_indices(self) -> list:
        return list(range(self.num_positivos, self.num_positivos + self.num_negativos))

    # ── Carga de SDF pre-computado ────────────────────────────────────────────
    def _load_sdf_np(self, basename: str) -> np.ndarray | None:
        """
        Carga el SDF pre-computado desde disco.
        Devuelve None si no existe (fallback a cómputo al vuelo).
        """
        npy_path = os.path.join(self.sdf_dir, basename + ".npy")
        if os.path.exists(npy_path):
            return np.load(npy_path)      # shape (H, W) float32
        return None

    def _compute_sdf_fallback(self, mask_np: np.ndarray) -> np.ndarray:
        """Fallback: calcula SDF con scipy si no hay archivo en disco."""
        from scipy.ndimage import distance_transform_edt
        posmask = mask_np > 0.5
        if posmask.any():
            dist_out = distance_transform_edt(~posmask)
            dist_in  = distance_transform_edt(posmask)
            sdf_map  = dist_out - dist_in
            max_d    = np.max(np.abs(sdf_map))
            if max_d > 0:
                sdf_map /= max_d
            return sdf_map.astype(np.float32)
        return np.ones_like(mask_np, dtype=np.float32)

    # ── Lectura de muestra cruda ──────────────────────────────────────────────
    def _load_oxford(self, idx: int):
        """Carga imagen, máscara y SDF pre-computado de Oxford."""
        row       = self.df.iloc[idx]
        filename  = str(row["file"])
        basename  = os.path.splitext(os.path.basename(filename))[0]

        img_path  = os.path.join(self.oxford_dir, "images", filename)
        if not os.path.exists(img_path):
            if os.path.exists(filename):
                img_path = filename
            elif filename.startswith("source/") and os.path.exists(filename[7:]):
                img_path = filename[7:]

        # Permitir usar columna mask si existe en el CSV
        is_fur_coat = "hard_negatives" in str(row.get("mask", "")) or "hard_negatives" in filename
        if "mask" in row and pd.notna(row["mask"]) and row["mask"] != "":
            mask_col = str(row["mask"])
            mask_path = mask_col
            if not os.path.exists(mask_path) and mask_col.startswith("source/") and os.path.exists(mask_col[7:]):
                mask_path = mask_col[7:]
        else:
            mask_path = os.path.join(self.oxford_dir, "masks", basename + ".png")

        image = np.array(Image.open(img_path).convert("RGB"))

        if os.path.exists(mask_path):
            raw_mask = np.array(Image.open(mask_path), dtype=np.int64)
            bin_mask = np.zeros_like(raw_mask, dtype=np.float32)
            
            # ESTANDARIZACIÓN ESTRICTA: 1.0 = Mascota, 0.0 = Fondo
            if is_fur_coat:
                # Nuestras máscaras custom de abrigos: 0 = Mascota, 1 = Fondo
                bin_mask[raw_mask == 0] = 1.0
                bin_mask[raw_mask == 1] = 0.0
            else:
                # Oxford original (etiquetas generadas por el usuario con SAM):
                # 0 = Fondo, 1 = Mascota, 2 = Borde
                bin_mask[raw_mask == 1] = 1.0 # Mascota
                bin_mask[raw_mask == 2] = 1.0 # Borde (opcionalmente mascota para no penalizar bordes)
                bin_mask[raw_mask == 0] = 0.0 # Fondo
        else:
            bin_mask = np.zeros(image.shape[:2], dtype=np.float32)

        # Intentar cargar SDF pre-computado (sin scipy)
        sdf_np = self._load_sdf_np(basename)
        if sdf_np is None:
            sdf_np = self._compute_sdf_fallback(bin_mask)

        return image, bin_mask, sdf_np

    def _load_ade20k(self, idx: int):
        """Carga imagen de ADE20K. Máscara y SDF son todo ceros/unos instantáneamente."""
        row      = self.df.iloc[idx]
        filename = str(row["file"])

        img_path = os.path.join(self.ade20k_dir, filename)
        if not os.path.exists(img_path):
            img_path = os.path.join(
                self.ade20k_dir, "images", "training", os.path.basename(filename)
            )
        
        if not os.path.exists(img_path):
            if os.path.exists(filename):
                img_path = filename
            elif filename.startswith("source/") and os.path.exists(filename[7:]):
                img_path = filename[7:]

        try:
            image = np.array(Image.open(img_path).convert("RGB"))
        except Exception as e:
            print(f"[AdversarialPetDataset] Error cargando {img_path}: {e}")
            image = np.zeros((*self.img_size, 3), dtype=np.uint8)

        H, W      = image.shape[:2]
        
        # ESTANDARIZACIÓN ESTRICTA: 0.0 = Fondo
        bin_mask  = np.zeros((H, W), dtype=np.float32)
        
        # SDF = 1.0 (Positivo significa que estás fuera del objeto, es decir, Fondo)
        # Esto penaliza si la red predice > 0.0 (Mascota)
        sdf_np    = np.ones((H, W), dtype=np.float32)

        return image, bin_mask, sdf_np

    # ── __getitem__ ───────────────────────────────────────────────────────────
    def __getitem__(self, idx: int):
        es_adversarial = bool(self.df.iloc[idx]["_is_adv"])

        if es_adversarial:
            image, bin_mask, sdf_np = self._load_ade20k(idx)
        else:
            image, bin_mask, sdf_np = self._load_oxford(idx)

        # 1. Transforms GEOMÉTRICOS: imagen + máscara + SDF con el mismo parámetro aleatorio
        geo = self._geo(image=image, mask=bin_mask, masks=[sdf_np])
        image    = geo["image"]          # ndarray RGB augmentado
        bin_mask = geo["mask"]           # ndarray máscara augmentada
        sdf_np   = geo["masks"][0]       # ndarray SDF con los mismos flips/crops

        # 2. Transforms FOTOMÉTRICOS: solo imagen (brillo, contraste, blur, normalize → tensor)
        photo       = self._photo(image=image)
        img_tensor  = photo["image"]                               # [3, H, W] float32

        # Máscara y SDF a tensor manualmente (no pasan por transforms fotométricos)
        mask_tensor = torch.from_numpy(bin_mask).unsqueeze(0)     # [1, H, W]
        sdf_tensor  = torch.from_numpy(sdf_np).unsqueeze(0)       # [1, H, W]

        return img_tensor, mask_tensor, sdf_tensor, es_adversarial

    def __len__(self) -> int:
        return len(self.df)


# ─── Sampler de Proporción Positivo / Negativo ───────────────────────────────

class NegativeAwareBatchSampler(Sampler):
    """
    Garantiza que cada batch contenga exactamente `num_pos_per_batch` positivos
    y `batch_size - num_pos_per_batch` negativos.

    Reglas:
        - Los positivos se barajan y consumen secuencialmente (sin reemplazo).
        - Los negativos se muestrean con replace=True si hay menos negativos que
          `num_neg_per_batch`, y con replace=False si hay suficientes.

    Args:
        positive_indices   : Lista de índices de muestras positivas en el dataset.
        negative_indices   : Lista de índices de muestras negativas.
        batch_size         : Tamaño total del batch (ej. 64).
        num_pos_per_batch  : Positivos por batch (ej. 54). Negativos = 64 - 54 = 10.
    """

    def __init__(
        self,
        positive_indices: list,
        negative_indices: list,
        batch_size: int = 64,
        num_pos_per_batch: int = 54,
    ):
        self.pos  = np.array(positive_indices)
        self.neg  = np.array(negative_indices)
        self.batch_size       = batch_size
        self.num_pos          = num_pos_per_batch
        self.num_neg          = batch_size - num_pos_per_batch
        self.num_batches      = len(self.pos) // self.num_pos

        if len(self.neg) == 0:
            raise ValueError("No hay índices negativos. Revisa tu df_ade20k.")
        if self.num_neg <= 0:
            raise ValueError("num_pos_per_batch debe ser menor que batch_size.")

    def __iter__(self):
        # Barajar positivos al inicio de cada época (sin reemplazo)
        pos_shuffled = self.pos.copy()
        np.random.shuffle(pos_shuffled)

        # Negativos: sin reemplazo si hay suficientes, con reemplazo si no
        replace_neg = len(self.neg) < self.num_neg

        for i in range(self.num_batches):
            pos_start = i * self.num_pos
            pos_end   = pos_start + self.num_pos
            batch_pos = pos_shuffled[pos_start:pos_end]

            batch_neg = np.random.choice(self.neg, size=self.num_neg, replace=replace_neg)

            batch = np.concatenate([batch_pos, batch_neg])
            np.random.shuffle(batch)          # mezcla interna para no dar orden fijo
            yield batch.tolist()

    def __len__(self) -> int:
        return self.num_batches


# ─── Factory: DataLoader listo para usar ─────────────────────────────────────

def get_adversarial_dataloader(
    df_oxford,
    df_ade20k,
    oxford_dir: str,
    ade20k_dir: str,
    batch_size: int = 64,
    num_pos_per_batch: int = 54,
    img_size: tuple = (384, 384),
    num_workers: int = 4,
    df_personas=None,           # DataFrame opcional con personas con ropa de piel
):
    """
    Construye el DataLoader único de adversarial fine-tuning con el sampler
    de proporción integrado.

    Args:
        df_personas : DataFrame con columna 'file' de imágenes de personas de ADE20K.
                      Si se provee, se concatena con df_ade20k como negativos adicionales.
                      Generado por utils/scripts/generate_person_negatives_csv.py

    Returns:
        (DataLoader, AdversarialPetDataset)
    """
    import pandas as pd
    from torch.utils.data import DataLoader

    # Combinar negativos de texturas + personas si se proveen
    if df_personas is not None:
        df_personas = df_personas.copy()
        # Normalizar columna a 'file'
        if "file" not in df_personas.columns:
            df_personas = df_personas.rename(columns={df_personas.columns[0]: "file"})
        df_negativos = pd.concat([df_ade20k, df_personas], ignore_index=True)
        print(
            f"ℹ️  Negativos combinados: "
            f"{len(df_ade20k)} texturas ADE20K + {len(df_personas)} personas = {len(df_negativos)} total"
        )
    else:
        df_negativos = df_ade20k

    dataset = AdversarialPetDataset(
        df_oxford=df_oxford,
        df_ade20k=df_negativos,
        oxford_dir=oxford_dir,
        ade20k_dir=ade20k_dir,
        img_size=img_size,
    )

    sampler = NegativeAwareBatchSampler(
        positive_indices=dataset.positive_indices,
        negative_indices=dataset.negative_indices,
        batch_size=batch_size,
        num_pos_per_batch=num_pos_per_batch,
    )

    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    print(
        f"✅ DataLoader adversarial listo — "
        f"{len(dataset.positive_indices)} positivos | "
        f"{len(dataset.negative_indices)} negativos | "
        f"{len(sampler)} batches/época ({num_pos_per_batch}P + {batch_size - num_pos_per_batch}N por batch)"
    )

    return loader, dataset


def get_adversarial_dataloaders(
    df_oxford,
    df_ade20k,
    oxford_dir: str,
    ade20k_dir: str,
    batch_size: int = 64,
    num_pos_per_batch: int = 54,
    img_size: tuple = (384, 384),
    num_workers: int = 4,
    val_split: float = 0.1,
    test_split: float = 0.1,
    random_state: int = 42,
    df_personas=None,           # Nuevo: personas con ropa de piel como hard negatives
):
    """
    Divide Oxford y ADE20K en train/val/test y devuelve 3 DataLoaders.

    - Train : NegativeAwareBatchSampler (proporción garantizada)
    - Val   : DataLoader normal con shuffle=False (no necesita proporción fija)
    - Test  : DataLoader normal con shuffle=False

    El flag `es_adversarial` sigue estando en cada muestra para poder separar
    métricas de positivos/negativos dentro del loop de validación.

    Args:
        val_split  : Fracción de cada fuente para validación (default 0.1 → 10%)
        test_split : Fracción de cada fuente para test        (default 0.1 → 10%)

    Returns:
        dict con claves 'train', 'val', 'test'
    """
    from torch.utils.data import DataLoader

    train_frac = 1.0 - val_split - test_split
    assert train_frac > 0, "val_split + test_split debe ser menor que 1.0"

    def _split_df(df):
        """Parte un DataFrame en train / val / test estratificadamente."""
        train = df.sample(frac=train_frac, random_state=random_state)
        rest  = df.drop(train.index)
        # Del resto, val ocupa val_split/(val_split+test_split)
        val_frac_local = val_split / (val_split + test_split)
        val  = rest.sample(frac=val_frac_local, random_state=random_state)
        test = rest.drop(val.index)
        return (
            train.reset_index(drop=True),
            val.reset_index(drop=True),
            test.reset_index(drop=True),
        )

    train_ox, val_ox, test_ox     = _split_df(df_oxford)
    train_ade, val_ade, test_ade  = _split_df(df_ade20k)

    # Split del CSV de personas si se proveyó
    import pandas as pd
    if df_personas is not None:
        df_personas = df_personas.copy()
        if "file" not in df_personas.columns:
            df_personas = df_personas.rename(columns={df_personas.columns[0]: "file"})
        train_per, val_per, test_per = _split_df(df_personas)
        # Combinar personas con ADE20K dentro de cada split
        train_ade_all = pd.concat([train_ade, train_per], ignore_index=True)
        val_ade_all   = pd.concat([val_ade,   val_per],   ignore_index=True)
        test_ade_all  = pd.concat([test_ade,  test_per],  ignore_index=True)
        print(
            f"ℹ️  Personas añadidas como hard negatives: "
            f"{len(train_per)} train | {len(val_per)} val | {len(test_per)} test"
        )
    else:
        train_ade_all = train_ade
        val_ade_all   = val_ade
        test_ade_all  = test_ade

    # ── TRAIN: con sampler de proporción ──────────────────────────────────────
    train_loader, _ = get_adversarial_dataloader(
        df_oxford=train_ox,
        df_ade20k=train_ade_all,
        oxford_dir=oxford_dir,
        ade20k_dir=ade20k_dir,
        batch_size=batch_size,
        num_pos_per_batch=num_pos_per_batch,
        img_size=img_size,
        num_workers=num_workers,
    )

    # ── VAL: sin sampler especial ─────────────────────────────────────────────
    val_dataset = AdversarialPetDataset(
        df_oxford=val_ox,
        df_ade20k=val_ade_all,
        oxford_dir=oxford_dir,
        ade20k_dir=ade20k_dir,
        img_size=img_size,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(batch_size // 4, 1),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    # ── TEST: sin sampler especial ────────────────────────────────────────────
    test_dataset = AdversarialPetDataset(
        df_oxford=test_ox,
        df_ade20k=test_ade_all,
        oxford_dir=oxford_dir,
        ade20k_dir=ade20k_dir,
        img_size=img_size,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=max(batch_size // 4, 1),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    neg_total_train = len(train_ade_all)
    print(
        f"✅ DataLoaders adversariales listos —\n"
        f"   Train : {len(train_ox)} pos + {neg_total_train} neg\n"
        f"   Val   : {len(val_ox)} pos + {len(val_ade_all)} neg\n"
        f"   Test  : {len(test_ox)} pos + {len(test_ade_all)} neg"
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}


