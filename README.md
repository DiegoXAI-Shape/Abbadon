# 🐾 Abbadon — Segmentación Semántica de Mascotas

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-EE4C2C?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-13.0-76B900?logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

**Segmentación semántica pixel-level de mascotas (perros y gatos) usando arquitecturas U-Net con encoders preentrenados, Attention Gates y canal Fourier.**

</div>

---

## 📖 Descripción

Abbadon es un proyecto de segmentación semántica que clasifica cada píxel de una imagen en 3 clases: **mascota**, **fondo** y **borde**. El proyecto ha evolucionado a través de múltiples iteraciones experimentales, desde una U-Net básica hasta arquitecturas con ConvNeXtV2 preentrenado y descomposición Fourier.

### Arquitecturas implementadas

| Modelo | Encoder | Bottleneck | Canales | Data Augmentation
|---|---|---|---|---|
| `Daowa_maad` | ResNet (from scratch) | — | 3 (RGB) | No
| `Daowa_maadV2` | ResNet (from scratch) | Transformer Encoder | 3 (RGB) | No
| `Daowa_maadV3-rc 1.0` | ConvNeXtV2 Tiny (pretrained) | — | 4 (RGB + Fourier) | No
| `Daowa_maadV3-rc 1.1` | ConvNeXtV2 Tiny (pretrained) | — | 4 (RGB + Fourier) | Si

---

## 🏗️ Estructura del Proyecto

```
Abbadon/
├── 📄 .gitignore
├── 📄 LICENSE
├── 📂 Models/                          # Pesos entrenados (.pth)
└── 📂 source/
    ├── 📓 Daowa_maadV3.ipynb           # Notebook principal de experimentación
    ├── 📓 Prueba.ipynb                 # Notebook de inferencia/pruebas
    ├── 📂 data/                        # Dataset de imágenes y máscaras
    ├── 📂 labels/                      # CSVs del dataset
    ├── 📂 logs/                        # CSVs de métricas + TensorBoard
    │   └── 📂 tensorboard/
    ├── 📂 images/                      # Capturas de resultados
    └── 📂 utils/                       # Módulos del proyecto
        ├── 📂 models/
        │   ├── blocks.py               # Bloques: BloqueResidual, UpSampling, AttentionGates, EncoderTrans
        │   ├── datasets.py             # Datasets: CustomDS, CusDataset (Fourier), get_dataloaders()
        │   ├── daowa_maad.py           # Modelos: Daowa_maad, TransformerDaowa_maad
        │   └── daowa_maadV3Prueba.py   # Modelos: Daowa_maadPrueba, Daowa_maadPrueba2 (ConvNeXtV2)
        ├── 📂 losses/
        │   └── dice_loss.py            # GeneralizedDiceLoss (ponderación automática por volumen)
        ├── 📂 metrics/
        │   └── iou.py                  # IoU global y por clase
        ├── 📂 train/
        │   └── trainer.py              # Loop de entrenamiento + TensorBoard + CSV logging
        ├── 📂 inference/
        │   └── predict.py              # Post-procesamiento de máscaras + visualización
        └── 📂 visualization/
            └── compare.py              # Comparación de N entrenamientos con gráficas premium
```

---

## 📊 Resultados Experimentales

### Comparativa de modelos — Mejor epoch de cada uno

| Métrica | Sin Transformer | Con Transformer | ConvNeXtV2 + Aug |
|:---|:---:|:---:|:---:|
| **Epochs entrenados** | 20 | 20 | 10 |
| **Mejor epoch** | 13 | 14 | 9 |
| **Val Loss** | 0.2382 | 0.2413 | 0.3166 |
| **Val Accuracy** | 92.01% | 91.89% | **92.83%** ✅ |
| **mIoU Global** | 0.7807 | 0.7780 | **0.7946** ✅ |
| **IoU Mascota** | 85.44% | 85.19% | **87.21%** ✅ |
| **IoU Fondo** | 92.15% | 92.02% | **93.72%** ✅ |
| **IoU Borde** ⚠️ | 56.62% | 56.19% | **57.47%** ✅ |

> **Nota:** ConvNeXtV2 + Aug alcanzó las mejores métricas en **la mitad de epochs**, demostrando la ventaja del transfer learning con backbones preentrenados.

### Curvas de entrenamiento

<div align="center">
<img src="source/images/Daowa_maadV3_rc 3.1.png" alt="Comparación de 3 modelos" width="100%">
</div>

---

## 🧠 Pipeline de Entrada

```
Imagen RGB ──► Resize (192×192) ──► Normalización ──┐
                                                      ├──► Tensor [4, 192, 192] ──► Modelo
Imagen Gray ──► FFT2D ──► Lowpass Filter ──► Norm ──┘
```

El 4to canal es un **filtro pasa-bajas de Fourier** que captura la estructura global de la imagen, eliminando detalles de alta frecuencia (texturas) para ayudar al modelo a enfocarse en siluetas y formas.

---

## ⚙️ Configuración de Entrenamiento

| Parámetro | Valor |
|---|---|
| **Optimizer** | AdamW (weight_decay=1e-2) |
| **LR Encoder** | 1e-5 (pre-trained, pasos pequeños) |
| **LR Decoder** | 1e-4 (entrenado from scratch) |
| **Batch Size** | 16 |
| **Loss** | CrossEntropyLoss + GeneralizedDiceLoss (schedule progresivo) |
| **CE Weights** | [1.0, 0.5, 2.0] (mascota, fondo, borde) |
| **Data Augmentation** | HorizontalFlip, RandomBrightnessContrast, CoarseDropout |
| **Mixed Precision** | BFloat16 (autocast) |

---

## 🚀 Uso Rápido

### Entrenamiento
```python
from utils.models.daowa_maadV3Prueba import Daowa_maadPrueba
from utils.models.datasets import get_dataloaders
from utils.losses.dice_loss import GeneralizedDiceLossFN
from utils.train.trainer import train_model

modelo = Daowa_maadPrueba(num_clases=3).to(device)
dataloaders = get_dataloaders(batch_size=16, num_workers=4)
train_model(modelo, loss_fn, optimizer, dataloaders, device, epochs=10)
```

### Inferencia
```python
from utils.inference.predict import prediccionPrueba

modelo.load_state_dict(torch.load('Models/ModeloPrueba2026-03-10.pth'))
prediccionPrueba(modelo, 'ruta/a/imagen.jpg', device)
```

### Comparar entrenamientos
```python
from utils.visualization.compare import comparar_entrenamientos

comparar_entrenamientos(
    ('logs/training_history2026-03-04.csv', 'Sin Transformer'),
    ('logs/training_history2026-03-04_2.csv', 'Con Transformer'),
    ('logs/training_history2026-03-10.csv', 'ConvNeXtV2 + Aug'),
)
```

---

## 🏷️ Versiones (Tags)

| Tag | Descripción |
|---|---|
| `v3.0-rc1` | Notebook monolítico pre-refactorización |
| `v3.0-rc2` | Código refactorizado en módulos + resultados con data augmentation |

---

## 📋 Requisitos

- Python 3.12+
- PyTorch 2.11+ (CUDA 13.0)
- timm, albumentations, torchinfo
- TensorBoard

```bash
pip install torch torchvision timm albumentations tensorboard torchinfo
```

---

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Ver [LICENSE](LICENSE) para más detalles.
