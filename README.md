# 🐾 Abbadon — Binary Pet Segmentation with Adversarial Fine-tuning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6-EE4C2C?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900?logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active%20Research-orange)

**Pixel-level binary segmentation of pets (dogs & cats) using a custom ConvNeXtV2 U-Net,
refined through an adversarial fine-tuning pipeline to eliminate false positives in complex scenes.**

</div>

---

## 📖 What does this project solve?

Abbadon segments pets from background at the pixel level. Given any image, the model outputs a
binary mask where `1 = pet` and `0 = background`.

The hard part is not detecting pets — it's learning what is **not** a pet.
The model originally confused **real fur textures** (fur-lined coats, fluffy sofas, wool blankets)
with actual animals, because those textures share the same frequency signatures in feature space.

The adversarial fine-tuning pipeline solves this by teaching the model to discriminate:
it shows the model thousands of fur-like textures and human silhouettes in fur clothing
(all labeled as background), with a controlled positive/negative ratio enforced per batch.

---

## 🏛️ Architecture

### Encoder — `ConvNeXtV2 Tiny` (pretrained on ImageNet-21k)
Replaces a ResNet trained from scratch. Transfer learning from large-scale pretraining
gives the model rich semantic representations from the start, converging in half the epochs.

### Decoder — Custom U-Net style
Skip connections from the encoder feed into residual decoder blocks with **Attention Gates**,
which learn to focus on relevant spatial regions at each resolution level.

### Input
```
Image [B, 3, 256, 256]
    ↓
ConvNeXtV2 Encoder → [96] → [192] → [384] → [768]
                        ↕       ↕       ↕       ↕
                     AttentionGate + BloqueResidual + UpSampling
                                          ↓
                              Output logit [B, 1, 256, 256]
                              sigmoid > 0.5 → binary mask
```

---

## 🎓 Training Strategy: 3-Phase Pipeline

```
Phase 1 — Knowledge Distillation from SAM 2
    YOLO v8 ──► bounding box ──► center point prompt
                                        ↓
                                    SAM 2 (teacher)
                                        ↓
                               soft logit distribution
                               ("dark knowledge")
                                        ↓
                         KL Divergence  ──►  Daowa_maad (student)
                         The student learns the teacher's uncertainty,
                         not just the hard binary label.

Phase 2 — Supervised refinement on Oxford-IIIT Pet
    Standard Dice + Boundary loss on curated ground-truth masks.

Phase 3 — Adversarial Fine-tuning
    Hard negatives from ADE20K (textures + persons in fur clothing)
    with a controlled positive/negative ratio per batch.
    → See section below.
```

### Why knowledge distillation instead of just using SAM?

SAM 2 is a massive foundation model (~300M parameters) that requires **interactive prompts** to run —
it cannot be deployed end-to-end without a prompt pipeline.
Daowa_maad is a lightweight student (~15M parameters) that runs **prompt-free** on any image.

The distillation step transfers SAM's semantic understanding into the student by training on
SAM's full **probability distribution** (soft labels), not just the binary mask.
This is what Hinton called *"dark knowledge"* — the relative probabilities that reveal
how the teacher reasons about uncertain regions (edges, occlusions, ambiguous textures).

The prompt heuristic to drive SAM during distillation:
1. **YOLOv8** detects the animal and returns a bounding box
2. The center point of the bounding box is extracted
3. That point is passed to **SAM 2** as a foreground prompt
4. SAM returns the mask with the highest confidence score
5. The logit distribution (pre-sigmoid) is used as the soft target for KL Divergence

This is the same principle Tesla uses to distill their large offline perception models
into lightweight real-time networks that run on vehicle hardware.



---

## 🔬 How it was built — iterative approach

| Version | Encoder | Key change | Val Acc | mIoU |
|---|---|---|:---:|:---:|
| `Daowa_maad` | ResNet (scratch) | Baseline | 92.01% | 78.07% |
| `Daowa_maadV2` | ResNet + Transformer | Transformer bottleneck | 91.89% | 77.80% |
| `Daowa_maadV3-rc3` | **ConvNeXtV2 Tiny** | Pretrained encoder + Fourier channel | **92.83%** | **79.46%** |
| `Daowa_maadV3 (binary)` | ConvNeXtV2 Tiny | Binary task + adversarial fine-tuning | — | **~0.984 score** |

> The Transformer bottleneck did not improve over the base ResNet, but ConvNeXtV2
> with pretrained weights reached better metrics in half the epochs.

---

## ⚔️ Adversarial Fine-tuning Pipeline

### The problem
After reaching high IoU on Oxford-IIIT Pet, real-world testing revealed false positives:
fur coats, fluffy sofas, and wool blankets were predicted as pets.
This is an **out-of-distribution (OOD)** issue — the model had never seen
"fur-like texture attached to a non-animal silhouette."

### The solution — not post-processing, genuine learning
Instead of suppressing false positives with a secondary detector (e.g. YOLO),
the model was fine-tuned on a curated adversarial mix:

| Negative source | Why it's hard |
|---|---|
| ADE20K — sofas, carpets, blankets | Same fur-like texture as pets |
| ADE20K — persons with fabric/clothing | Human silhouette + textile, closest to the fur coat failure case |

### Engineering components

**`NegativeAwareBatchSampler`**
Without it, ~90% of each batch would be Oxford pets and the model would ignore the negatives.
This sampler enforces a fixed ratio (e.g. 13 pos + 3 neg) regardless of dataset proportions.

**`BurnInAdversarialLoss`**
Combines Dice loss and Boundary loss (computed on pre-computed SDF maps) with epoch-dependent weights:
- Early epochs → Dice dominates: stable convergence, no catastrophic forgetting of prior training
- Late epochs → Boundary grows: sharpens contour fidelity and suppresses false activations

**`precompute_sdfs.py`**
Signed Distance Fields were originally computed on-the-fly with `scipy` — synchronous,
CPU-bound, causing GPU stalls every batch. This script pre-computes all SDFs to `.npy` files.
The DataLoader reads them from disk, eliminating the bottleneck entirely.

**`generate_person_negatives_csv.py`**
Scans ADE20K for scenes containing a person co-occurring with fabric/textile classes.
These are the highest-quality hard negatives: human silhouette + real textile,
directly targeting the failure mode.

---

## 📊 Results

| Metric | Base model | After adversarial fine-tuning |
|---|:---:|:---:|
| **IoU_pos** (pets, Oxford val) | ~0.87 | ~0.85 |
| **IoU_neg** (distractors, ADE20K val) | — | ~0.984 |
| **Global score** (0.7×neg + 0.3×pos) | — | **~0.984** |

> Score formula weights adversarial suppression (70%) over positive detection (30%),
> since the model's positive detection was already strong.

---

## 🏗️ Project Structure

```
Abbadon/
├── source/
│   ├── 📓 train_daowa_adversarial.ipynb   # Adversarial fine-tuning notebook
│   ├── 📓 Daowa_maadV3.ipynb              # Original multi-class training
│   ├── 📓 prueba.ipynb                    # Inference & visual validation
│   └── utils/
│       ├── models/
│       │   ├── blocks.py                  # BloqueResidual, AttentionGates, UpSampling
│       │   └── daowa_maadV3Prueba.py      # Daowa_maadPrueba (binary, ConvNeXtV2)
│       ├── losses/
│       │   ├── adversarial_loss.py        # BurnInAdversarialLoss
│       │   └── boundary_loss.py           # SDF-based boundary loss
│       ├── train/
│       │   ├── adversarial_dataset.py     # AdversarialPetDataset + NegativeAwareBatchSampler
│       │   └── trainer_adversarial_v2.py  # trainer + evaluate_model + predict_random_samples
│       └── scripts/
│           ├── precompute_sdfs.py
│           ├── generate_ade20k_csv.py
│           └── generate_person_negatives_csv.py
```

---

## 🚀 Quick Start

```bash
# 1. Pre-compute SDF maps (one time)
python source/utils/scripts/precompute_sdfs.py

# 2. Generate adversarial negative CSVs (one time)
python source/utils/scripts/generate_ade20k_csv.py
python source/utils/scripts/generate_person_negatives_csv.py --strict
```

```python
# 3. Fine-tune adversarially
import pandas as pd
import torch
from utils.train import get_adversarial_dataloaders, train_model
from utils.models import Daowa_maadPrueba

device = torch.device("cuda")
modelo = Daowa_maadPrueba(num_clases=1)
modelo.load_state_dict(torch.load("Models/your_base_model.pth", map_location=device))

loaders = get_adversarial_dataloaders(
    df_oxford=pd.read_csv("labels/dataset_generated.csv"),
    df_ade20k=pd.read_csv("labels/ade20k_adversarial_train.csv"),
    df_personas=pd.read_csv("labels/ade20k_person_negatives.csv"),
    oxford_dir="data/oxford",
    ade20k_dir="data/ADEChallengeData2016",
    batch_size=16, num_pos_per_batch=13,
)

optimizer = torch.optim.AdamW(modelo.parameters(), lr=5e-6, weight_decay=1e-2)
train_model(modelo, loaders["train"], loaders["val"], optimizer, device, epochs=10)
```

```python
# 4. Evaluate and visualize
from utils.train import evaluate_model, predict_random_samples

metrics = evaluate_model(modelo, loaders["val"], device)
fig     = predict_random_samples(modelo, loaders["val"], device, n_samples=8)
```

---

## ⚙️ Training Configuration

| Parameter | Value |
|---|---|
| **Optimizer** | AdamW (lr=5e-6, weight_decay=1e-2) |
| **Scheduler** | CosineAnnealingLR (T_max=10, eta_min=1e-7) |
| **Effective batch** | 64 (batch=16 × accum_steps=4) |
| **Loss** | BurnInAdversarialLoss (Dice + Boundary SDF) |
| **Mixed Precision** | BFloat16 (torch.amp.autocast) |
| **Datasets** | Oxford-IIIT Pet (positives) + ADE20K (negatives) |

---

## 📋 Requirements

```bash
pip install torch torchvision timm albumentations tensorboard torchinfo scipy rich pandas
```

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.
