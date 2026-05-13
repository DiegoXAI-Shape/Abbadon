import torch
import pandas as pd
import os
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader, Dataset

current_dir = os.path.abspath(os.path.dirname(__file__))
img_dir = os.path.join(current_dir, '..', '..', 'data', 'image', 'images')
mask_dir = os.path.join(current_dir, '..', '..', 'data', 'image', 'masks')
labels_dir = os.path.join(current_dir, '..', '..', 'labels', 'dataset_daowaV2.csv')

class CustomDS(Dataset):
    def __init__(self, dataframe, img_dir, mask_dir, transformador = None):
        self._dataframe = dataframe
        self._img_dir = img_dir
        self._mask_dir = mask_dir
        self._transformador = transforms.Compose([
            transforms.Resize((192, 192)),
            transforms.ToTensor()
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

def main():
    df = pd.read_csv(labels_dir)
    df["raza"] = df['file'].apply(lambda x: "_".join(x.split("_")[:-1])) 

    transform = transforms.Compose([
        transforms.Resize((512, 512), interpolation=InterpolationMode.NEAREST),
        transforms.ToTensor()
    ])
    ds = CustomDS(df, img_dir=img_dir, mask_dir=mask_dir)
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    sum_ = torch.zeros(3)
    sum_sq = torch.zeros(3)
    num_pixels = len(ds) * 192 * 192  # Total de píxeles en el dataset

    for data, _ in dl:
        sum_ += data.sum(dim=[0, 2, 3])
        sum_sq += (data**2).sum(dim=[0, 2, 3])

    mean = sum_ / num_pixels
    std = torch.sqrt((sum_sq / num_pixels) - mean**2)

    print(f"Mean: {mean}")
    print(f"Std: {std}")

if __name__ == '__main__':
    main()