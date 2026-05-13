import torch
import os
import numpy as np
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import time
import sys

current_path = os.path.dirname(os.path.abspath(__file__))
utils_path = os.path.join(current_path, '..', 'utils')
models_path = os.path.join(current_path, '..', '..', 'Models')
labels_path = os.path.join(current_path, '..', 'labels')
#data_path = os.path.join(current_path, '..', 'data', 'image')
data_path = os.path.join(current_path, '..', 'data', 'PetImages')

sys.path.append(utils_path)
from utils_med import TransformerDaowa_maad, Mendicant_Biasv3, CustomDS_Med
"""
class Dataset_test(Dataset):
    def __init__(self, df, transformer = None):
        self.df = df
        self.data_path = data_path
        if not transformer:
            self.transformer = transforms.Compose([
            transforms.Resize((192, 192)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
            ])

        else:
            self.transformer = transformer
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.df.iloc[idx, 0]
        mask_path = self.df.iloc[idx, 1]
        img_path = os.path.join(self.data_path, 'images', image_path)
        maskPath = os.path.join(self.data_path, 'masks', mask_path)
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(maskPath)

        img = self.transformer(img)
        mask = transforms.functional.resize(mask, (192, 192), interpolation=transforms.InterpolationMode.NEAREST) 

        mask_np = np.array(mask)
        mask_tensor = torch.from_numpy(mask_np).long()
        mask_tensor = torch.clamp(mask_tensor, 0, 2)
        return img, mask_tensor
"""

def getDS(df):
    """
    Attributes:
        - df: Pandas' dataframe. This object has all data information, like: filename and mask 
    """

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df['label'],
        random_state=20
    )

    val_df, dev_df = train_test_split(
        val_df,
        test_size=0.1,
        stratify=val_df['label'],
        random_state=20
    )

    print(f"Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(dev_df)}")
    
    train_ds = CustomDS_Med(train_df,  images_dir=data_path, transform=None)
    val_ds = CustomDS_Med(val_df, images_dir=data_path, transform=None)
    dev_ds = CustomDS_Med(dev_df, images_dir=data_path, transform=None)
    
    return train_ds, val_ds, dev_ds

def getDataLoader(dataset):
    """
    Attributes:
        - dataset: This ds contains all information
    """
    
    dl = DataLoader(dataset, batch_size = 1, shuffle=True, num_workers=0)
    return dl

def timepred(model, dataloader):
    """
    Attributes:
        - model: This attribute contains the model, with the Daowa-Maad's architecture and him weights with bias
        - datalaoder: This attribute is the dataloader just to get a simple prediction, with the objective to get the time's model execution
    """
    #Definimos el dispositivo, en este caso es una GPU NVIDIA RTX 5070 TI
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    #Ponemos el modelo en evaluación
    model.eval()
    model = model.to(device)

    result = next(iter(dataloader))
    image = result[0]

    image = image.to(device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(image)
    
    if device.type == 'cuda': torch.cuda.synchronize()

    iterations = 100
    tic = time.time()
    
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(image)
    
    if device.type == 'cuda': torch.cuda.synchronize()
    toc = time.time()


    return float(toc-tic) / iterations

def main():
    modelo_piloto = Mendicant_Biasv3()
    modelo_piloto.load_state_dict(torch.load(os.path.join(models_path, 'Mendicant_BiasV3.pth'), weights_only=True))

    df = pd.read_csv(os.path.join(labels_path, 'labels.csv'))
    #df['raza'] = df['file'].apply(lambda x: "_".join(x.split("_")[:-1]))

    train_ds, _, _ = getDS(df)
    train_dl = getDataLoader(train_ds)

    tiempo = timepred(modelo_piloto, train_dl)
    #print(f"Tiempo de ejecución de Daowa-maad: {tiempo}")
    print(f"Tiempo de ejecución de Mendicant_Biasv3: {tiempo}")

if __name__ == '__main__':
    main()