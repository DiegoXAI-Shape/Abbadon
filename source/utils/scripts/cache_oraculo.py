import os
import cv2
import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

def run_cache():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    
    # Cargar el oráculo
    oracle_path = 'source/weights/Daowa_Oracle_Frozen.pt'
    if not os.path.exists(oracle_path):
        print("Oráculo no encontrado. Ejecuta export_oracle.py primero.")
        return
        
    oracle = torch.jit.load(oracle_path, map_location=device)
    oracle.eval()
    
    # Rutas
    csv_path = 'source/labels/dataset_mendicantV3.csv'
    img_dir = 'source/data/PetImages'
    mask_dir = 'source/data/PetMasks'
    
    os.makedirs(os.path.join(mask_dir, 'Cat'), exist_ok=True)
    os.makedirs(os.path.join(mask_dir, 'Dog'), exist_ok=True)
    
    df = pd.read_csv(csv_path)
    
    tf = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print(f"Iniciando pre-cálculo de máscaras para {len(df)} imágenes...")
    
    with torch.no_grad():
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            img_name = row['filename']
            label_str = row['label'].capitalize().strip() # Seguridad extra
            
            img_path = os.path.join(img_dir, label_str, img_name)
            mask_path = os.path.join(mask_dir, label_str, img_name.replace('.jpg', '.png'))
            
            if os.path.exists(mask_path):
                continue # Skip if already processed
                
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue
                
            from PIL import Image
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb).resize((384, 384))
            
            img_tensor = tf(img_pil).to(device).unsqueeze(0)
            
            oracle_logits = oracle(img_tensor)
            if oracle_logits.shape[1] > 1:
                oracle_logits = oracle_logits[:, 0:1, :, :]
            
            daowa_mask = torch.sigmoid(oracle_logits)
            
            # Convertir probabilidad [0, 1] a imagen [0, 255]
            mask_numpy = (daowa_mask.squeeze().cpu().numpy() * 255).astype(np.uint8)
            
            cv2.imwrite(mask_path, mask_numpy)

if __name__ == "__main__":
    run_cache()
    print("¡Pre-cálculo finalizado exitosamente!")
