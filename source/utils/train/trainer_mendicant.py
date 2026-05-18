import os
import csv
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

# Intentar importar las arquitecturas
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.models.mendicant_bias import MendicantBias_ConvNeXt

class MendicantTrainer:
    def __init__(self, mendicant_model, oracle_model, loaders, device, save_dir="source/Models"):
        self.device = device
        self.mendicant = mendicant_model.to(self.device)
        self.oracle = oracle_model.to(self.device) # Ya viene congelado (TorchScript o eval sin gradientes)
        
        self.loaders = loaders
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Hyperparametros
        self.epochs = 50
        # Optimizador moderno para ConvNeXt
        # ¡CORRECCIÓN! Usar 1e-3 con pesos pre-entrenados causa "Catastrophic Forgetting" (destruye los pesos y empieza a adivinar 50/50).
        # Para fine-tuning de ConvNeXt, la tasa de aprendizaje correcta es entre 1e-4 y 5e-5.
        self.optimizer = AdamW(self.mendicant.parameters(), lr=5e-5, weight_decay=0.05)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=1e-6)
        
        # Loss de Clasificación
        self.criterion = nn.CrossEntropyLoss()
        
        # Peso de la penalización L2 del Attention Gate (Defensa B)
        self.lambda_gate = 0.01 
        # Probabilidad de apagar colores (Defensa C)
        self.drop_rgb_prob = 0.15 
        
        self.best_acc = 0.0

    def run_epoch(self, epoch, phase):
        is_train = (phase == 'train')
        self.mendicant.train() if is_train else self.mendicant.eval()
        
        running_loss = 0.0
        running_ce_loss = 0.0
        running_gate_loss = 0.0
        corrects = 0
        total = 0
        
        loader = self.loaders[phase]
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{self.epochs} [{phase.upper()}]")
        
        for inputs, masks, labels in pbar:
            inputs = inputs.to(self.device)
            masks = masks.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            with torch.set_grad_enabled(is_train):
                # 1. MENDICANT BIAS: Forward pass con Gate y RGB Dropout (Usamos la máscara cacheada del disco)
                current_drop_prob = self.drop_rgb_prob if is_train else 0.0
                logits = self.mendicant(inputs, masks, drop_rgb_prob=current_drop_prob)
                
                # 3. CÁLCULO DE LOSS (Loss Compuesto)
                ce_loss = self.criterion(logits, labels)
                gate_loss = self.mendicant.get_gate_regularization_loss()
                
                total_loss = ce_loss + (self.lambda_gate * gate_loss)
                
                # 4. BACKPROPAGATION
                if is_train:
                    total_loss.backward()
                    # Opcional: Gradient Clipping para mayor estabilidad
                    torch.nn.utils.clip_grad_norm_(self.mendicant.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    
            # Estadísticas
            running_loss += total_loss.item() * inputs.size(0)
            running_ce_loss += ce_loss.item() * inputs.size(0)
            running_gate_loss += gate_loss.item() * inputs.size(0)
            
            _, preds = torch.max(logits, 1)
            corrects += torch.sum(preds == labels.data)
            total += inputs.size(0)
            
            acc = (corrects.double() / total).item()
            pbar.set_postfix({"Loss": total_loss.item(), "Acc": acc})
            
        epoch_loss = running_loss / total
        epoch_ce_loss = running_ce_loss / total
        epoch_gate_loss = running_gate_loss / total
        epoch_acc = corrects.double() / total
        
        print(f"[{phase.upper()}] Total Loss: {epoch_loss:.4f} | CE Loss: {epoch_ce_loss:.4f} | Gate Reg: {epoch_gate_loss:.4f} | Acc: {epoch_acc:.4f}")
        return epoch_loss, epoch_acc.item()

    def train(self, history_file="source/logs/history_mendicant.csv"):
        # Preparar archivo de historial
        with open(history_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc', 'lr'])
            
        for epoch in range(1, self.epochs + 1):
            train_loss, train_acc = self.run_epoch(epoch, 'train')
            val_loss, val_acc = self.run_epoch(epoch, 'val')
            
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Guardar historial
            with open(history_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, train_loss, val_loss, train_acc, val_acc, current_lr])
                
            # Guardar mejor modelo
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                save_path = os.path.join(self.save_dir, f"BestScore_Mendicant_{val_acc:.4f}.pth")
                torch.save(self.mendicant.state_dict(), save_path)
                print(f"⭐ ¡Nuevo mejor modelo guardado! Accuracy: {val_acc:.4f}")

if __name__ == "__main__":
    from mendicant_dataset import get_mendicant_dataloaders
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Iniciando entrenamiento en: {device}")
    
    # 1. Cargar Oráculo Congelado (TorchScript)
    try:
        oracle = torch.jit.load('source/weights/Daowa_Oracle_Frozen.pt', map_location=device)
        print("✅ Oráculo TorchScript cargado perfectamente.")
    except Exception as e:
        print(f"Error cargando el oráculo: {e}")
        sys.exit(1)
        
    # 2. Instanciar Mendicant Bias (Aprendiz)
    mendicant = MendicantBias_ConvNeXt(pretrained=True)
    print("✅ Aprendiz ConvNeXtV2 Atto instanciado.")
    
    # 3. Preparar DataLoaders (Asegúrate de ajustar los nombres de tus CSV)
    csv_train = "source/labels/dataset_mendicantV3.csv" 
    csv_val = "source/labels/dataset_mendicantV3.csv" # TODO: Cambiar por tu CSV real de validación
    img_dir = "source/data/PetImages"
    
    loaders = get_mendicant_dataloaders(csv_train, csv_val, img_dir, batch_size=16, num_workers=4)
    print("✅ DataLoaders listos.")
    
    # 4. Iniciar Entrenamiento
    trainer = MendicantTrainer(mendicant, oracle, loaders, device)
    trainer.train()
