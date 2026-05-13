import os
import sys
import torch

current_dir= os.path.abspath(os.path.dirname(__file__))
utils_path = os.path.join(current_dir, '..', 'utils')
models_path = os.path.join(current_dir, '..', '..', 'Models')
data_path = os.path.join(current_dir, '..', 'data', 'PetImages')

sys.path.append(utils_path)

def main():
    from models.utils_xai import gradCam
    from models.utils_med import Mendicant_Biasv3

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    modelo = Mendicant_Biasv3()
    modelo.load_state_dict(torch.load(os.path.join(models_path, 'Mendicant_BiasV3.pth'), weights_only=True))

    data = {
        "filename": "7.jpg",
        'label':['Dog']
    }

    print("Empezando el proceso de generación de GradCAM")

    gradCam(modelo, device, 1, data, data_path)

    print("Ha terminado el proceso")


if __name__ == "__main__":
    main()
