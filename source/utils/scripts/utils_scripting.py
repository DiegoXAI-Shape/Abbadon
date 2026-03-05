import os
import sys
import shutil
import json
from datetime import date
from PIL import Image
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import torch
from sklearn.metrics import f1_score
import warnings

#Solo para que me importe la arquitectura de mi modelo XD
current_path = os.path.dirname(os.path.abspath(__file__))
logs_path = os.path.join(current_path, '..', '..', 'logs')
models_path = os.path.join(current_path, '..', '..', '..', 'Models')
label_path = os.path.join(current_path, '..', '..', 'labels')
data_path = os.path.join(current_path, '..', '..', 'data', 'PetImages')

#Variables que ando usando todavía en desarrollo
dia = date.today()
warnings.simplefilter('error', Image.DecompressionBombError)
warnings.simplefilter('error', UserWarning)

#Hardcodeamos el mejor umbral que nos ha dado
mejor_umbral = 0.55

def setup_logger(name, path:str, level = logging.INFO):
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    
    full_log_path = os.path.join(path, name)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Crear el handler (el que escribe al archivo)
    handler = logging.FileHandler(full_log_path)        
    handler.setFormatter(formatter)

    # Crear el logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    #print(f"Ruta configurada: {path}")

    return logger

def move_Files(df, logger):
    """
    Attributes:
        - df: Es el dataframe filtrado, sobre el cual iteraremos para moverlo a una carpeta para revisarlos manualmente
    """

    #Generamos primero la carpeta
    dir_path = os.path.join(data_path, "revision")

    #Si no existe, crea una carpeta
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    print("Empezando a enviar los archivos con menos confianza a revisión manual: ")
    count = 0
    mapeado = {
        0: "Cat",
        1: 'Dog'
    }
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):


        origen = os.path.join(data_path, mapeado[row['true_labels']], row['filenames'])
        destino = os.path.join(dir_path, f"{row['filenames']}_{row['true_labels']}.jpg")
        try:
            shutil.move(origen, destino)
            logger.info(f"Se ha enviado el archivo {row['filename']} a {destino}")
            count += 1

        except Exception as ex:
            logger.error(f"ERROR al mover {row['filenames']}: {str(ex)}")

def setup_dl(data, batch):
    dl = DataLoader(dataset=data, batch_size=batch, shuffle=False, pin_memory=True, num_workers=4)
    return dl

def predict_dataset(modelo, dataloader, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """
    Attributes:
        - modelo: Clase del modelo generado y además que tiene los pesos pre-entrenados de Mendicant BiasV3
        - dataloader: Esta clase nos ayuda a pasar las imágenes al modelo (contiene todos los datos para ser predecidos)
        - device: Se define en la misma función, pero si el usuario desea cambiarlo, es libre de hacerlo
    """

    #Generamos la lista donde vamos a estar guardando las predicciones y una variable para saber cuántas correctas acertó
    results = []
    correct = 0
    total_samples = len(dataloader.dataset)

    #Ponemos el modelo en evaluación y lo mandamos a la GPU
    modelo.eval()
    modelo = modelo.to(device)

    #Ahora empezamos a predecir
    with torch.no_grad():
        #Iteramos sobre todo el dataloader
        for image, label, filename in dataloader:

            #Mandamos la imagen a la GPU, porque si no lo hacemos, nos va a dar el error: 
            #   - RuntimeError: Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the same
            image = image.to(device)

            #Ahora si pasamos la imagen a predecir
            output = modelo(image)
            pred = torch.nn.functional.softmax(output, dim = 1)
            confianza, _predict = torch.max(pred, dim = 1)

            #Iteramos sobre el batch de entrenamiento, es decir, vamos iterando sobre cada una de las imágenes del batch
            for i in range(image.size(0)):
                #Añadimos un diccionario a la lista, para en otra función, desempaquetar todo y meterlo a un df
                results.append({
                    "filename":filename[i],
                    "confianza":confianza[i].item(),
                    "predict_label":_predict[i].item(),
                    "true_label":label[i].item(),
                })

                #Si la etiqueta fue correcta, le suma uno
                if _predict[i] == label[i]:
                    correct += 1

        #Simplemente para darme una idea de cómo fue la predicción del modelo
        precision = (correct / total_samples) * 100

        #Retornamos los resultados y la precision
        return results, precision

def get_DF(lista:list):
    filenames, true, predict, confidence = [], [], [], []

    for diccionario in lista:

        filename = diccionario['filename']
        true_label = diccionario['true_label']
        predicted_label= diccionario['predict_label']
        confianza = diccionario['confianza']
        filenames.append(filename)
        true.append(true_label)
        predict.append(predicted_label)
        confidence.append(confianza)

    mydict = {
        "filenames":filenames,
        "true_labels":true,
        "predict_labels":predict,
        "confianzas":confidence
    }

    return pd.DataFrame(mydict)

def convert_to_JSON(diccionario:dict, dirname:str, filename:str, logger):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    
    path = os.path.join(dirname, filename)

    with open(path, 'w', encoding='utf-8') as archivo:
        json.dump(diccionario, archivo, indent = 4, ensure_ascii=False)
    
    logger.info(f"Se he enviado el diccionario resultante de chequeo de imágenes al archivo con ruta {path}")

    return print("Se ha enviado el JSON con información a el archivo correspondiente")

def check_images(images_dir:str, logger) -> dict:
    """
    Attributes:
        -Images_dir: Este es el PATH hacía donde se dirige PILLOW para abrir todas las imágenes y checar que realmente sea una imagen o que tan si quiera se pueda abrir

        -logger: Para mandar la información relevante al archivo Log
    """

    trash_dirs = ['imagenesNoJPEG', 'Noise', 'revision']

    revision_path = "../data/petimages/revision"
    if os.path.exists(revision_path):
        os.makedirs(revision_path, exist_ok=True)

    log_nt = {
        "Count":0,
        "Files_moved": [],
        "Checked":0
    }

    for dirpath, dirnames, filenames in os.walk(images_dir):
        for trash in trash_dirs:
            if trash in dirnames:
                dirnames.remove(trash)
        
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            #logger.info(full_path)

            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue;
                
            try:
                with Image.open(full_path) as img:
                    img.verify()
                    log_nt['Checked'] += 1
            
            except (IOError, UserWarning, Exception) as err:
                logger.info(f"Archivo corrupto o dañado: {full_path}")
                logger.error(f"Ha habido un error: {err}")
                carpeta_actual = os.path.basename(dirpath)
                shutil.move(full_path, r'..\data\PetImages\revision')
                log_nt['Files_moved'].append(f"{carpeta_actual}_{filename}")
                print("\t¡Woops! Ha ocurrido un error. \n\tConsulte el log correspondiente para más información...")

            except FileNotFoundError as e:
                print(f"No se ha encontrado el archivo. \n{e}")
            
            log_nt['Count'] += 1

    convert_to_JSON(log_nt, '../logs/JSON', f'diccionario_chequeo{dia}.json', logger=logger)

    return log_nt

def find_Threshold(df, logger):
    """
    Attributes:
        - df: Este es el dataframe de los resultados de las predicciones
        - logger: Este es logger que usaremos para estar enviando la información al archivo.log
    """

    thresholds = np.arange(0.5, 1.0, 0.01)
    total_samples = len(df)

    for t in thresholds:
        aceptados = df[df['confianzas'] >= t]
        total_aceptados = len(aceptados)

        if total_aceptados == 0:
            logger.error(f"El umbral {t} no ha tenido ningun aceptado.")
            continue

        verdad = aceptados[aceptados['true_labels'] == aceptados['predict_labels']]
        precision = (len(verdad) / total_aceptados) * 100
        cobertura = (total_aceptados / total_samples) * 100

        logger.info(f"Umbral: {t}, Precision: {precision}, Cobertura: {cobertura}")

def get_best_f1_score(dataframe, logger):
    umbrales = np.arange(0, 1, 0.01)
    mejor_f1 = 0
    mejor_umbral = 0

    dataframe['probs'] = np.where(
        dataframe['predict_labels'] == 1,
        dataframe['confianzas'],
        1- dataframe['confianzas']
    )

    probs = dataframe['probs'].values
    trues = dataframe['true_labels'].values

    for t in umbrales:
        
        pred_din = (probs >= t).astype(int)

        f1 = f1_score(trues, pred_din, average="macro")

        if f1 > mejor_f1:
            mejor_f1 = f1
            mejor_umbral = t
        
    logger.info(f"El modelo ha dado que el mejor umbral para maximizar la f1-score es: {mejor_umbral} con un valor de f1-score de: {mejor_f1}")
    
    return mejor_f1, mejor_umbral
