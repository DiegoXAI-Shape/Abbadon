import json
import os
import sys
import time
import argparse
import torch

current_path = os.path.abspath(os.path.dirname(__file__))
utils_modelsPath = os.path.join(current_path, '..', 'models')

sys.path.append(utils_modelsPath)

def get_dataframe():
    print("Comenzando la importación de librerías...\n", flush=True)
    import pandas as pd
    
    #Importación de funciones importantes
    from utils_scripting import find_Threshold, predict_dataset, get_best_f1_score, setup_logger, setup_dl, get_DF, move_Files
    from utils_med import Mendicant_Biasv3, CustomDS_Med

    #Importación de variables
    from utils_scripting import label_path, models_path, data_path, logs_path, dia

    print("Importación de librerías terminada\n", flush=True)

    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    #
    #   En este código se usan valores pasados por scripting con Bash
    #   dichos valores se pasan como parámetros a la hora de ejecutar
    #   con el intérprete de python desde Bash
    #
    #   Ejemplo:
    #
    #   python3 ./check_threshold.py -name "dxngxrz" o -n "dxngxrz"
    #
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    #Obtención de los valores pasados por el script de bash
    """
    parser = argparse.ArgumentParser(description="Script para limpieza de datos en imágenes, en base a modelos pre-entrenados de IA hechos por mí")
    parser.add_argument("-l1", "--label_1", type=str, required=True, help="Nombre de la clase 1", default="Cat")
    parser.add_argument("-l2", "--label_2", type=str, required=True, help="Nombre de la clase 2", default="Dog")

    pars = parser.parse_args()
    label_1, label_2 = pars.label_1, pars.label_2
    """
    #Aquí lo que pasa es que si yo hago el cálculo el f1-score desde solo una neurona, obtendré el mejor umbral para esa neurona, pero como quiero
    #que haya un punto medio, por eso es que calculamos la f1-score de ambas neuronas, para después sumarlas y promediar cuál es el mejor umbral para ambas clases

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    modelo = Mendicant_Biasv3()
    modelo.load_state_dict(torch.load(os.path.join(models_path, 'Mendicant_BiasV3.pth'), weights_only=True))
    modelo = modelo.to(device)

    df = pd.read_csv(os.path.join(label_path, 'dataset_mendicantV3.csv'))
    logger = setup_logger('pruebas_linux.log', path = logs_path)

    ds = CustomDS_Med(df, data_path, transform=None, target_transform=None)
    dl = setup_dl(ds, batch=64)

    tic = time.time()
    results, precision = predict_dataset(modelo=modelo, dataloader=dl, device = device)
    toc = time.time()
    
    df_predict = get_DF(results)

    """PRUEBA"""
    
    f1_score_value, umbral = get_best_f1_score(df_predict, logger)

    data = {
        "Umbral óptimo": umbral,
        "F1-score":f1_score_value
    }

    with open(os.path.join(current_path, '..', '..', 'config', 'Config.json'), 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(type(df_predict[(df_predict['filenames'] == '7.jpg') & (df_predict['true_labels'] == '0')]['true_labels']))

    df_filtrado = df_predict[(df_predict['confianzas'] <= umbral) | (df_predict['true_labels'] != df_predict['predict_labels'])]
    df_filtrado.to_csv(os.  path.join(label_path, 'dataset_mendicantV3FILTRADO.csv'), index=False)

    #logger.info(f"Se ha enviado el dataset en forma de CSV a la dirección: {os.path.join(label_path, 'dataset_mendicantV3FILTRADO.csv')}")
    #logger.info(f"También se creo un archivo en la ruta {os.path.join(current_path, '..', 'config', 'Config.json')}")

    return df_filtrado


if __name__ == '__main__':

    from utils_scripting import move_Files, setup_logger
    from utils_scripting import logs_path
    
    info_logger = setup_logger('limpieza.log', logs_path)

    tic = time.time()
    dataframe = get_dataframe()
    move_Files(dataframe, info_logger)
    toc = time.time()
    print("$" * 50, "\n") 
    print(f"Tiempo de ejecución: {toc-tic} segundos\n")
    print("$" * 50, '\n')