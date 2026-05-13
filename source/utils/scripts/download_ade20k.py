import os
import urllib.request
import zipfile
import sys

def reporte_descarga(bloque_n, tamano_bloque, tamano_total):
    descargado = bloque_n * tamano_bloque
    porcentaje = 100 * descargado / tamano_total
    # Evitar superar el 100% y actualizar en la misma linea
    if porcentaje > 100:
        porcentaje = 100
    sys.stdout.write(f"\rDescargando ADE20K: {descargado/(1024*1024):.1f} MB / {tamano_total/(1024*1024):.1f} MB ({porcentaje:.1f}%)")
    sys.stdout.flush()

def main():
    url = "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"
    # Guardaremos esto en la carpeta recomendada: source/data/ADE20K
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    zip_path = os.path.join(data_dir, "ADEChallengeData2016.zip")
    extract_dir = os.path.join(data_dir, "ADE20K")
    
    print("Iniciando infraestructura de descarga del MIT CSAIL...")
    
    if not os.path.exists(zip_path):
        print(f"Descargando el dataset pesado (900 MB) localmente...")
        urllib.request.urlretrieve(url, filename=zip_path, reporthook=reporte_descarga)
        print("\nDescarga finalizada.")
    else:
        print("El archivo ZIP de ADE20K ya existe en tu disco duro.")
        
    print("Extrayendo archivos pesados (esto puede tardar un par de minutos)...")
    if not os.path.exists(os.path.join(extract_dir, "ADEChallengeData2016", "images")):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print("¡Extracción completa y exitosa!")
    else:
        print("Los datos ya se encontraban extraídos.")
        
    print("-" * 50)
    print("RUTAS PARA TU DATASET:")
    print(f"Imágenes: {os.path.join(extract_dir, 'ADEChallengeData2016', 'images', 'training')}")
    print(f"Máscaras: {os.path.join(extract_dir, 'ADEChallengeData2016', 'annotations', 'training')}")
    print("-" * 50)

if __name__ == "__main__":
    main()
