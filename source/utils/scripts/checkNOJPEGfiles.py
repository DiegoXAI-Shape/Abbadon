import shutil
from pathlib import Path
import os

base_dir = Path("../data/PetImages")
no_jpeg_dir = Path("../data/PetImages/imagenesNoJPEG")
no_jpeg_dir.mkdir(exist_ok=True)

def main():
    count = 0
    moved = 0
    jpeg_firma = b"\xff\xd8\xff"

    print("Iniciando el proceso de chequeo de bits", flush=True)

    for img_path in base_dir.rglob("*"):
        if img_path.is_file():
            count += 1
            try:
                with open(img_path, 'rb') as f:
                    header = f.read(3)
                    if header != jpeg_firma:
                        # Si no empieza con los bytes de JPEG, pal pasillo
                        dest = no_jpeg_dir / img_path.name
                        shutil.move(str(img_path), str(dest))
                        moved += 1
            except Exception as e:
                print(f"Ha ocurrido un error en {img_path}: {e}")
    
    print(count, moved)

if __name__ == "__main__":
    main()
