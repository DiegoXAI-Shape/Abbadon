#!/bin/bash
current_dir=$(pwd)


echo "Iniciando con el proceso de obtención de umbral y acto seguido la generación del archivo de imágenes filtradas para reproducibilidad"

python3 "$current_dir/check_threshold.py"

echo "Se ha completado el proceso"
