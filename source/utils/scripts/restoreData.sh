#!/bin/bash

current_path=$(pwd)
original_data_path="/mnt/c/Users/PC/.cache/kagglehub/datasets/shaunthesheep/microsoft-catsvsdogs-dataset/versions/1/PetImages"
data_path="$current_path/../../data/"

#Nomás pa' hacer algo de programaación defensiva por si no existe la carpeta
mkdir -p "$data_path"

echo "Iniciando el proceso de mover archivos de $original_data_path a $data_path"

cp -r "$original_data_path" "$data_path"

echo "Ha finalizado el proceso de mover archivos"

#------------------------------------------------------------------------------------------------------------------------------

echo "Iniciando el proceso para filtrar archivos que sus metadatos no sean JPG"

bash "$current_path/findNoJPGfiles.sh"

echo "Proceso finalizado..."