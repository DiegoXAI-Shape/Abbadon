#!/bin/bash

data_path="/mnt/c/users/PC/.cache/kagglehub/datasets/shaunthesheep/microsoft-catsvsdogs-dataset/versions/1"

cd ..
general_path=$(pwd)
target_dir="$general_path/data/PetImages/*"


echo "Iniciando el copiado de archivos..."

cp -r "$data_path/PetImages" "$target_dir"

echo "Se ha terminado"
