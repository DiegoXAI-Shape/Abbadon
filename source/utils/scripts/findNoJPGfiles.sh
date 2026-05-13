#!/bin/bash

current_path=$(pwd)
data_path="$current_path/../../data/PetImages"
cd $data_path

mkdir "imagenesNoJPEG"
dir_path="$data_path/imagenesNoJPEG"

find . -type f | while read -r archivo; do
	if ! file -b --mime-type "$archivo" | grep -qE "image/jpeg"; then
		echo "Moviendo archivo no-JPEG: $archivo"
		mv "$archivo" "$dir_path"
	fi

done

echo "¡Proceso completado!"
