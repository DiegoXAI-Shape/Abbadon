#!/bin/bash

cd ..

labeldir_path="$(pwd)/labels"
data_path="$(pwd)/data/petimages"

find $data_path -maxdepth 1 -type d | while IFS= read -r directory; do
	directory_name=$(basename "$directory")
	echo 'filename,label' >> "$labeldir_path/dataset_mendicantV3.csv"
	if [[ $directory_name == "Cat" || $directory_name == "Dog" ]]; then
		for file in "$directory"/*; do
			if [[ -f $file ]]; then
				filename=$(basename "$file")
				echo "$filename,$directory_name" >> "$labeldir_path/dataset_mendicantV3.csv"
			fi
		done
	fi
done

echo "Se ha creado el dataset para las imágenes del dataset $data_path"

echo "Se ha creado en la ruta: $labeldir_path"
