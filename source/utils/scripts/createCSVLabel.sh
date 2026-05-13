#!/bin/bash

cd ..
general_path=$(pwd)
images_path="$general_path/data/petimages"
labels_path="$general_path/labels"

find $images_path -maxdepth 1 -type d | while IFS= read -r directory; do
	directory_name=$(basename "$directory")
	if [[ $directory_name != "petimages" && $directory_name != "imagenesNoJPEG" ]]; then
		echo "filename, label" >> "$labels_path/labels_$directory_name.csv"
		for file in "$directory"/*; do
			if [[ -f $file ]]; then
				filename=$(basename "$file")
				echo "$filename,$directory_name" >> "$labels_path/labels_$directory_name.csv"
			fi
		done
		echo "Se ha creado el label para el directorio $directory_name"
	fi
done
