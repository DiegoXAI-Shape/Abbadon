#!/bin/bash

cd ..
images_dir="$(pwd)/data/petimages/imagenesNoJPEG"
cd $images_dir

find . -type f | while read -r archivo; do
	name="${archivo%.*}"
	new_path="${name}.jpg"

	convert "$archivo" "$new_path"

	echo "Se ha convertido el archivo $archivo a $new_path"

	if [ -f "$new_path"]; then
		rm "$archivo"
	fi
done

#Nunca lo ejecutes JAJAJ, mejor borro esas imágenes porque ya las revolví, valgo v.
