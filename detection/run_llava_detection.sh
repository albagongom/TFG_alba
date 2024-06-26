#!/bin/bash

# Definimos la ruta de la carpeta de videos
VIDEOS_DIR="../detect-videos"
CURRENT_DIR=$(pwd)

# Verificamos que la carpeta de videos existe
if [ ! -d "$VIDEOS_DIR" ]; then
  echo "La carpeta 'videos' no existe. Por favor, crea la carpeta y coloca los videos en ella."
  exit 1
fi

# Procesamos cada archivo de video en la carpeta de videos
for video in "$VIDEOS_DIR"/*; do
  if [ -f "$video" ]; then
    # Extraemos el nombre del archivo sin la extensión
    base_name=$(basename "$video" | sed 's/\.[^.]*$//')
    
    # Creamos una carpeta para almacenar los frames del video
    output_frames_dir="${CURRENT_DIR}/frames-${base_name}"
    
    # Borramos el directorio si ya existe y creamos uno nuevo
    rm -rf "$output_frames_dir"
    mkdir -p "$output_frames_dir"
    
    # Ejecutamos el comando ffmpeg para extraer frames del video
    ffmpeg -hide_banner -i "$video" -q:v 0 -vf fps=1 "${output_frames_dir}/frame%d.png"
    echo -e "Procesado: $video -> Frames en output_frames_dir.\n"
  
    # Ejecutamos el script Python face-detector.py en la carpeta de frames
    python3 face_detector.py "$output_frames_dir"
    echo -e "Ejecutado face-detector.py.\n"

     # Definimos la carpeta para almacenar las caras detectadas
    output_faces_dir="${CURRENT_DIR}/faces-${base_name}"

    # Ejecutamos el script Python look-at-camera.py en la carpeta de faces
    python3 look_at_camera.py "$output_faces_dir"
    echo -e "Ejecutado look_at_camera.py.\n"
  
  fi
done

echo "Procesamiento deteccion_llava.sh completado."
