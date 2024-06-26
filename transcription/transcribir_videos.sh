#!/bin/bash

# Definimos las rutas de las carpetas
VIDEOS_DIR="../videos"
TRANSCRIPCIONES_DIR="./transcripciones"
CURRENT_DIR=$(pwd)

# Verificamos que la carpeta de videos existe
if [ ! -d "$VIDEOS_DIR" ]; then
  echo "La carpeta '$VIDEOS_DIR' no existe. Por favor, crea la carpeta y coloca los videos en ella."
  exit 1
fi

# Verificamos que la carpeta de transcripciones existe, si no, la creamos
if [ ! -d "$TRANSCRIPCIONES_DIR" ]; then
  mkdir -p "$TRANSCRIPCIONES_DIR"
fi

# Procesamos cada archivo de video en la carpeta de videos
for video in "$VIDEOS_DIR"/*; do
  if [ -f "$video" ]; then
    # Extraemos el nombre del archivo sin la extensión
    base_name=$(basename "$video" | sed 's/\.[^.]*$//')
    # Definimos el nombre del archivo de salida .mp3
    output_audio="${CURRENT_DIR}/${base_name}.mp3"
    # Definimos el nombre del archivo de salida .json
    output_json="${CURRENT_DIR}/${base_name}.json"
    
    # Ejecutamos el comando ffmpeg para convertir el video a audio .mp3
    ffmpeg -y -hide_banner -loglevel error -i "$video" -c:v copy -af loudnorm=I=-23:LRA=7:TP=-2.0:measured_I=-17.31:measured_LRA=5.71:measured_TP=-1.35 "$output_audio"
    echo "Procesado: $video -> $output_audio"
    
    # Ejecutamos el comando whisperx en el archivo de audio .mp3
    whisperx --model large --output_format json --vad_onset 0.2 "$output_audio" -o "$CURRENT_DIR"
    echo "Procesado con whisperx: $output_audio -> $output_json"
  fi
done

# Ejecutamos el script Python leer_json.py para cada archivo .json
for json_file in "$CURRENT_DIR"/*.json; do
  if [ -f "$json_file" ]; then
    # Ejecutamos el script Python leer_json.py en el archivo .json
    python3 leer_json.py "$json_file"
    echo "Procesado con leer_json.py: $json_file"
    
    # Movemos el archivo .txt resultante a la carpeta de transcripciones
    base_name=$(basename "$json_file" .json)
    output_txt="${CURRENT_DIR}/${base_name}-transcript.txt"
    
    if [ -f "$output_txt" ]; then
      mv "$output_txt" "$TRANSCRIPCIONES_DIR/"
      echo "Añadido: $output_txt -> $TRANSCRIPCIONES_DIR/"
    else
      echo "Error: $output_txt no encontrado."
    fi
  fi
done

echo "Procesamiento de transcripción de audio completado."

