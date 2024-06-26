import time
import os
import argparse
import shutil
from scenedetect import open_video, SceneManager
from scenedetect.detectors import AdaptiveDetector
from scenedetect.video_splitter import split_video_ffmpeg

THRESHOLD = 5.0

def detect_scenes(video_path):

    video = open_video(video_path)

    scene_manager = SceneManager()
    scene_manager.add_detector(AdaptiveDetector(adaptive_threshold=THRESHOLD))
    scene_manager.detect_scenes(video, show_progress=True)
    
    scene_list = scene_manager.get_scene_list()
    return scene_list

def save_video_scenes(video_path, scene_list, output_video_folder):
    split_video_ffmpeg(video_path, scene_list, output_dir=output_video_folder, show_progress=True)
    

def process_videos (folder_path):

    output_folder = "./scene-clips"
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(folder_path):
        
         if os.path.isfile(os.path.join(folder_path, filename)):
            
            video_path = os.path.join(folder_path, filename)
            video_name = os.path.splitext(filename)[0]
            output_video_folder = os.path.join(output_folder, f"scenes-clips-{video_name}")
            
            if os.path.exists(output_video_folder):
                shutil.rmtree(output_video_folder)
            os.makedirs(output_video_folder, exist_ok=True)

            print(f"Procesando {filename}")
            start_time = time.time()
            
            scene_list = detect_scenes(video_path)
            
            if scene_list:
                save_video_scenes(video_path, scene_list, output_video_folder)

            end_time = time.time()
            
            print(f"Proceso de división de escenas del vídeo {video_path} terminado. La salida se encuentra en {output_video_folder}.")
            print(f"Tiempo empleado: {end_time - start_time:.2f} segundos.")

def main():

    parser = argparse.ArgumentParser(description="Detecta y divide las escenas de un vídeo y devuelve las escenas en archivos de vídeo individuales.")
    parser.add_argument("folder_path", type=str, help="Ruta al directorio que contiene los vídeos que se quieren procesar.")
    args = parser.parse_args()

    process_videos(args.folder_path)

if __name__ == "__main__":
    main()
