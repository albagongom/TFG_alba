import os
import time
import argparse
from scenedetect import open_video, SceneManager
from scenedetect.detectors import AdaptiveDetector

THRESHOLD = 5.0

def detect_scenes(video_path, threshold=5.0):

    video = open_video(video_path)

    scene_manager = SceneManager()
    scene_manager.add_detector(AdaptiveDetector(adaptive_threshold=THRESHOLD))
    scene_manager.detect_scenes(video, show_progress=True)
    
    scene_list = scene_manager.get_scene_list()
    return scene_list


def save_scene_times (scene_list, output_path):
    
    with open(output_path, 'w') as f:
        
        for i, scene in enumerate(scene_list):
            start_time = scene[0].get_timecode()
            end_time = scene[1].get_timecode()
            
            f.write(f"Escena {i+1}: Comienzo {start_time}, Fin {end_time}\n")


def process_videos (folder_path):

    output_folder = "./scenes-times"
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(folder_path):
        
         if os.path.isfile(os.path.join(folder_path, filename)):
            
            video_path = os.path.join(folder_path, filename)
            output_filename = f"scenes_times_{os.path.splitext(filename)[0]}.txt"
            output_path = os.path.join(output_folder, output_filename)
            
            print(f"Procesando {filename}")
            start_time = time.time()
            
            scene_list = detect_scenes(video_path)
            if scene_list:
                save_scene_times(scene_list, output_path)
            
            end_time = time.time()
            
            print(f"Proceso de detección de escenas del vídeo {video_path} terminado. La salida se encuentra en {output_path}.")
            print(f"Tiempo empleado: {end_time - start_time:.2f} segundos.")

def main():

    parser = argparse.ArgumentParser(description="Detecta escenas dentro de un vídeo y devuelve los tiempos de comienzo y fin de las distintas escenas.")
    parser.add_argument("folder_path", type=str, help="Ruta al directorio que contiene los vídeos que se quieren procesar.")
    args = parser.parse_args()

    process_videos(args.folder_path)

if __name__ == "__main__":
    main()

