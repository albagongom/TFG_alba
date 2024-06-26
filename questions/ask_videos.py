import torch
import av
import numpy as np
import time
import glob
import os
import argparse
import sys
import shutil
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration, BitsAndBytesConfig

def read_input_folder(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('folder_path', type=str, help='Proporciona la ruta a la carpeta que contiene los videos que desea procesar.', nargs='?')
    args = parser.parse_args()
    if args.folder_path is None:
        parser.print_help(sys.stderr)
        sys.exit(1)
    return args.folder_path

def read_video_pyav(container, indices):
    '''
    Decodificar el video con el decodificador PyAV.

    Args:
        container (av.container.input.InputContainer): Contenedor de PyAV.
        indices (List[int]): Lista de índices de fotogramas a decodificar.

    Returns:
        np.ndarray: Array de numpy de fotogramas decodificados con forma (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]

    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)

    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


# Funcion que le pasa las preguntas a un video y obtiene las respuestas
def ask_video_questions (video_path, questions, model, processor):

    container = av.open(video_path)

    # Toma 8 frames de todo el video
    total_frames = container.streams.video[0].frames
    indices = np.arange(0, total_frames, total_frames / 8).astype(int)
    clip = read_video_pyav(container, indices)

    processor.tokenizer.padding_side = "left"
    
    responses = []
    for question in questions: 

        prompt = f"USER: <video>{question} ASSISTANT:"
        inputs = processor(prompt, videos=clip, return_tensors="pt").to(model.device)
        
        generate_kwargs = {"max_new_tokens":500, "do_sample":False}

        output = model.generate(**inputs, **generate_kwargs)
        generated_text = processor.batch_decode(output, skip_special_tokens=True)

        responses.append(generated_text[0])
    
    return responses


# Procesar todos los videos
def process_videos (videos_dir, input_file, model, processor, output_folder):
    
    # Crear la carpeta 'responses' si no existe
    #if os.path.exists(output_folder):
    #    shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    # Obtener los videos de la carpeta 'videos'
    videos = glob.glob(os.path.join(videos_dir, "*"))
    if not videos: 
        return "No se encontraron videos en la carpeta."
    
    # Obtener las preguntas
    with open(input_file, 'r') as file: 
        questions = file.read().splitlines()
    
     # Obtener las respuestas para todos los videos y guardarlas en archivos separados
    for video in videos:
        
        video_name = os.path.splitext(os.path.basename(video))[0]
        responses = ask_video_questions(video, questions, model, processor)
        
        response_file_path = os.path.join(output_folder, f"responses-{video_name}.txt")
        with open(response_file_path, 'w') as file:
            for i, answer in enumerate(responses):
                file.write(f"Pregunta {i+1}:\n{answer}\n")

def main(): 

    start_time = time.time() # Para obtener el tiempo total que tarda en ejecutarse el programa
         
    videos_dir = read_input_folder()
    input_file = "./questions.txt"
    output_dir = "./responses"

    # Verificar existencia de la carpeta videos y que contenga elementos
    if not os.path.exists(videos_dir):
        print("No existe la carpeta 'videos'. Necesita ser creada y contener videos para poder ejecutar el código de manera correcta.")
        return
    elif not os.listdir(videos_dir):
        print("Insertar los videos a procesar dentro de la carpeta 'videos'.")
        return

    # Verificar existencia del archivo questions.txt
    if not os.path.isfile(input_file):
        print("No se ha encontrado el archivo 'questions.txt'. Es necesario que exista e incluya las preguntas que se quieren realizar.")
        return

    quantization_config = BitsAndBytesConfig (
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
    )

    model_id = "LanguageBind/Video-LLaVA-7B-hf"
    processor = VideoLlavaProcessor.from_pretrained(model_id)
    model = VideoLlavaForConditionalGeneration.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto")

    process_videos (videos_dir, input_file, model, processor, output_dir)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds.\n")
    print(f"Procesamiento de video con VideoLLaVA completado. Las respuestas se encuentran en la carpeta {output_dir}\n")

if __name__ == "__main__":
    main()


