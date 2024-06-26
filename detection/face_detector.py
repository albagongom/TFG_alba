import cv2
import supervision as sv
from ultralytics import YOLO
import os
import argparse
import sys
import shutil

# Funcion que permite ampliar el contexto de la imagen recortada

def expand_bbox(xyxy, img_width, img_height, margin=0.5):
    """
    Expande el cuadro delimitador en ambas direcciones.

    :param xyxy: Coordenadas del cuadro delimitador [x_min, y_min, x_max, y_max]
    :param img_width: Ancho de la imagen
    :param img_height: Altura de la imagen
    :param margin: Margen adicional como fracción del tamaño del cuadro delimitador
    :return: Coordenadas expandidas del cuadro delimitador [x_min, y_min, x_max, y_max]
    """
    x_min, y_min, x_max, y_max = xyxy
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min

    x_min = max(0, x_min - margin * bbox_width)
    y_min = max(0, y_min - margin * bbox_height)
    x_max = min(img_width, x_max + margin * bbox_width)
    y_max = min(img_height, y_max + margin * bbox_height)

    expanded_bbox = [int(x_min), int(y_min), int(x_max), int(y_max)]

    return expanded_bbox

# La ruta de la carpeta de imágenes que se desea procesar se obtiene como parámetro por la linea de comandos.
parser = argparse.ArgumentParser()
parser.add_argument('folder_path', type=str, help='Proporciona la ruta a la carpeta que contiene las imágenes que desea procesar.', nargs='?')
args = parser.parse_args()
if args.folder_path is None: 
    parser.print_help(sys.stderr)
    sys.exit(1)
    
input_dir = args.folder_path

# Obtener el nombre del video de la carpeta de entrada para crear la carpeta de salida
folder_name = os.path.basename(input_dir)
if folder_name.startswith("frames-"):
    video_name = folder_name[len("frames-"):]
else:
    print("Error: El nombre de la carpeta no sigue el formato esperado 'frames-nombrevideo'.")
    sys.exit(1)

# Crear la carpeta donde se almacena la salida. Si ya existe se borra su contenido. 
output_dir = f"faces-{video_name}"
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

# Obtener una lista de nombres de archivos en la carpeta
image_names = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png'))]

# Cargar el modelo YOLO (detector de objetos)
model = YOLO("yolov8n-face.pt")

# Dimensiones deseadas para las imágenes redimensionadas
target_width = 1080
target_height = 1350

# Contador para nombrar las imágenes recortadas
face_counter = 1

# Procesar cada archivo de imagen en la carpeta
for image_name in image_names:
    image_path = os.path.join(input_dir, image_name)
    image = cv2.imread(image_path)
    img_h, img_w, _ = image.shape  # Obtener las dimensiones de la imagen
    results = model(image)[0] # Se obtienen los resultados de aplicar YOLO a la imagen

    # Ahora se cargan en Supervision las predicciones del modelo
    detections = sv.Detections.from_ultralytics(results)
    detections_pos = detections.xyxy

    # Verificar si hay detecciones y procesar cada una
    if detections_pos.any():
        with sv.ImageSink(target_dir_path=output_dir) as sink:
            
            for xyxy in detections_pos:
            #for xyxy in detections.xyxy:
                expanded_xyxy = expand_bbox(xyxy, img_w, img_h, margin=0.5)
                cropped_image = sv.crop_image(image=image, xyxy=expanded_xyxy)

                # Redimensionar la imagen recortada
                resized_cropped_image = cv2.resize(cropped_image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

                # Nombre de los archivos generados
                cropped_image_filename = f"face{face_counter}.jpg"
                sink.save_image(image=resized_cropped_image, image_name=cropped_image_filename)
                face_counter += 1
    
print(f"Procesamiento de detección de caras completado. Las imágenes recortadas se han guardado en la carpeta '{output_dir}'.")