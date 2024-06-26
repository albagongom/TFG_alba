import base64
from io import BytesIO
from PIL import Image
import os 
import glob
import argparse
import sys 

def convert_to_base64(pil_image):
    """
    Convert PIL images to Base64 encoded strings

    :param pil_image: PIL image
    :return: Re-sized Base64 string
    """

    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

#Para leer las imagenes desde una carpeta por linea de comandos
parser = argparse.ArgumentParser()
parser.add_argument('folder_path', type=str, nargs='?', help='Proporciona la ruta a la carpeta que contiene las imágenes que desea procesar.')
args = parser.parse_args()
if args.folder_path is None: 
    parser.print_help()
    parser.exit()

folder_path = args.folder_path

# Obtener el nombre del video de la carpeta de entrada
folder_name = os.path.basename(folder_path)

if folder_name.startswith("faces-"):
    video_name = folder_name[len("faces-"):]
else:
    print("Error: El nombre de la carpeta no sigue el formato esperado 'faces-nombrevideo'.")
    sys.exit(1)

# Obtener las imagenes a procesar
image_files = glob.glob(os.path.join(folder_path, "*.jpg"))

# Crear el archivo de texto (en el directorio actual) para imprimir en él las respuestas
current_dir = os.path.dirname(os.path.abspath(__file__))
txt_filename = f"answers-{video_name}.txt"
txt_filepath = os.path.join(current_dir, txt_filename)

from langchain.schema import HumanMessage
from langchain_community.chat_models import ChatOllama

# Cargar el modelo llava
llm = ChatOllama(base_url="http://localhost:11435", model="llava", temperature=0)

def prompt_func(data):
    text = data["text"]
    image = data["image"]

    image_part = {
        "type": "image_url",
        "image_url": f"data:image/jpeg;base64,{image}",
    }

    content_parts = []

    text_part = {"type": "text", "text": text}

    content_parts.append(image_part)
    content_parts.append(text_part)

    return [HumanMessage(content=content_parts)]


from langchain_core.output_parsers import StrOutputParser

# Preguntar a imagen y obtener respuesta
with open(txt_filepath, 'w') as txt_file:
    for file_path in image_files:

        pil_image = Image.open(file_path)
        image_b64 = convert_to_base64(pil_image)

        chain = prompt_func | llm | StrOutputParser()

        query_chain = chain.invoke(
            {"text": "Analyze the following image and determine if the person appears to be looking directly at the camera. Respond with 'Yes' if the person is looking directly at the camera, and 'No' otherwise.", "image": image_b64}
        )

        # Escribir la respuesta en el archivo de texto
        txt_file.write(f"{os.path.basename(file_path)}\n")
        txt_file.write(f"{query_chain}\n\n")

print(f"Procesamiento de imágenes con LLaVA completado. Las respuestas se han guardado en '{txt_filepath}'.")

