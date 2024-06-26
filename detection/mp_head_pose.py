import os
import sys
import time
import argparse
import random
import shutil
import cv2
import numpy as np
import mediapipe as mp


# Leer la carpeta de imágenes de entrada
def get_images(): 

    parser = argparse.ArgumentParser()
    parser.add_argument('folder_path', type=str, help='Proporciona la ruta a la carpeta que contiene las imágenes que desea procesar.', nargs='?')
    args = parser.parse_args()
    if args.folder_path is None:
        parser.print_help(sys.stderr)
        sys.exit(1)

    input_folder = args.folder_path
    print(input_folder.count)
    
    return input_folder

def get_response (forward_images, input_folder):

    # Definir la carpeta de salida
    base_name = input_folder.split("faces-")[-1]
    output_folder = f"mp-rostros-{base_name}"
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    if len(forward_images) > 0:
        message = "Sí se ha detectado la existencia de persona o personas mirando a cámara en el vídeo.\n"

        num_images = min(len(forward_images), 10)
        selected_images = random.sample(forward_images, num_images)

        for img, original_name in selected_images:
            original_filename = os.path.basename(original_name)
            cv2.imwrite(os.path.join(output_folder, original_filename), img)

        message += f'En la carpeta "{output_folder}" se ha almacenado una muestra de rostros que se considera que miran a cámara.\n'
    else: 
        message = "No se ha detectado a ninguna persona que mire a cámara."
    
    txt_file = f"mp_answers_{base_name}.txt"
    with open(txt_file, "w") as file:
        file.write(message)

    return txt_file

# Inicializar mediapipe face mesh
def create_face_mesh():

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_faces=1, refine_landmarks=True,
        static_image_mode=True)

    return face_mesh


def head_pose_estimation(face_mesh, input_folder):

    # Índices de los landmarks de los ojos izquierdo y derecho
    left_eye_indices = [33, 133, 160, 158, 153, 144, 163, 7, 362]
    right_eye_indices = [263, 362, 387, 385, 380, 373, 374, 382, 33]
    
    forward_images = []
    # Procesar cada imagen en la carpeta de entrada
    for image_name in os.listdir(input_folder):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, image_name)
            image = cv2.imread(image_path)

            start = time.time()

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = face_mesh.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            img_h, img_w, img_c = image.shape
            face_2d = []
            face_3d = []

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    for idx, lm in enumerate(face_landmarks.landmark):
                        if idx in [33, 263, 1, 61, 291, 199, 6, 168, 105, 334, 152, 10, 468, 473] and left_eye_indices and right_eye_indices:

                            #x, y = int(lm.x * img_w), int(lm.y * img_h)
                            x, y = int(lm.x * img_w), int(lm.y * img_h)
                            face_2d.append([x, y])
                            face_3d.append(([x, y, lm.z]))

                    face_2d = np.array(face_2d, dtype=np.float64)
                    face_3d = np.array(face_3d, dtype=np.float64)

                    focal_length = 1 * img_w

                    cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                           [0, focal_length, img_w / 2],
                                           [0, 0, 1]])
                    distortion_matrix = np.zeros((4, 1), dtype=np.float64)

                    success, rotation_vec, translation_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, distortion_matrix)

                    rmat, jac = cv2.Rodrigues(rotation_vec)

                    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                    x = angles[0] * 360
                    y = angles[1] * 360
                    z = angles[2] * 360

                    if y < -4:
                        text = "Looking Left"
                    elif y > 4:
                        text = "Looking Right"
                    elif x < -4:
                        text = "Looking Down"
                    elif x > 4:
                        text = "Looking Up"
                    else:
                        text = "Forward"
            
            if text == "Forward":
                forward_images.append((image, image_name)) 
            
    return forward_images

def main(): 

    start = time.time()

    input_folder = get_images()
    face_mesh = create_face_mesh()
    forward_images = head_pose_estimation(face_mesh, input_folder)
    txt_file = get_response (forward_images, input_folder)

    end = time.time()  # Fin del tiempo de procesamiento
    print(f"Tiempo empleado: {end - start:.2f} segundos.\n")
    print(f"Procesamiento de rostros con FaceMesh y MediaPipe completado. Salida en {txt_file}.")

if __name__ == "__main__":
    main()   

