import numpy as np
import cv2
import mediapipe as mp
import os
import argparse
import sys
import random
import shutil
import time

EAR_THRESHOLD = 0.27

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
    
    output_folder = f"mp-combined-{base_name}"
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

# Crear face_mesh
def create_face_mesh():
   
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5,
        max_num_faces=1,
        refine_landmarks=True,
        static_image_mode=True
    )
    return face_mesh

def get_pupil_position (eye_center, pupil, img_w, img_h):

    if pupil.x < (eye_center[0] / img_w) - 0.01:
        return "Left"
    elif pupil.x > (eye_center[0] / img_w) + 0.01:
        return "Right"
    elif pupil.y < (eye_center[1] / img_h) - 0.05:
        return "Up"
        
    elif pupil.y > (eye_center[1] / img_h) + 0.05:
        return "Down"
    else:
        return "Center"
    

def gaze_direction (results, img_w, img_h): 

    gaze_direction = [] 
    if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:

                # Calcular el centro de cada ojo (derecho e izquierdo)
                right_eye_landmarks = [[lm.x * img_w, lm.y * img_h] for idx, lm in enumerate(face_landmarks.landmark) if idx in [374, 386, 362, 263]]
                left_eye_landmarks = [[lm.x * img_w, lm.y * img_h] for idx, lm in enumerate(face_landmarks.landmark) if idx in [145, 159, 33, 133]]

                if right_eye_landmarks and left_eye_landmarks:

                    right_eye_center = np.mean(right_eye_landmarks, axis=0)
                    left_eye_center = np.mean(left_eye_landmarks, axis=0)

                    # Obtener la posición de las pupilas
                    right_pupil = face_landmarks.landmark[473]
                    left_pupil = face_landmarks.landmark[468]

                    if right_pupil and left_pupil:

                        right_pupil_position = get_pupil_position(right_eye_center, right_pupil, img_w, img_h)
                        gaze_direction.append(right_pupil_position)
                        left_pupil_position = get_pupil_position(left_eye_center, left_pupil, img_w, img_h)
                        gaze_direction.append(left_pupil_position)

    return gaze_direction
      

def head_pose_estimation (results, img_w, img_h): 

    face_2d = []
    face_3d = []

    orientation = "No face detected"

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in [33, 263, 1, 61, 291, 199, 6, 168, 105, 334, 152, 10]:

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

            if y < -5:
                orientation = "Left"
            elif y > 5:
                orientation = "Right"
            elif x < -5:
                orientation = "Down"
            elif x > 5:
                orientation = "Up"
            else:
                orientation = "Center"

    return orientation


 # Función para calcular la distancia euclidiana entre dos puntos
def euclidean_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

# Función para calcular el EAR
def calculate_ear(eye_landmarks, img_w, img_h):

    p1 = np.array([eye_landmarks[0].x * img_w, eye_landmarks[0].y * img_h])
    p2 = np.array([eye_landmarks[1].x * img_w, eye_landmarks[1].y * img_h])
    p3 = np.array([eye_landmarks[2].x * img_w, eye_landmarks[2].y * img_h])
    p4 = np.array([eye_landmarks[3].x * img_w, eye_landmarks[3].y * img_h])
    p5 = np.array([eye_landmarks[4].x * img_w, eye_landmarks[4].y * img_h])
    p6 = np.array([eye_landmarks[5].x * img_w, eye_landmarks[5].y * img_h])

    # Calcular las distancias verticales
    A = euclidean_distance(p2, p6)
    B = euclidean_distance(p3, p5)

    # Calcular la distancia horizontal
    C = euclidean_distance(p1, p4)

    # Calcular el EAR
    ear = (A + B) / (2.0 * C)
    return ear

def eyes_closed(results, img_w, img_h):

    # Landmarks del ojo derecho (p1, p2, p3, p4, p5, p6)
    right_eye_indices = [362, 385, 387, 263, 373, 380]

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            right_eye_landmark = [face_landmarks.landmark[idx] for idx in right_eye_indices]

            # Calcular el EAR para el ojo derecho
            ear_right = calculate_ear(right_eye_landmark, img_w, img_h)
        
    if ear_right < EAR_THRESHOLD:
        return True
    
    return False


# Estudiar si la persona mira o no a cámara
def looking_at_camera (face_mesh, input_folder): 

    forward_images = []

    for image_name in os.listdir(input_folder):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            
            image_path = os.path.join(input_folder, image_name)
            image = cv2.imread(image_path)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = face_mesh.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            img_h, img_w, img_c = image.shape
      
            head_pose = head_pose_estimation(results, img_w, img_h)
            gaze = gaze_direction(results, img_w, img_h)

            if gaze:

                left_eye = gaze[0]
                right_eye = gaze[1]

                if head_pose == "Center":
                    eyes_are_closed = eyes_closed(results, img_w, img_h) 
                    
                    if (not eyes_are_closed) and (left_eye == "Center" or right_eye == "Center"): 
                        forward_images.append((image, image_name))
                        print(f" ojo abiertos {image_name}")
            
    return forward_images


def main(): 

    start = time.time()

    input_folder = get_images()

    face_mesh = create_face_mesh()

    forward_images = looking_at_camera (face_mesh, input_folder)

    txt_file = get_response(forward_images, input_folder)

    end = time.time()  # Fin del tiempo de procesamiento
    print(f"Tiempo empleado: {end - start:.2f} segundos.\n")
    print(f"Procesamiento de rostros con FaceMesh y MediaPipe completado. Salida en {txt_file}.")

if __name__ == "__main__":
    main()   
