import json
import sys
import os
from datetime import timedelta


TIEMPO_SEG = 4

# Funcion que permite quitar las tildes del texto a procesar
def quitar_tildes (texto):
    
    mapa_tildes = {'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u'}

    return ''.join(mapa_tildes.get(caracter, caracter) for caracter in texto)

# Funcion para mostrar el tiempo con el formato deseado (HH:MM:SS)
def formatear_tiempo (seconds):

    td = timedelta(seconds=seconds)
    horas = td.seconds // 3600
    minutos = (td.seconds // 60) % 60
    segundos = td.seconds % 60

    return f"{horas:02}:{minutos:02}:{segundos:02}"


# Función para leer un archivo json y extraer el texto
def leer_json (archivo_json, archivo_txt, palabras_clave):

    notas_inicio = False

    with open(archivo_json, 'r') as archivo:
        datos_json = json.load(archivo)
        
        tiempo_anterior_fin = None
        
        # Creo el archivo txt correspondiente para archivo json
        with open(archivo_txt, "w") as notas: 

            # Si existe la clave segments y si es de tipo lista
            if "segments" in datos_json and isinstance(datos_json["segments"], list):
                
                # Iterar sobre cada elemento en la lista de segments
                for segment in datos_json["segments"]:
                    
                    # Si hay archivos de texto se escriben el archivo txt
                    if "text" in segment and "start" in segment and "end" in segment and "speaker" in segment: 

                        texto_original = segment["text"]
                        texto_procesado = quitar_tildes(texto_original.lower())

                        tiempo_inicio = segment["start"]
                        tiempo_formateado = formatear_tiempo(tiempo_inicio)
                        tiempo_fin = segment["end"]

                        hablante = segment["speaker"]    

                        if any (palabra in texto_procesado for palabra in palabras_clave):
                            
                            if "accion" in texto_procesado:

                                notas.write("\nCOMIENZO DEL DIÁLOGO\n\n")
                                notas.write(f"{tiempo_formateado} {hablante} {texto_original}\n")

                            else: #Corten

                                notas.write(f"{tiempo_formateado} {hablante} {texto_original}\n")
                                notas.write("\nFIN DEL DIÁLOGO\n\n")
                                notas.write("NOTAS\n") # Notas después de Corten
                        
                        # No hay texto en el segmento procesado        
                        else: 

                            # Si el tiempo transcurrido entre el inicio de una frase y el final de la anterior 
                            # es mayor que 4 segundos se entiende que hay un SILENCIO        
                            if tiempo_anterior_fin is not None:
                                tiempo_transcurrido = tiempo_inicio - tiempo_anterior_fin
                                if tiempo_transcurrido > TIEMPO_SEG: 
                                    notas.write ("\nSILENCIO\n\n")
                            
                            tiempo_anterior_fin = tiempo_fin

                            if not notas_inicio:
                                    notas.write("NOTAS\n\n") # notas antes de Accion
                                    notas_inicio = True
                            
                            notas.write(f"{tiempo_formateado} - {hablante} {texto_original}\n")
                
            else:
                print("El archivo JSON {archivo_json} no tiene la estructura esperada.")


def main(): 
    
    # Se toma el nombre de los archivos json por linea de comandos, al menos tiene que haber 1. 
    if len(sys.argv) < 2: 
        print("Uso: python tu_script.py <ruta_archivo_json1> [<ruta_archivo_json2>...]")
        sys.exit(1)

    archivos_json = sys.argv[1:]

    for archivo_json in archivos_json: 

        # Se asigna nombre al archivo txt que va a almacenar el audio transcrito
        nombre_archivo_sin_extension = os.path.splitext(os.path.basename(archivo_json))[0]
        archivo_txt = nombre_archivo_sin_extension + "-transcript.txt"
        
        palabras_clave = ["accion", "corten", "corte"] # Palabras que interesa buscar en el archivo json
        leer_json(archivo_json, archivo_txt, palabras_clave)


if __name__ == "__main__":
    main()

