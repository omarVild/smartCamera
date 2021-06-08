from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import time
import cv2
import os

def crear_nuevo_archivo_video(path_tmp, size_tmp):
   print("Creando nuevo archivo de video")
   path_actual = genera_nuevo_directorio(genera_nombre_directorio_fecha_actual(path_tmp))
   return crea_nuevo_archivo_video(path_actual, size_tmp)

def genera_nuevo_directorio(name_path):
   if not os.path.exists(name_path):
       try:
           Path(name_path).mkdir(parents=True, exist_ok=True)
       except OSError:
           print("Creation of the directory %s failed" % name_path)
   return name_path

def crea_nuevo_archivo_video(path_tmp, size):
   horaActual = str(datetime.now().strftime("%H-%M-%S").upper())
   nombre_video = path_tmp + "video--" + horaActual + ".avi"
   print(nombre_video)
   out = cv2.VideoWriter(nombre_video, cv2.VideoWriter_fourcc(*'MJPG'), 20, size)
   return out

def genera_nombre_directorio_fecha_actual(path_tmp):
   name_tmp = path_tmp + str(datetime.now().strftime("%Y/%b/%d").upper() + "/")
   #print(name_tmp)
   return name_tmp

def tamanio_video(cap):
   frame_width = int(cap.get(3))
   frame_height = int(cap.get(4))
   size = (frame_width, frame_height)
   return size

def detectar_objetos(image, net,layer_names, labels):
   confidence = 0.5
   height, width = image.shape[:2]
   # Create a blob and pass it through the model
   blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
   net.setInput(blob)
   outputs = net.forward(layer_names)
   confidences = []
   classIDs = []
   filtro_objetos = ['person', 'bicycle', 'car', 'motorbike', 'bus', 'truck']


   for output in outputs:
       for detection in output:
           scores = detection[5:]
           classID = np.argmax(scores)
           conf = scores[classID]
           # Consider only the predictions that are above the confidence threshold
           if conf > confidence and labels[classID] in filtro_objetos:
               confidences.append(float(conf))
               classIDs.append(classID)
   return confidences, classIDs

def imprime_fecha_y_hora(frame_tmp):
   font = cv2.FONT_ITALIC
   dt = str(datetime.now().strftime("%b %d %Y %H:%M:%S").upper())
   frame_tmp = cv2.putText(frame_tmp, dt, (10, 20), font, .6, (0, 0, 255), 2, cv2.LINE_4)
   return frame_tmp

def main():
    directorio_raiz = "/home/keith/Documents/openCV/"
    directorio_grabaciones = directorio_raiz + "grabaciones/"
    opencv_files = directorio_raiz + "openCV_files_tiny/"
    #video = directorio_raiz + "GH020268.MP4"
    entrada_video = "rtsp://Admin:xxxxx@192.168.1.82/live"
    labels = open(opencv_files + 'coco.names').read().strip().split('\n')
    net = cv2.dnn.readNetFromDarknet(opencv_files + 'yolov4-tiny.cfg', opencv_files + 'yolov4-tiny.weights')
    layer_names = net.getLayerNames()
    layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    delay_tiempo_deteccion_objetos = datetime.now() - timedelta(seconds=5)
    cap = cv2.VideoCapture(entrada_video)
    size = tamanio_video(cap)

    is_objetos_detectados = False
    isGrabacionActiva=False
    archivo_video_activo=False
    tiempo_margen_disponible=True


    if (cap.isOpened()== False):
        print("Error opening video stream or file")
    while(cap.isOpened()):
        tiempo_actual = datetime.now()
        ret, frame = cap.read()
        if ret == True:
            #cv2.imshow('Frame',frame)
            if tiempo_actual > delay_tiempo_deteccion_objetos:
                print("******DETECTANDO OBJETOS********" + time.strftime('%X %x %Z'))
                confidences, classIDs = detectar_objetos(frame, net, layer_names, labels)
                delay_tiempo_deteccion_objetos = datetime.now() + timedelta(seconds=5)
                if classIDs:
                    print("*****objeto detectado*****" + time.strftime('%X %x %Z'))
                    #for clase in classIDs:
                        #print(labels[clase])
                    is_objetos_detectados = True
                    tiempo_margen_disponible = True
                    delay_tiempo_deteccion_objetos = datetime.now() + timedelta(seconds=30)
                else:
                    if tiempo_margen_disponible:
                        print("iniciando tiempo margen")
                        delay_tiempo_deteccion_objetos = datetime.now() + timedelta(seconds=5)
                        tiempo_margen_disponible=False
                    else:
                        is_objetos_detectados=False
                        isGrabacionActiva=False

            if is_objetos_detectados and isGrabacionActiva is False:
                print("INCIANDO NUEVA GRABACION" + time.strftime('%X %x %Z'))
                salida_video_actual = crear_nuevo_archivo_video(directorio_grabaciones, tamanio_video(cap))
                isGrabacionActiva = True
                archivo_video_activo = True

            if isGrabacionActiva:
                salida_video_actual.write(frame)
            else:
                if archivo_video_activo:
                    print("CERRANDO ARCHIVO DE VIDEO" + time.strftime('%X %x %Z'))
                    salida_video_actual.release()
                    archivo_video_activo = False

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

if __name__ == "__main__":
   main()
