from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import logging
import imutils
import time
import cv2
import os


def crear_nuevo_archivo_video(path_tmp, size_tmp):
   logger.info("Creando nuevo archivo de video")
   path_actual = genera_nuevo_directorio(genera_nombre_directorio_fecha_actual(path_tmp))
   return crea_nuevo_archivo_video(path_actual, size_tmp)

def genera_nuevo_directorio(name_path):
   if not os.path.exists(name_path):
       try:
           Path(name_path).mkdir(parents=True, exist_ok=True)
       except OSError:
           logger.info("Creation of the directory %s failed" % name_path)
   return name_path

def crea_nuevo_archivo_video(path_tmp, size):
   horaActual = str(datetime.now().strftime("%H-%M-%S").upper())
   nombre_video = path_tmp + "video--" + horaActual + ".avi"
   logger.info(nombre_video)
   out = cv2.VideoWriter(nombre_video, cv2.VideoWriter_fourcc(*'XVID'), 30, size)
   return out

def genera_nombre_directorio_fecha_actual(path_tmp):
   name_tmp = path_tmp + str(datetime.now().strftime("%Y/%b/%d").upper() + "/")
   #logger.info(name_tmp)
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

def get_gray_frame(frame_tmp):
    gray = cv2.cvtColor(frame_tmp, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    return gray

def detector_movimiento(firstframe_tmp, frame_tmp, MINIMUM_AREA_TMP):
    movimiento_detectado = False
    gray = get_gray_frame(frame_tmp)

    delta = cv2.absdiff(firstframe_tmp, gray)
    thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]

    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    for c in cnts:
        if cv2.contourArea(c) < MINIMUM_AREA_TMP:
            continue
        #(x, y, w, h) = cv2.boundingRect(c)
        #cv2.rectangle(frame_tmp, (x, y), (x + w, y + h), (0, 255, 0), 2)
        movimiento_detectado = True
    #cv2.imshow("VibrAlert v0.1", frame_tmp)
    return movimiento_detectado

logging.basicConfig(filename='camaraLogFile.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
# Let us Create an object
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())
# Now we are going to Set the threshold of logger to DEBUG
logger.setLevel(logging.INFO)

def main():
    person = 0
    MINIMUM_AREA = 700
    firstFrame = None
    sin_movimiento_conteo=0
    limite_conteo_sin_movimiento=2
    directorio_raiz = "/home/keith/Documents/TMP/"
    directorio_grabaciones = directorio_raiz + "grabaciones/"
    opencv_files = directorio_raiz + "openCV_files_tiny/"

    #entrada_video = "/home/keith/Documents/TMP/" + "video--21-14-38.avi"
    entrada_video = "/home/keith/Documents/TMP/" + "calle.MP4"

    #entrada_video = "rtsp://Admin:Trey7h@192.168.1.82/live"
    labels = open(opencv_files + 'coco.names').read().strip().split('\n')
    net = cv2.dnn.readNetFromDarknet(opencv_files + 'yolov4-tiny.cfg', opencv_files + 'yolov4-tiny.weights')
    layer_names = net.getLayerNames()
    layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    segundos_delay_deteccion_objetos=5
    duracion_minima_video=30
    tiempo_deteccion_objetos = datetime.now() - timedelta(seconds=segundos_delay_deteccion_objetos)
    cap = cv2.VideoCapture(entrada_video)
    actualizar_imagen_fondo=True

    is_objetos_detectados = False
    isGrabacionActiva=False
    archivo_video_activo=False
    tiempo_margen_disponible=True


    if (cap.isOpened()== False):
        logger.info("Error opening video stream or file")
    while(cap.isOpened()):
        tiempo_actual = datetime.now()
        ret, frame = cap.read()
        if ret == True:
            #cv2.imshow('Frame',frame)
            if tiempo_actual > tiempo_deteccion_objetos:
                logger.info("******DETECTANDO OBJETOS********" + time.strftime('%X %x %Z'))
                confidences, classIDs = detectar_objetos(frame, net, layer_names, labels)
                tiempo_deteccion_objetos = datetime.now() + timedelta(seconds=segundos_delay_deteccion_objetos)

                if classIDs:
                    for clase in classIDs:
                        logger.info(labels[clase])
                    if person not in classIDs:
                        logger.info("NO HAY PERSONAS DETECTADAS")
                        frame_detector_movimiento = cv2.resize(frame, (640, 480))
                        if actualizar_imagen_fondo:
                            firstFrame = get_gray_frame(frame_detector_movimiento)
                            actualizar_imagen_fondo=False
                        else:
                            logger.info("**iniciando deteccion movimiento**")
                            move_detected = detector_movimiento(firstFrame, frame_detector_movimiento, MINIMUM_AREA)
                            logger.info(move_detected)
                            actualizar_imagen_fondo = True
                            if not move_detected:
                                sin_movimiento_conteo += 1
                                logger.info("NO HAY MOVIMIENTO, conteo:" + str(sin_movimiento_conteo))
                            else:
                                sin_movimiento_conteo = 0
                                logger.info("MOVIMIENTO DETECTADO")
                    else:
                        sin_movimiento_conteo = 0

                    if sin_movimiento_conteo >= limite_conteo_sin_movimiento:
                        logger.info("NO HAY MOVIMIENTOS")
                        is_objetos_detectados = False
                        isGrabacionActiva = False
                    else:
                        logger.info("*****objeto detectado*****" + time.strftime('%X %x %Z'))
                        is_objetos_detectados = True
                        tiempo_margen_disponible = True
                        tiempo_deteccion_objetos = datetime.now() + timedelta(seconds=duracion_minima_video)
                else:
                    if tiempo_margen_disponible:
                        logger.info("iniciando tiempo margen")
                        tiempo_deteccion_objetos = datetime.now() + timedelta(seconds=10)
                        tiempo_margen_disponible=False
                    else:
                        is_objetos_detectados=False
                        isGrabacionActiva=False

            if is_objetos_detectados and isGrabacionActiva is False:
                logger.info("INCIANDO NUEVA GRABACION" + time.strftime('%X %x %Z'))
                salida_video_actual = crear_nuevo_archivo_video(directorio_grabaciones, tamanio_video(cap))
                isGrabacionActiva = True
                archivo_video_activo = True

            if isGrabacionActiva:
                salida_video_actual.write(frame)
            else:
                if archivo_video_activo:
                    logger.info("CERRANDO ARCHIVO DE VIDEO" + time.strftime('%X %x %Z'))
                    salida_video_actual.release()
                    archivo_video_activo = False

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

if __name__ == "__main__":
   main()
