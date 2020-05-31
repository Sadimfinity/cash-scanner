#--------------------------------------------------------------------------
#------- PLANTILLA DE CÓDIGO ----------------------------------------------
#------- Proyecto final de PDI-------------------------------------------
#------- Por: Santiago Gaviria  santiago.gaviriaz@udea.edu.co--------------
#-------      Estudiante Facultad de Ingenieria ---------------------------
#-------      CC 1035440028, Tel 5963717 ----------------------------------
#------- Curso Procesamiento Digital de Imágenes --------------------------
#------- Junio de 2020 ----------------------------------------------------
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
#------- Importamos los paquetes necesarios ---------------------------------
#--------------------------------------------------------------------------

# -*- coding: utf-8 -*-
from multiprocessing import Process
import cv2
import numpy as np
import requests
import math
import time

#--------------------------------------------------------------------------
#------- Función que captura cada una de las imágenes de ------------------
#------- entrenamiento y a través del uso del método SIFT -----------------
#------- detecta los puntos clave de los billetes de  ---------------------
#------- cada deniminación por ambos lados. -------------------------------
#--------------------------------------------------------------------------

def initialize_images(detector, denominations):
    trainImgs = []
    trainKPs = []
    trainDescs = []
    for i in denominations:
        for j in range(2):
            img_url = './resources/train-data/' + str(i) + '_' + str(j+1) + '.jpg'
            trainImg = cv2.imread(img_url)
#------- En la siguiente línea se hace uso del método detectAndCompute() --
#------- para capturar los puntos claves y descriptores de cada imagen ----            
            trainKP,trainDesc=detector.detectAndCompute(trainImg,None)
            trainImgs.append(trainImg)
            trainKPs.append(trainKP)
            trainDescs.append(trainDesc)
    return trainImgs, trainKPs, trainDescs

#--------------------------------------------------------------------------
#------- Esta función permite paralelizar la ejecución de dos -------------
#------- métodos recibiendo como parámetro, el nombre de cada -------------
#------- una de las funciones a ejecutar. ---------------------------------
#--------------------------------------------------------------------------

def run_in_parallel(*fns):
  proc = []
  for fn in fns:
    p = Process(target=fn)
    p.start()
    proc.append(p)
  for p in proc:
    p.join()

#--------------------------------------------------------------------------
#------- Esta función fue obtenida del repositorio de Github: -------------
#------- https://github.com/asharma327/Read_Gif_OpenCV_Python -------------
#--------------------------------------------------------------------------

def convert_gif_to_frames(gif):

    # Initialize the frame number and create empty frame list
    frame_num = 0
    frame_list = []

    # Loop until there are frames left
    while True:
        try:
            # Try to read a frame. Okay is a BOOL if there are frames or not
            okay, frame = gif.read()
            # Append to empty frame list
            frame_list.append(frame)
            # Break if there are no other frames to read
            if not okay:
                break
            # Increment value of the frame number by 1
            frame_num += 1
        except KeyboardInterrupt:  # press ^C to quit
            break
    return frame_list

#--------------------------------------------------------------------------
#------- Esta función reproduce el gif de carga por un tiempo -------------
#------- estimado de 90 segundos. -----------------------------------------
#--------------------------------------------------------------------------

def display_loading():
    gif = cv2.VideoCapture('./resources/loading.gif')
    frames = convert_gif_to_frames(gif)
    size = len(frames)

    ## Display the gif
    i = 0
    t_end = time.time() + 90
    while time.time() < t_end:
        frames[i] = cv2.resize(frames[i], (800,480))
#------- En la siguiente línea se muestra cada uno de los frames ----------
#------- en los cuáles está compuesto el gif usando cv2.imshow() ----------
        cv2.imshow('Cargando...', frames[i])
        if cv2.waitKey(100)&0xFF == 27:
            break
        i = i+1
        if i == size-1:
            i = 0
    cv2.destroyAllWindows()

#--------------------------------------------------------------------------
#------- Esta función no fue usada en la versión final del proyecto -------
#------- pero servía para, usando los puntos claves obtenidos, ------------
#------- dibujar el contorno estimado asociado al emparejamiento ----------
#--------------------------------------------------------------------------

def draw_contours(goodMatches, maxPointsIndexes, trainKPs, trainImgs):
    tp=[]
    qp=[]
    for m in goodMatches[maxPointsIndexes[i]]:
        trainKP = trainKPs[maxPointsIndexes[i]]
        tp.append(trainKP[m.trainIdx].pt)
        qp.append(queryKP[m.queryIdx].pt)
    tp,qp=np.float32((tp,qp))
    H,status=cv2.findHomography(tp,qp,cv2.RANSAC,3.0)
    trainImg = trainImgs[maxPointsIndexes[i]]
    h,w,_=trainImg.shape
    trainBorder=np.float32([[[0,0],[0,h-1],[w-1,h-1],[w-1,0]]])
    queryBorder=cv2.perspectiveTransform(trainBorder,H)
    # convert the grayscale image to binary image
    ret,thresh = cv2.threshold(QueryImg,127,255,0)
    cv2.polylines(QueryImgBGR,[np.int32(queryBorder)],True,(255,0,0),5)

#--------------------------------------------------------------------------
#------- Esta es la función principal del proyecto, detecta el ------------
#------- dinero que haya en pantalla y lo imprimirá en consola ------------
#--------------------------------------------------------------------------

def scan_cash():
#------- IP desde la cuál transmitirá la cámara a través del uso de  ------
#------- la aplicación para Android: IP Webcam PRO ------------------------
    url = 'http://192.168.0.11:8080/shot.jpg'
#------- Definimos el valor suficiente de puntos clave para que una  ------
#------- imagen se detecta como billete. ----------------------------------
    MIN_MATCH_COUNT=80
#------- Inicializamos el algoritmo SIFT ----------------------------------
    detector=cv2.xfeatures2d.SIFT_create()

    FLANN_INDEX_KDITREE=0
    flannParam=dict(algorithm=FLANN_INDEX_KDITREE,tree=5)
#------- Inicializamos el flann (Fast Library for Approximate Nearest) ----
#------- Neighbors que nos permitirá entrenar los clasificadores de -------
#------- manera rápida. ---------------------------------------------------
    flann=cv2.FlannBasedMatcher(flannParam,{})
    denominations = [1,2,5,10,20,50]
#------- Obtenemos las imágenes de entrenamiento, descriptores y puntos ---
#------- clave ------------------------------------------------------------
    trainImgs, trainKPs, trainDescs = initialize_images(detector, denominations)
    total = 0
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
#------- Capturamos la imagen a través de un GET a la IP definida y la  ---
#------- formateamos para poder manipularla -------------------------------
    QueryImgBGR = cv2.imdecode(img_arr, -1)
    QueryImg=cv2.cvtColor(QueryImgBGR,cv2.COLOR_RGB2GRAY)
    queryKP,queryDesc=detector.detectAndCompute(QueryImg,None)
    matches = []
    for trainDesc in trainDescs:
#------- En la siguiente línea se clasifica usando el método KNN, los------
#------- descriptores de cada una de las imágenes de entrenamiento --------
#------- y el descriptor de la imagen capturada. --------------------------
        match = flann.knnMatch(queryDesc, trainDesc, k=2)
        matches.append(match)
    goodMatches = []
    for i in range(len(matches)):
        goodMatch = []
        goodMatches.append(goodMatch)
    i = 0
    for match in matches:
        for m, n in match:
#------- Obtiene los mejores emparejamientos usando la estrategia ---------
#------- descrita por Lowe (Lowe's ratio test) para esta índole -----------
            if(m.distance<0.75*n.distance):
                goodMatches[i].append(m)
        i = i + 1
    i = 0
    arri = []
#------- Guarda en una arreglo el tamaño de cada uno de los mejores--------
#------- emparejamientos y obtiene los dos valores máximos ----------------
    for goodMatch in goodMatches:
        leng = len(goodMatch)
        arri.append(leng)
    maxPointsIndexes = np.argpartition(arri, -2)[-2:]
    i = 0

#------- Recorremos los mejores emparejamientos y verificamos que ---------
#------- estos cumplan con el MIN_MATCH_COUNT, si esto se cumple, ---------
#------- obtenemos la denominación del billete en cuestión y lo -----------
#------- sumamos a la cuenta total. ---------------------------------------
    for i in range(len(maxPointsIndexes)):
        if(len(goodMatches[maxPointsIndexes[i]])>MIN_MATCH_COUNT):
            #draw_contours(goodMatches, maxPointsIndexes, trainKPs, trainImgs)
            total = total + denominations[int(math.floor(maxPointsIndexes[i]/2))]
        i = i + 1
    if total > 0:
        print('Hay ' + str(total) + ' mil pesos')
    else:
        print('No se encontraron billetes de ninguna denominación')
    cv2.destroyAllWindows()

#--------------------------------------------------------------------------
#------- Aquí se ejecuta el detector de billetes paralelamente junto ------
#------- con la gif de carga. ---------------------------------------------
#--------------------------------------------------------------------------

def main():
    url = 'http://192.168.0.11:8080/shot.jpg'
    while True:
        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        QueryImgBGR = cv2.imdecode(img_arr, -1)
        cv2.imshow('Imagen',QueryImgBGR)
        if cv2.waitKey(10)==ord('q'):
            cv2.destroyAllWindows()
            run_in_parallel(scan_cash, display_loading)
        if cv2.waitKey(13)==13:
            break
    cv2.destroyAllWindows()

main()
