# -*- coding: utf-8 -*-
from multiprocessing import Process
import cv2
import numpy as np
import requests
import math
import time

def initialize_images(detector, denominations):
    trainImgs = []
    trainKPs = []
    trainDescs = []
    for i in denominations:
        for j in range(2):
            img_url = '/home/sady/Descargas/PythonVirtualEnv2.7/The-Maze/Billetes-descargados/' + str(i) + '_' + str(j+1) + '.jpg'
            trainImg = cv2.imread(img_url)
            trainKP,trainDesc=detector.detectAndCompute(trainImg,None)
            trainImgs.append(trainImg)
            trainKPs.append(trainKP)
            trainDescs.append(trainDesc)
    return trainImgs, trainKPs, trainDescs
    
def runInParallel(*fns):
  proc = []
  for fn in fns:
    p = Process(target=fn)
    p.start()
    proc.append(p)
  for p in proc:
    p.join()

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

def display_loading():
    gif = cv2.VideoCapture('loading.gif')

    frames = convert_gif_to_frames(gif)

    size = len(frames)

    ## Display the gif
    i = 0

    t_end = time.time() + 105
    while time.time() < t_end:
        frames[i] = cv2.resize(frames[i], (800,480))
        cv2.imshow('Cargando', frames[i])
        if cv2.waitKey(100)&0xFF == 27:
            break
        i = i+1
        if i == size-1:
            i = 0

    cv2.destroyAllWindows()

def main_real():
    url = 'http://192.168.0.11:8080/shot.jpg'
    while True:
        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        QueryImgBGR = cv2.imdecode(img_arr, -1)
        cv2.imshow('Imagen',QueryImgBGR)
        if cv2.waitKey(10)==ord('q'):
            cv2.destroyAllWindows()
            runInParallel(main, display_loading)
        if cv2.waitKey(13)==13:
            break
    cv2.destroyAllWindows()


def main():
    url = 'http://192.168.0.11:8080/shot.jpg'
    MIN_MATCH_COUNT=80

    detector=cv2.xfeatures2d.SIFT_create()

    FLANN_INDEX_KDITREE=0
    flannParam=dict(algorithm=FLANN_INDEX_KDITREE,tree=5)
    flann=cv2.FlannBasedMatcher(flannParam,{})
    denominations = [1,2,5,10,20,50]
    trainImgs, trainKPs, trainDescs = initialize_images(detector, denominations)
    total = 0
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    QueryImgBGR = cv2.imdecode(img_arr, -1)
    QueryImg=cv2.cvtColor(QueryImgBGR,cv2.COLOR_RGB2GRAY)
    queryKP,queryDesc=detector.detectAndCompute(QueryImg,None)
    matches = []
    for trainDesc in trainDescs:
        match = flann.knnMatch(queryDesc, trainDesc, k=2)
        matches.append(match)
    goodMatches = []
    for i in range(len(matches)):
        goodMatch = []
        goodMatches.append(goodMatch)
    i = 0
    for match in matches:
        for m, n in match:
            if(m.distance<0.75*n.distance):
                goodMatches[i].append(m)
        i = i + 1
    i = 0
    arri = []
    for goodMatch in goodMatches:
        leng = len(goodMatch)
        arri.append(leng)
    maxPointsIndexes = np.argpartition(arri, -2)[-2:]
    i = 0
    print(len(goodMatches[maxPointsIndexes[0]]))
    print(len(goodMatches[maxPointsIndexes[1]]))
    for i in range(len(maxPointsIndexes)):
        if(len(goodMatches[maxPointsIndexes[i]])>MIN_MATCH_COUNT):
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
            #cv2.polylines(QueryImgBGR,[np.int32(queryBorder)],True,(255,0,0),5)
            total = total + denominations[int(math.floor(maxPointsIndexes[i]/2))]
        i = i + 1
    if total > 0:
        print('Hay ' + str(total) + ' mil pesos')
    else:
        print('No se encontraron billetes de ninguna denominación')
    cv2.destroyAllWindows()

#main()
main_real()
