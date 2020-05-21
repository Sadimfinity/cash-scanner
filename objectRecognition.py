import cv2
import numpy as np
import requests
import math

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



def main():
    url = 'http://192.168.0.11:8080/shot.jpg'
    MIN_MATCH_COUNT=60

    detector=cv2.xfeatures2d.SIFT_create()

    FLANN_INDEX_KDITREE=0
    flannParam=dict(algorithm=FLANN_INDEX_KDITREE,tree=5)
    flann=cv2.FlannBasedMatcher(flannParam,{})
    denominations = [1,2,5,10,20,50]
    trainImgs, trainKPs, trainDescs = initialize_images(detector, denominations)
    while True:
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
        for goodMatch in goodMatches:
            if(len(goodMatch)>MIN_MATCH_COUNT):
                tp=[]
                qp=[]
                for m in goodMatch:
                    trainKP = trainKPs[i]
                    tp.append(trainKP[m.trainIdx].pt)
                    qp.append(queryKP[m.queryIdx].pt)
                tp,qp=np.float32((tp,qp))
                H,status=cv2.findHomography(tp,qp,cv2.RANSAC,3.0)
                trainImg = trainImgs[i]
                h,w,_=trainImg.shape
                trainBorder=np.float32([[[0,0],[0,h-1],[w-1,h-1],[w-1,0]]])
                queryBorder=cv2.perspectiveTransform(trainBorder,H)
                # convert the grayscale image to binary image
                ret,thresh = cv2.threshold(QueryImg,127,255,0)
                cv2.polylines(QueryImgBGR,[np.int32(queryBorder)],True,(255,0,0),5)
                total = total + denominations[int(math.floor(i/2))]
            i = i + 1
        print('Hay ' + str(total) + ' mil pesos')
        cv2.imshow('result',QueryImgBGR)
        if cv2.waitKey(10)==ord('q'):
            break
    cv2.destroyAllWindows()

main()
