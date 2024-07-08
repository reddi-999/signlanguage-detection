import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np 
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
Classifier = Classifier("C:/Users/reddi/OneDrive/Desktop/Model/converted_keras/keras_model.h5" , "C:/Users/reddi/OneDrive/Desktop/Model/converted_keras/labels.txt")
offset = 20
imgsize = 300
counter = 0

labels = ["A","B","C"]


while True :
    success , img = cap.read ()
    imgOutput = img.copy()
    hands , img = detector.findHands(img)
    if hands :
        hand = hands [0]
        x,y,w,h = hand ['bbox']

        imgwhite = np.ones((imgsize,imgsize,3),np.uint8)*255

        imgCrop = img[y-offset : y + h + offset , x-offset : x + w + offset]
        imgCropshape = imgCrop.shape

        aspectratio = h/w

        if aspectratio > 1 :
            k = imgsize / h
            wcal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop , (wcal , imgsize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgsize- wcal )/2)
            imgwhite[: ,wGap : wcal + wGap ] = imgResize
            prediction , index = Classifier.getPrediction(imgwhite, draw= False)
            print(prediction, index)

        else :
            k = imgsize / w
            hcal = math.ceil(k*h)
            imgResize = cv2.resize(imgCrop , (imgsize, hcal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgsize- hcal )/2)
            imgwhite[hGap : hcal + hGap , : ] = imgResize
            prediction , index = Classifier.getPrediction(imgwhite, draw= False)


            cv2.rectangle(imgOutput,(x-offset,y-offset-70),(x -offset+400, y - offset+60-50),(0,255,0),cv2.FILLED)

            cv2.putText(imgOutput, labels[index],(x,y-30),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,0),2)
            cv2.rectangle(imgOutput,(x-offset,y-offset),(x + w + offset, y+h + offset),(0,255,0),4)
            
            cv2.imshow('ImageCrop' , imgCrop)
            cv2.imshow('Imagewhite' , imgwhite)

    cv2.imshow('Image' , imgOutput)
    cv2.waitKey(1)






