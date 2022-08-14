from calendar import c
from tkinter import font
import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot
import datetime

from numpy import var

 
url = 'http://192.168.226.131:8080/video'
cap = cv2.VideoCapture(url)
detector = FaceMeshDetector(maxFaces=1)
plotY = LivePlot(640, 360, [20, 50], invert=True)
 
idList = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]
ratioList = []
blinkCounter = 0
counter = 0
color = (255, 0, 255)
 
while True:
 
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
 
    success, img = cap.read()
    img, faces = detector.findFaceMesh(img, draw=False)
 
    if faces:
        face = faces[0]
        for id in idList:
            cv2.circle(img, face[id], 5,color, cv2.FILLED)
 
        leftUp = face[159]
        leftDown = face[23]
        leftLeft = face[130]
        leftRight = face[243]
        lenghtVer, _ = detector.findDistance(leftUp, leftDown)
        lenghtHor, _ = detector.findDistance(leftLeft, leftRight)
 
        cv2.line(img, leftUp, leftDown, (0, 200, 0), 3)
        cv2.line(img, leftLeft, leftRight, (0, 200, 0), 3)
 
        ratio = int((lenghtVer / lenghtHor) * 100)
        ratioList.append(ratio)
        if len(ratioList) > 3:
            ratioList.pop(0)
        ratioAvg = sum(ratioList) / len(ratioList)
 
        if ratioAvg < 35 and counter == 0:
            blinkCounter += 1
            color = (0,200,0)
            counter = 1
        if counter != 0:
            counter += 1
            if counter > 10:
                counter = 0
                color = (255,0, 255)
     
        if  blinkCounter >= 6:
           blinkCounter = 0
        
        
        

 
        cvzone.putTextRect(img, f'Blink Count: {blinkCounter}', (50, 100),
                           colorR=color)
 
        
        img = cv2.resize(img, (640, 360))
        imgStack = cvzone.stackImages([img], 2, 1)
    else:
        img = cv2.resize(img, (640, 360))
        imgStack = cvzone.stackImages([img, img], 2, 1)
    #dat = str(datetime.datetime.now())
    #imgStack = cv2.putText(imgStack, dat, (10,50), font, 1, (255, 255, 0), 2, cv2.LINE_AA)
    

    font = cv2.FONT_HERSHEY_SIMPLEX
    
    datet = str(datetime.datetime.now())
    
    imgStack = cv2.putText(imgStack, datet, (10, 100), font, 1,
                           (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Image",  imgStack)
    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()