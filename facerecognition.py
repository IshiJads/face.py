import cv2
import numpy as np
import os
haarfile = "haarcascade_frontalface_default.xml"
datasets = "datasets"
subdata = "Ishita"
path = os.path.join(datasets,subdata)
if not path:
    os.mkdir(path)
(width,height) = (200,200)
face_cascade = cv2.CascadeClassifier(haarfile)
webcam = cv2.VideoCapture(0)
count = 1
while count<=30:
    (status,image) = webcam.read()
    grey = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grey,1.3,4)
    for (x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        face = grey[y:y+h,x:x+w]
        faceresize = cv2.resize(face,(width,height))
        cv2.imwrite("%s/%s.png"%(path,count),faceresize)
    count+=1
    cv2.imshow("facedetector",image)
    cv2.waitKey(0)
