import cv2
import os
import imutils

personName='con_mascarilla'
dataPath='C:/Users/xoant/Desktop/Proyecto JIC/Data'
personPath=dataPath +'/'+personName
if not os.path.exists(personPath):
    print('Carpeta creada: ', dataPath)
    os.makedirs(personPath)

cap= cv2.VideoCapture('ConMascarillaEdwar.mp4')
faceClassif=cv2.CascadeClassifier(cv2.data+'haarcascade_frontalface_default.xml')
cont=0

while True:
    ret,frame = cap.read()
    if ret== False: break
    frame= imutils.resize(frame, width=640)
    gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame=frame.copy()
    faces= faceClassif.detectMultScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,+h),(0,255,0),2)
        rostro= auxFrame[y:y+h,x:x+w]
        rostro=cv2.resize(rostro,(150,150), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(personPath+'/rostro_{}.jpg'.format(cont),rostro)
        cont=cont+1

    cv2.imshow('frame',frame)

    k=cv2.waitKey(1)
    if k== 27 or cont>= 150:
        break
cap.release()
cv2.destroyAllWindows