import cv2
import numpy as np 
import sys
import tempfile
import subprocess

GREEN = (000,255,000)
RED = (000,000,255)

cascPath = "VAMOS_VER/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

camera_video = cv2.VideoCapture(0)
cinza = cv2.COLOR_BGR2GRAY

rodando = True 

#cv2.namedWindow("Detecção de Faces", cv2.WINDOW_NORMAL)


while rodando:
    ret, frame = camera_video.read()

    gray = cv2.cvtColor(frame, cinza)

    caras = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30,30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    if len(caras) > 0:
        for (x, y, w, h) in caras:
            cv2.rectangle(frame, (x, y), (x + w, y + h), GREEN, 2)
        mensagem = "Rosto detectado"
        cor_mensagem = GREEN
        #print("Rosto Detectado")
    else:
        mensagem = "Nenhum rosto detectado"
        cor_mensagem = RED
        #print("Rosto não Detectado")

    cv2.putText(frame, mensagem, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, cor_mensagem, 2)
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera_video.release()
cv2.destroyAllWindows()