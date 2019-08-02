import cv2
import numpy as np
from keras.models import load_model

detection_model_path = "haarcascade_frontalface_default.xml"
emotion_model_path = 'fer2013_mini_XCEPTION.102-0.66.hdf5'

faceCascade = cv2.CascadeClassifier(detection_model_path)

cap = cv2.VideoCapture(0)
if (cap.isOpened() == False): 
  print("Unable to read camera feed")
 
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
 
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
 
while(True):
  ret, frame = cap.read()
 
  if ret == True: 
     
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(1,1),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    name = "Unknown"

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y - 35), (x+w, y), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (x + 6, y - 6), font, 1.0, (255, 255, 255), 1)

    out.write(frame)
 
    cv2.imshow('frame',frame)
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
 
  else:
    break 
 
cap.release()
out.release()
cv2.destroyAllWindows() 

