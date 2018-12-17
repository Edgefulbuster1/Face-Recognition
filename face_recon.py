import numpy as np
import cv2
import pickle

#included Cascade
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt2.xml')

#recognition
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

#Accessing labels
labels={}
with open("labels.pickle",'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

cap=cv2.VideoCapture(0)

while(True):

    #Caapture frame-by-frame video
    ret, frame =cap.read()

    #convertion to gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    #printing face coordinates
    for (x,y,w,h) in faces:
        #print (x,y,w,h)
        #defining region of interest (roi)
        roi_gray=gray[y:y+h, x:x+w]     #[ycord, ycord+height] [xcord, xcord+width]
        roi_color=frame[y:y+h, x:x+w]

        #pridiction
        id_, conf = recognizer.predict(roi_gray)
        if conf>=4 and conf<=85:
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255,255,255)
            stroke = 2
            cv2.putText(frame,name,(x,y),font,1,color,stroke, cv2.LINE_AA)
        
        #saving face
        img_item = "18.png"
        cv2.imwrite(img_item, roi_color)

        #making a rectangle
        color=(255,0,0) #BGR
        stroke=2
        width=x+w
        height=y+h
        cv2.rectangle(frame,(x,y),(width,height),color,stroke)
    

    
    #display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

#Releasing the capture
cap.release()
cv2.destroyAllWindows()
