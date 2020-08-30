from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import cv2
import os

model_face = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')
model_eye = cv2.CascadeClassifier('data/haarcascades/haarcascade_eye.xml')

image_counter = 0

classes = ["background", "aeroplane", "bicycle", "bird", "boat",
"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
"sofa", "train", "tvmonitor"]

net = cv2.dnn.readNetFromCaffe("data/MobileNetSSD/MobileNetSSD_deploy.prototxt.txt", "data/MobileNetSSD/MobileNetSSD_deploy.caffemodel")
colors = np.random.uniform(0, 255, size=(len(classes), 3))

vs = VideoStream().start()
video_capture = cv2.VideoCapture(0)

fps = FPS().start()

while True:
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = model_face.detectMultiScale(gray, 1.3, 5)

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.75:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            if startX < 0:
                startX = 0
            if startY < 0:
                startY = 0
            if endX < 0:
                endX = 0
            if endY < 0:
                endY = 0
            _frame = frame[startY:endY, startX:endX]
            
            label_name = classes[idx]
            label_confidence = confidence

            y = startY - 15 if startY - 15 > 15 else startY + 15

            label = "{}: {:.2f}%".format(classes[idx],
                    confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                colors[idx], 2)
            cv2.putText(frame, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)   
            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                eyes = model_eye.detectMultiScale(roi_gray)
                for (ex,ey,ew,eh) in eyes:
                    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow('frame',frame)
    if not os.path.exists('images'):
            os.makedirs('images')
            print("Created new directory " + 'images')

    k = cv2.waitKey(1)
    if k % 256 & 0xFF == ord('q'):
        break
    elif k % 256 == 32: 
        img_name = "opencv_frame_{}.png".format(image_counter)
        cv2.imwrite("images/frame%d.jpg" % image_counter, frame)
        print("Image Saved: images/frame%d.jpg" % image_counter)
        image_counter += 1

video_capture.release()
cv2.destroyAllWindows()