#!/opt/intel/openvino/Python

#python3 inference_URL_Faces.py

import time
import cv2
import numpy as np
import cv2 as cv
import math

url  =  'http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/TearsOfSteel.mp4'
vcap = cv2.VideoCapture(url)

#if not vcap.isOpened():
#    print "File Cannot be Opened"

LABELS = ('background',
          'person', 'bicycle', 'bird', 'boat',
          'bottle', 'bus', 'car', 'cat', 'chair',
          'cow', 'diningtable', 'dog', 'horse',
          'motorbike', 'person', 'pottedplant',
          'sheep', 'sofa', 'train', 'tvmonitor')

# Load the model.
net = cv.dnn.readNet('face-detection-adas-0001.xml','face-detection-adas-0001.bin')
#net = cv2.dnn.readNet('mobilenet-ssd.xml','mobilenet-ssd.bin')
#net = cv2.dnn.readNet('MobileNetSSD_deploy.xml','MobileNetSSD_deploy.bin')
#net = cv2.dnn.readNet('frozen_yolo_v3.xml','frozen_yolo_v3.bin')

# Specify target device.
net.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD) 

width  = 640
height = 480
dim = (width, height)

while(True):
    # Capture frame-by-frame
    ret, frame = vcap.read()
    
    # Resize frame
    image = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    
    # Prepare input blob and perform an inference.
    blob = cv.dnn.blobFromImage(frame, size=(width, height), ddepth=cv.CV_8U)
    net.setInput(blob)
    out = net.forward()
    
    # Draw detected faces on the frame.
    for detection in out.reshape(-1, 7):
        confidence = float(detection[2])
        xmin = int(detection[3] * frame.shape[1])
        ymin = int(detection[4] * frame.shape[0])
        xmax = int(detection[5] * frame.shape[1])
        ymax = int(detection[6] * frame.shape[0])
        if confidence > 0.5:
            cv.rectangle(frame, (xmin, ymin), (xmax, ymax), color=(0, 255, 0))
    

    
    
    #print cap.isOpened(), ret
    if image is not None:
        # Display the resulting frame
        cv2.imshow('video',image)
        # Press q to close the video windows before it ends if you want
        if cv2.waitKey(22) & 0xFF == ord('q'):
            break
    else:
        print ("Frame is None")
        break

# When everything done, release the capture
vcap.release()
cv2.destroyAllWindows()
print ("Video stop")
