#######################################################
# DNN SSD MODELİ İLE TEK SINIFLI GÖRÜNTÜ SINIFLANDIRMA
#######################################################
import streamlit as st
import numpy as np
import cv2 as cv



model_bin = "MobileNetSSD_deploy.caffemodel"
config_test = "MobileNetSSD_deploy.prototxt.txt"

objName = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

net = cv.dnn.readNetFromCaffe(config_test, model_bin)

image = cv.imread("dogs.png")
h = image.shape[0]
w = image.shape[1]

layerNames = net.getLayerNames()
lastLayerID = net.getLayerId(layerNames[-1])
lastLayer = net.getLayer(lastLayerID)

blobImage = cv.dnn.blobFromImage(image, 0.007843, (300, 300), (127.5, 127.5, 127.5), False, crop=False)
net.setInput(blobImage)
cvOut = net.forward()

for detection in cvOut[0, 0, :, :]:
    score = float(detection[2])
    objIndex = int(detection[1])
    if score > 0.5:
        left = detection[3] * w
        top = detection[4] * h
        right = detection[5] * w
        bottom = detection[6] * h
        cv.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 0), 2)
        cv.putText(image, "score:%.2f, %s" % (score, layerNames[objIndex]),
                   (int(left) - 10, int(top) - 5), cv.FONT_HERSHEY_SIMPLEX,
                   0.7, (0, 0, 255), 2, 8)


# Assuming 'image' is the OpenCV image you want to display
image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
st.image(image_rgb, caption='Processed Image')



