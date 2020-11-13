import cv2
import numpy as np

# Non-maximum suppression parameters. The first is a lower bound for the prediction confidence;
# the second is the overlap threshold.
confThreshold = 0.1
nmsThreshold = 0.3
# Banana ID from the COCO dataset
banana_id = 46


def find_banana(nn_output, im):
    ht, wt, ct = im.shape
    bbox = []
    conf = []

    # Detect boxes containing bananas
    for output in nn_output:
        for det in output:
            scores = det[5:]
            class_id = np.argmax(scores)
            if class_id == banana_id:
                confidence = scores[class_id]
                if confidence > confThreshold:
                    w, h = int(det[2]*wt), int(det[3]*ht)
                    x, y = int((det[0]*wt)-w/2), int((det[1]*ht)-h/2)
                    bbox.append([x, y, w, h])
                    conf.append(float(confidence))
    # Indices of non-maximum suppression boxes
    indices = cv2.dnn.NMSBoxes(bbox, conf, confThreshold, nmsThreshold)

    # Draw boxes around bananas
    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(im, (x, y), (x+w, y+h), (255, 0, 255), 2)
        cv2.putText(im, "BANANA"+f' {int(conf[i]*100)}%',
                    (x, y-10), cv2.FONT_HERSHEY_PLAIN, 0.6, (255, 0, 255), 1)


# Input video
cap = cv2.VideoCapture("test.mp4")
# Image scaling parameter required for NN input
wh = 320

# YOLOv3 network architecture and model weights
modelConfig = 'yolov3-tiny.cfg'
modelWeights = 'yolov3-tiny.weights'
# OpenCV deep learning from the Darknet NN framework
net = cv2.dnn.readNetFromDarknet(modelConfig, modelWeights)

while True:
    success, img = cap.read()
    # Pre-process image
    blob = cv2.dnn.blobFromImage(img, 1/255, (wh, wh), [0, 0, 0], 1, crop=False)
    # Apply NN model
    net.setInput(blob)
    layerNames = net.getLayerNames()
    outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)
    # Determine which image regions contain bananas and draw boxes around them
    find_banana(outputs, img)
    cv2.imshow("NanaTracker", img)
    # Quit before end of video by pressing 'q' on keyboard
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
