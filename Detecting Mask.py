# Importing the required libraries
from tensorflow import keras
import imutils 
import numpy as np
import time
import cv2
import os
from imutils.video import VideoStream
import sys

# Detect and Predict mask function
def detect_and_predict(frame, faceNet, maskNet):
    """
    Detects the mask on the face and returns face location and predictions
    """
    # Extracting height and width from the shape of frame
    (h, w) = frame.shape[:2]
    
    # Creating blob out of the input frame
    blob = cv2.dnn.blobFromImage(frame,
                                 1.0,  # Scaling factor
                                 (224, 224), # size
                                 (104.0, 177.0, 123.0)) # Mean subtraction
    
    # Passing the blob through the network and get detection output
    faceNet.setInput(blob)
    detections = faceNet.forward()
    
    faces = []
    locs = []
    preds = []
    
    # Looping over the detections and finding the coordinates to draw boxes around
    for i in range(0, detections.shape[2]):
        # Retrieving the probabilities/confidence from the detections
        confidence = detections[0, 0, i, 2]
        threshold = 0.5
        
        # Filter out weakdetections
        if confidence > threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h]) # Detected face area
            (startX, startY, endX, endY) = box.astype("int")
            
            # Ensuring the bounding box fall within the dimensions of the frame
            (startX, startY) = (max(0, startX), max(0, startY))
             
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            
            # Extract the face ROI and convert it from BGR to RGB channel
            # and resize it to 224x224 and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = keras.preprocessing.image.img_to_array(face)
            face = keras.applications.mobilenet_v2.preprocess_input(face)
            
            # adding the face and box coordinates to the lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))
            
    # Make predictions only if it has atleast one face
    if len(faces) > 0:
            faces = np.array(faces, dtype="float32")
            preds = maskNet.predict(faces, batch_size=32)
    
    return (locs, preds)
        

""" prototxt file(s) which define the model architecture 
 (i.e., the layers themselves)
 caffemodel file which contains the weights for the actual layers
"""

# Loading our serialized face detector model 
prototxtPath = "face_detector/deploy.prototxt"
weightsPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Loading our face mask detector model from disk
maskNet = keras.models.load_model("mask_detector.h5")


print("[INFO] Starting video stream....")
vs = cv2.VideoCapture(0)
# address = "http://192.168.43.1:8080/video"
# vs.open(address)
#vs = VideoStream(src=0).start()


# Looping over frames from the video
while True:
    # Grab the frame from the video and resize it to 400 pixels
    _, frame = vs.read()
    frame = imutils.resize(frame, width=800)
    
    # Detect faces in the frame 
    (locs, preds) = detect_and_predict(frame, faceNet, maskNet)
    # loop over the detected face locations
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred
        
        # Determine the class label and color
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        
        # Include probability in the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask)*100)
        
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.rectangle(frame, (startX, startY-40), (endX, startY), color, -1)
        
        # Display the label and bounding box rectangle on the output frame
        cv2.putText(frame, # Image
                    label, # Text to put
                    (startX, startY - 10), # Coordinates of bottom-left corner of the text
                    cv2.FONT_HERSHEY_SIMPLEX, # Font
                    0.8, # font scale
                    (255, 255, 255),
                    2 # thickness
                    )
        
    # Show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF  #0xFF == 11111111
    
    # Quit when pressed 'q'
    if key == ord("q"):
        break
    
cv2.destroyAllWindows()
vs.release()

"""
# Testing
## Starting video stream
vs = VideoStream(src=0).start()
frame = vs.read()

## Creating blob
blob = cv2.dnn.blobFromImage(frame, 1, (224, 224), (104, 177, 123))

## Inputting blob to our faceNet(face detection) model
faceNet.setInput(blob)
## Forwarding the input to the faceNet model
detects = faceNet.forward() 

## Closing everthing
vs.stop()
cv2.destroyAllWindows()
"""

# References:-
#**Youtube
#**pyimagesearch.com
#**keras.io
#**towardsdatascience.com
