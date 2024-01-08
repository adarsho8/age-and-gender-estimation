# Use following command to download the necessary modules
#pip3 install mediapipe numpy tensorflow opencv-contrib-python

import mediapipe as mp #pip3 install mediapipe
import time
import math
import numpy as np #pip3 install numpy
# import systemcheck


#pip3 install tensorflow
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json

import cv2 #pip3 install opencv-contrib-python


# constants
Close_frames = 1
FONTS = cv2.FONT_HERSHEY_SIMPLEX
TEXT_COLOR = (0,250, 0) # BGR 0 - 255


face_outline=[ 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,67, 109]

LIPS=[ 61, 146, 91, 181, 84, 17, 314, 405, 321, 375,291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78 ]

LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]

RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  

LEFT_EYEBROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]

RIGHT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]

NOSE = [8,240,460]

face_detection = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)#conf=True)
face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence =0.5, min_tracking_confidence=0.5)

########### GENDER DETECTION MODEL LOAD ############################
classes = ['Male','Female']
model = load_model('./models/gender_detection.model') # Load Gender Detection Model

########### AGE DETECTION MODEL LOAD ############################

ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

ageProto = "./models/age_deploy.prototxt"
ageModel = "./models/age_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

ageNet = cv2.dnn.readNet(ageModel,ageProto) #Load Age Detection DNN Model

##########################################

cam = cv2.VideoCapture(0) #Get access to Camera

while True:
    _, raw_img = cam.read() # Read Image from Camera
    if _:
        start_time = time.time() #Start time, helpful in estimating FPS

        print("Raw Image:", raw_img.shape)

        ########## FACE DETECTION ##########
        x,y,w,h = 0,0,0,0
        img_w = raw_img.shape[1]
        img_h = raw_img.shape[0]

        face_detection_results = face_detection.process(raw_img[:,:,::-1])# Detect Faces in Images

        if face_detection_results.detections:
            for face in face_detection_results.detections:

                print(f'FACE CONFIDENCE: {round(face.score[0], 2)}')
                if face.score[0] < 0.8: # Check confidence of next Face if confidence of current face is less
                    continue

                face_data = face.location_data

                x,y,w,h = int(img_w*face_data.relative_bounding_box.xmin), \
                            int(img_h*face_data.relative_bounding_box.ymin), \
                            int(img_w*face_data.relative_bounding_box.width), \
                            int(img_h*face_data.relative_bounding_box.height)
                break #Break if found Face as we are using only 1 Face

      
        if x+y+w+h > 0:
            print("Detected Face Points:", x,y,w,h)
            x = x - 10 # Extend X axis to cover whole Face
            y = y - 40 # Extend Y axis to cover forehead
            w = w + 20 # Extend width as we moved X axis bit left
            h = h + 40 # Extend height as we moved Y axis bit up

            if x<0:
                x = 0
            if y<0:
                y = 0

            # cv2.imshow("IMG", raw_img)
            # cv2.imshow("Face Image", face_img)
            # cv2.waitKey(10)


            face_img = raw_img[y:y+h, x:x+w]
            print("Face Image:", face_img.shape)
            cv2.rectangle(raw_img, (x,y), (x+w, y+h), (255,0,0), 2) # Draw Rectangle around Face

            ########## GENDER DETECTION ##########
            face_crop = cv2.resize(face_img, (96,96)) # Resize the image based on model requirement
            face_crop = face_crop.astype("float") / 255.0 # Normalize the Image
            face_crop = img_to_array(face_crop) #Convert Image to Array
            face_crop = np.expand_dims(face_crop, axis=0) # [R][G][B] -> [[R][G][B]]

            conf = model.predict(face_crop)[0] #Get Confidence Prediction for both Gender

            idx = np.argmax(conf)# Get the index of maximum confidence
            label = "{}: {:.2f}%".format(classes[idx], conf[idx] * 100)
            cv2.putText(raw_img, label, (x, y-50), FONTS, 1, TEXT_COLOR, 1)

        
            ########## AGE DETECTION ##########
            blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False) # Convert Image to blob (Required by DNN type Model)
            ageNet.setInput(blob) # Feed the blob Image to Age Detector
            agePreds = ageNet.forward() #Get the Predictions Confidences
            age = ageList[agePreds[0].argmax()] # Get the Estimated Age Label

            print("Age : {}, conf = {:.3f}".format(age, agePreds[0].max()))
            cv2.putText(raw_img, "Age : {}".format(age), (x, y-30), FONTS, 1, TEXT_COLOR, 1)



        cv2.imshow('frame', raw_img)
        cv2.waitKey(1)

        # calculating  frame per seconds FPS
        time_taken = time.time()-start_time
        fps = 1/time_taken
        print("FPS:", fps)

