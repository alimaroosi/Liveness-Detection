
###### origin code from 
###https://github.com/ternaus/datasouls_antispoof
##matplotlib inline

import matplotlib
from pylab import imshow
import pylab
import matplotlib.pyplot as plt
import numpy as np

import cv2

import torch

import seaborn as sns

import pandas as pd

#########
import imutils
import time

#########




##!pip install -U albumentations

import albumentations as albu

from albumentations.pytorch.transforms import ToTensorV2
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
from iglovikov_helper_functions.utils.image_utils import load_rgb

#!pip install iglovikov_helper_functions > /dev/null

##!wget https://habrastorage.org/webt/g7/nc/jw/g7ncjwscv4o5ft86fj2okl4adec.png > /dev/null
##!wget https://habrastorage.org/webt/yd/4i/r8/yd4ir8hnmhgmly1hgwcoyyhbgdm.jpeg > /dev/null

##!pip install datasouls_antispoof  > /dev/null

from datasouls_antispoof.pre_trained_models import create_model

from datasouls_antispoof.class_mapping import class_mapping
# instanciar camara
# cv2.namedWindow('liveness_detection')
# cam = cv2.VideoCapture(0)
# ret, im = cam.read()
# im = imutils.resize(im, width=720)
# imshow(im)



###### package for liveness and face detections from 
###https://github.com/AhmetHamzaEmra/Intelegent_Lock
import face_recognition
from FromLock.livenessmodel import get_liveness_model
Level_Liveness0to95=0; ### 0 to  0.95   
########

# Get the liveness network
model_live = get_liveness_model()

# load weights into new model
model_live.load_weights("FromLock/model/model.h5")
print("Loaded model from disk")


video_capture = cv2.VideoCapture(0)
video_capture.set(3, 640)
video_capture.set(4, 480)

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
input_vid = []

##### create model to test image is {"real": 0, "replay": 1, "printed": 2, "2dmask": 3})
model = create_model("tf_efficientnet_b3_ns")
# model = create_model("swsl_resnext50_32x4d")


model.eval();

######


while True:
    # Grab a single frame of video
    if len(input_vid) < 24:

        ret, frame = video_capture.read()

        liveimg = cv2.resize(frame, (100,100))
        liveimg = cv2.cvtColor(liveimg, cv2.COLOR_BGR2GRAY)
        input_vid.append(liveimg)
    else:
        ret, frame = video_capture.read()
        if Level_Liveness0to95==0 :
          pred=[[1]]
        else:
          liveimg = cv2.resize(frame, (100,100))
          liveimg = cv2.cvtColor(liveimg, cv2.COLOR_BGR2GRAY)
          input_vid.append(liveimg)
          inp = np.array([input_vid[-24:]])
          inp = inp/255
          inp = inp.reshape(1,24,100,100,1)
          pred = model_live.predict(inp)
          input_vid = input_vid[-25:]

        if pred[0][0]>= Level_Liveness0to95: #.95:

            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = frame; #cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Only process every other frame of video to save time
            if process_this_frame:
                # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(small_frame)

            process_this_frame = not process_this_frame
            frame_Bkup=frame

            # Display the results
            for (top, right, bottom, left) in face_locations:
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size when small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                # top *= 4
                # right *= 4
                # bottom *= 4
                # left *= 4
                ## expand the box around the face to cover part of body and environment
                right_temp=np.min([small_frame.shape[1],right+(right-left)])
                left_temp=np.max([2,left-(right-left)])
                top_temp=np.max([2,top-(np.floor((bottom-top)/2))])
                bottom_temp=np.min([small_frame.shape[0],bottom+(bottom-top)])
                right=right_temp
                left=left_temp
                top=top_temp.astype('int64')
                bottom=bottom_temp
                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                plt.imshow(frame)
                plt.pause(2)
                # plt.show()
                ### cropp the image to choose a person
                frame_crop= frame_Bkup[top:bottom,left:right] 

                #### analysis image of the person {"real": 0, "replay": 1, "printed": 2, "2dmask": 3}
                transform = albu.Compose([albu.PadIfNeeded(min_height=400, min_width=400),
                          albu.CenterCrop(height=400, width=400), 
                          albu.Normalize(p=1), 
                          albu.pytorch.ToTensorV2(p=1)], p=1)

                with torch.no_grad():
                  prediction = model(torch.unsqueeze(transform(image=frame_crop)['image'], 0)).numpy()[0]
                  df2= pd.DataFrame({"prediction": prediction, "class_name": class_mapping.keys()})
                  print(df2)
                  cv2.destroyAllWindows()
                  cv2.destroyAllWindows()
                  plt.imshow(frame_crop)
                  # plt.show()
                  plt.pause(2)
                  plt.close()
                  sns.barplot(y="prediction", x="class_name",data=df2);#, x="prediction", y="class_name")
                  # plt.show()
                  plt.pause(2)

                
                tt=0
            # cv2.imshow('bbbv211',frame)
            # key = cv2.waitKey(1) & 0xFF

            




# model = create_model("tf_efficientnet_b3_ns")
# model.eval();

# image_replay = load_rgb("g7ncjwscv4o5ft86fj2okl4adec.png")
image_replay=frame
imshow(image_replay)
plt.show()
transform = albu.Compose([albu.PadIfNeeded(min_height=400, min_width=400),
                          albu.CenterCrop(height=400, width=400), 
                          albu.Normalize(p=1), 
                          albu.pytorch.ToTensorV2(p=1)], p=1)
with torch.no_grad():
  prediction = model(torch.unsqueeze(transform(image=image_replay)['image'], 0)).numpy()[0]
  df2= pd.DataFrame({"prediction": prediction, "class_name": class_mapping.keys()})
  print(df2)
  cv2.destroyAllWindows()
  cv2.destroyAllWindows()
sns.barplot(y="prediction", x="class_name",data=df2);#, x="prediction", y="class_name")
plt.pause(5)
# image_real = load_rgb("yd4ir8hnmhgmly1hgwcoyyhbgdm.jpeg")
plt.show()

