import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

#holistic model
mp_holistic = mp.solutions.holistic
#drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #color conversion
    image.flags.writeable = False #image not writable
    results = model.process(image) #make prediction
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) #undoing color conversion
    return image, results

def draw_landmarks(image, results): #draws connections on frame
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)


cap = cv2.VideoCapture(0)
#set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holostic:
    while cap.isOpened():
        #reads feed
        ret, frame = cap.read()


        #mke detection
        image , results = mediapipe_detection(frame , holostic)
        print(results)

        #draw landmarks
        draw_landmarks(image, results)

        #showing to screen
        cv2.imshow('OpenCV Feed',image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

