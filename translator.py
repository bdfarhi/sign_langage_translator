import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
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
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_face_mesh.FACEMESH_TESSELATION , mp_drawing.DrawingSpec(color=(80,110,100), thickness=1, circle_radius=1), mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), mp_drawing.DrawingSpec(color=(80,44,21), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))


def extract_endpoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    lh = np.array([[res.x, res.y, res.z, res.visibility] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    rh = np.array([[res.x, res.y, res.z, res.visibility] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    face = np.array([[res.x, res.y, res.z, res.visibility] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
    return np.concatenate([pose, face, lh, rh])


# path for exported data
DATA_PATH = os.path.join('MP_Data')
# Actions trying to detect
actions = np.array(['hello', 'thanks', 'iloveyou'])

# 30 videos worth of data
no_sequences = 30
# each video is 30 frames
sequence_length = 30

for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

cap = cv2.VideoCapture(0)
#set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holostic:
    for action in actions:
        for sequence in range(no_sequences):
            for frame_num in range(sequence_length):

                #reads feed
                ret, frame = cap.read()


                #mke detection
                image , results = mediapipe_detection(frame , holostic)
                print(results)

                #draw landmarks
                draw_landmarks(image, results)

                #applying collection wait logic
                if frame_num == 0 :
                    cv2.putText(image, 'STARTING COLLECTION', (120,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),1, cv2.LINE_AA)
                    # showing to screen
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(2000)
                else:
                    cv2.putText(image, 'Collecting frame for {} Video Number {}'.format(action, sequence), (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),1, cv2.LINE_AA)
                    # showing to screen
                    cv2.imshow('OpenCV Feed', image)
                #saving keypoints
                keypoints = extract_endpoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                #showing to screen
                # cv2.imshow('OpenCV Feed',image)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
    cap.release()
    cv2.destroyAllWindows()

#
# label_map = {label:num for num , label in enumerate(actions)}
# EXPECTED_SHAPE = 1662
#
# sequences , labels = [], []
# for action in actions :
#     for sequence in range(no_sequences):
#         window = []
#         for frame_num in range(sequence_length):
#             res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
#             if res.shape[0] != 1662:
#                 print(
#                     f"Unexpected shape at action: {action}, sequence: {sequence}, frame: {frame_num}, shape: {res.shape}")
#                 continue
#
#             window.append(res)
#         sequences.append(window)
#         labels.append(label_map[action])
# # for action in actions:
# #     for sequence in range(no_sequences):
# #         window = []
# #         for frame_num in range(sequence_length):
# #             file_path = os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num))
# #             try:
# #                 res = np.load(file_path)
# #                 if res.shape[0] != EXPECTED_SHAPE:
# #                     print(f"Padding corrupted data at action: {action}, sequence: {sequence}, frame: {frame_num}")
# #                     res = np.zeros(EXPECTED_SHAPE)  # Replace corrupted data with zeros
# #             except FileNotFoundError:
# #                 print(f"File not found: {file_path}. Padding with zeros.")
# #                 res = np.zeros(EXPECTED_SHAPE)
# #             window.append(res)
# #         sequences.append(window)
# #         labels.append(label_map[action])
# x = np.array(sequences)
# y = to_categorical(labels).astype(int)
#
# X_train, X_test, y_train, y_test = train_test_split(x ,y ,test_size=0.05)
#
# log_dir = os.path.join('Logs')
# tb_callback = TensorBoard(log_dir=log_dir)
#
# #Building neural network rchitecure
#
# model = Sequential()
# model.add(LSTM(64, return_sequences=True, activation='relu',input_shape=(30,1662)))
# model.add(LSTM(128, return_sequences=True, activation='relu'))
# model.add(LSTM(64, return_sequences=False, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(actions.shape[0], activation='softmax'))
#
# model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
#
# model.fit(x=X_train, y=y_train, epochs=2000, callbacks=[tb_callback])
