import cv2
import mediapipe as mp
import numpy as np
import time
import math

WIDTH, HEIGHT = 1280, 720
SMOOTHING = 0.15
PINCH_THRESHOLD = 50
model_path = 'hand_landmarker.task'

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands = 1, min_detection_configdence=0.7)
mp_draw = mp.solutions.drawing_utils

while True:
  success, frame = cap.read()
  if not success:
    break

  frame = cv2.flip(frame, 1)
  rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  cv2.imshow("Hand Drawer", frame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

  results = hands.process(rgb_frame)




cap.release()
cv2.destroyAllWindows()
