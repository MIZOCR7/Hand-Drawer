import cv2
import mediapipe as mp
import numpy as np
import time
import math

WIDTH, HEIGHT = 1280, 720
SMOOTHING = 0.15
PINCH_THRESHOLD = 50
model_path = 'hand_landmarker.task'

COLORS = [(255, 0, 255), (255, 0, 0), (0, 255, 0), (0, 255, 255), (0, 0, 0)]
color_names = ["PURPLE", "BLUE", "GREEN", "YELLOW", "ERASER"]

canvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)


cur_color = COLORS
px, py = 0, 0
plocX, plocY = 0, 0
latest_result = None

CONNECTIONS = [(0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8), (5,9), (9,10), (10,11), (11,12), (9,13), (13,14), (14,15), (15,16), (13,17), (0,17), (17,18), (18,19), (19,20)]

def result_callback(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

def show_menu():
    menu_img = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    cv2.rectangle(menu_img, (0, 0), (WIDTH, HEIGHT), (40, 40, 40), cv2.FILLED)
    cv2.putText(menu_img, "Hand Drawer", (320, 150), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 3)

    cv2.line(menu_img, (320, 170), (960, 170), (0, 255, 0), 2)


    instructions = [
    "1. COLOR SELECT: Hover index finger over top bars.",
    "2. DRAW: Pinch Index + Thumb (Distance < 50).",
    "3. SMOOTHING: Built-in EMA Filter for clean lines",
    "4. CLEAR CANVAS: Press 'C' key",
    "5. EXIT APP: Press 'Q' key",
    "",
    ">>> PRESS 'S' TO START DRAWING <<<"
    ]

    for i, line in enumerate(instructions):
        color = (0, 255, 0) if "START" in line else (200, 200, 200)
        cv2.putText(menu_img, line, (350, 250 + (i * 60)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


    while True:
        cv2.imshow("Hand Drawer", menu_img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            break
        if key == ord('q'):
            exit()

show_menu()


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=result_callback,
    num_hands=1,
    min_hand_detection_confidence=0.8
)


with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    cap.set(3, WIDTH)
    cap.set(4, HEIGHT)

    while cap.isOpened():
      success, frame = cap.read()
      if not success:
          break






      cv2.imshow("Hand Drawer", frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break


cap.release()
cv2.destroyAllWindows()
