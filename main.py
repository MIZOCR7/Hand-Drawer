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
cur_color = COLORS[0]
px, py = 0, 0
plocX, plocY = 0, 0
latest_result = None

CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]


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

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        landmarker.detect_async(mp_image, int(time.time() * 1000))

        cv2.rectangle(frame, (0, 0), (WIDTH, 100), (25, 25, 25), cv2.FILLED)
        for i, color in enumerate(COLORS):
            x1 = 10 + i * 250
            x2 = 240 + i * 250
            cv2.rectangle(frame, (x1, 10), (x2, 90), color, cv2.FILLED)
            cv2.putText(frame, color_names[i], (x1 + 60, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if latest_result and latest_result.hand_landmarks:
            for landmarks in latest_result.hand_landmarks:
                ix = int(landmarks[8].x * WIDTH)
                iy = int(landmarks[8].y * HEIGHT)
                tx = int(landmarks[4].x * WIDTH)
                ty = int(landmarks[4].y * HEIGHT)

                clocX = plocX + (ix - plocX) * SMOOTHING
                clocY = plocY + (iy - plocY) * SMOOTHING

                dist = math.hypot(ix - tx, iy - ty)
                is_drawing = dist < PINCH_THRESHOLD

                line_color = (0, 255, 0) if is_drawing else (0, 0, 255)
                cv2.line(frame, (ix, iy), (tx, ty), line_color, 2)

                for start, end in CONNECTIONS:
                    p1 = landmarks[start]
                    p2 = landmarks[end]
                    cv2.line(
                        frame,
                        (int(p1.x * WIDTH), int(p1.y * HEIGHT)),
                        (int(p2.x * WIDTH), int(p2.y * HEIGHT)),
                        (255, 255, 255),
                        1,
                    )

                if clocY < 100:
                    px, py = 0, 0
                    if 10 < clocX < 240:
                        cur_color = COLORS[0]
                    elif 260 < clocX < 490:
                        cur_color = COLORS[1]
                    elif 510 < clocX < 740:
                        cur_color = COLORS[2]
                    elif 760 < clocX < 990:
                        cur_color = COLORS[3]
                    elif 1010 < clocX < 1250:
                        cur_color = COLORS[4]
                elif is_drawing:
                    if px == 0 and py == 0:
                        px, py = clocX, clocY
                    if cur_color == (0, 0, 0):
                        cv2.circle(canvas, (int(clocX), int(clocY)), 80, (0, 0, 0), -1)
                    else:
                        cv2.line(canvas, (int(px), int(py)), (int(clocX), int(clocY)), cur_color, 12)
                    px, py = clocX, clocY
                else:
                    px, py = 0, 0

                plocX, plocY = clocX, clocY
                
        img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(img_gray, 5, 255, cv2.THRESH_BINARY)
        frame_bg = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
        final_img = cv2.add(frame_bg, canvas)
        
        cv2.imshow("Hand Drawer", final_img)
        key = cv2.waitKey(1)
        
        if key & 0xFF == ord('q'):
            break
        if key & 0xFF == ord('c'):
            canvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
cap.release()
cv2.destroyAllWindows()        


