# Hand Drawer Application

## Project Overview
Hand Drawer is a computer vision application that allows users to draw on a digital canvas using hand gestures. The system utilizes MediaPipe for real-time hand landmark detection and OpenCV for graphics rendering.

## Prerequisites
To run the source code, you will need:
* Python 3.9 or higher
* A functional webcam

## Installation
1. Clone or download the project folder.
2. Ensure the following files are in the same directory:
   * main.py (Entry point)
   * feature_1.py (Gesture logic)
   * hand_landmarker.task (AI Model)
   * gesture_recognizer.task (Gesture Model)
3. Install the required libraries:
   pip install opencv-python mediapipe numpy

## Execution
Run the application using the following command:
python main.py

## Controls and Gestures
* Start Application: Press 'S' on the menu screen.
* Drawing: Pinch your Index finger and Thumb together (Distance < 50).
* Color Selection: Hover your index finger over the colored rectangles at the top of the screen.
* Clear Canvas: Either press the 'C' key or show an Open Palm gesture (all 5 fingers up).
* Screenshot: Show a Peace Sign gesture (Index and Middle fingers up).
* Exit: Press the 'Q' key.

## Project Structure
* main.py: Handles the camera feed, model initialization, coordinate smoothing (EMA Filter), and the main drawing loop.
* feature_1.py: Contains the process_gestures function to interpret finger states and trigger actions like clearing the canvas or saving screenshots.
* hand_landmarker.task: The binary model file required by MediaPipe to track 21 hand joints.

## Build Instructions (For Developers)
To create a standalone executable, use the following PyInstaller command:
pyinstaller --noconfirm --onefile --windowed --add-data "hand_landmarker.task;." --add-data "gesture_recognizer.task;." --collect-all mediapipe main.py

## Troubleshooting
* Camera not found: Ensure no other application (like Zoom or Teams) is using the webcam.
* Performance: If lines are lagging, ensure the environment is well-lit to improve landmark detection accuracy.
* Executable: When running the .exe version, the application may take several seconds to initialize as it unpacks dependencies into a temporary directory.
