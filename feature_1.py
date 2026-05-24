import time 
import numpy as np
import math
import cv2

def process_gestures(landmarks, canvas, frame):
  """
  
  Analyzes hand landmarks to trigger special actions.
  Returns: Updated ad a status message.
  """
  fingers = []
  if landmarks[4].x < landmarks[3].x:
    fingers.append(1)
  else:
    fingers.append(0)
    
  for tip, pip in [(8,6), (12,10), (16,14), (20,18)]:
    if landmarks[tip].y < landmarks[pip].y:
      fingers.append(1)
    else:
      fingers.append(0)
      
    msg = ""
    
    if fingers == [0,1,1,0,0]:
      timestamp = int(time.time())
      cv2.imwrite(f"drawing_{timestamp}.png", frame)
      msg = "SCREENSHOT SAVED"
      
    elif sum(fingers) == 5:
      canvas = np.zeros_like(canvas)
      msg = "Canvas Cleared"
  return canvas, msg
  
