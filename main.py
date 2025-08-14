import cv2
import mediapipe as mp
import pyautogui
import math
import time
from screeninfo import get_monitors
 
# ---------------- One Euro Filter ----------------
class OneEuroFilter:
    def __init__(self, freq=120, mincutoff=1.0, beta=0.0, dcutoff=1.0):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.last_time = None
        self.x_prev = None
        self.dx_prev = None
 
    def alpha(self, cutoff):
        tau = 1.0 / (2 * math.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)
 
    def filter(self, x, x_prev, alpha):
        return alpha * x + (1 - alpha) * x_prev
 
    def __call__(self, x):
        now = time.time()
        if self.last_time is None:
            self.last_time = now
            self.x_prev = x
            self.dx_prev = 0
            return x
 
        self.freq = 1.0 / (now - self.last_time)
        self.last_time = now
        dx = (x - self.x_prev) * self.freq
        dx_hat = self.filter(dx, self.dx_prev, self.alpha(self.dcutoff))
        cutoff = self.mincutoff + self.beta * abs(dx_hat)
        x_hat = self.filter(x, self.x_prev, self.alpha(cutoff))
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat
 
# ---------------- Setup ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
 
# Screen size
monitor = get_monitors()[0]
screen_width, screen_height = monitor.width, monitor.height
 
# Filters
filter_x = OneEuroFilter()
filter_y = OneEuroFilter()
 
# PyAutoGUI failsafe
pyautogui.FAILSAFE = False
 
# Gesture states
dragging = False
tracking_enabled = True
 
# ---------------- Helper functions ----------------
def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])
 
# ---------------- Main loop ----------------
cap = cv2.VideoCapture(0)
 
while True:
    ret, frame = cap.read()
    if not ret:
        break
 
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
 
    if results.multi_hand_landmarks and tracking_enabled:
        for hand_landmarks in results.multi_hand_landmarks:
            lm = hand_landmarks.landmark
 
            # Index fingertip
            x = int(lm[8].x * screen_width)
            y = int(lm[8].y * screen_height)
 
            # Apply smoothing
            smooth_x = int(filter_x(x))
            smooth_y = int(filter_y(y))
 
            # Detect distances for gestures
            index_tip = (lm[8].x, lm[8].y)
            thumb_tip = (lm[4].x, lm[4].y)
            middle_tip = (lm[12].x, lm[12].y)
 
            # Distances
            dist_index_thumb = distance(index_tip, thumb_tip)
            dist_middle_thumb = distance(middle_tip, thumb_tip)
            dist_index_middle = distance(index_tip, middle_tip)
 
            # Gestures
            if dist_index_thumb < 0.04:  # Left click
                pyautogui.click()
                print("Left Click")
                time.sleep(0.2)
 
            elif dist_middle_thumb < 0.04:  # Right click
                pyautogui.rightClick()
                print("Right Click")
                time.sleep(0.2)
 
            elif dist_index_middle < 0.05 and not dragging:  # Start drag
                pyautogui.mouseDown()
                dragging = True
                print("Drag Start")
 
            elif dragging and dist_index_middle > 0.06:  # End drag
                pyautogui.mouseUp()
                dragging = False
                print("Drag End")
 
            elif lm[4].y < lm[3].y and lm[8].y > lm[6].y:  # Scroll up
                pyautogui.scroll(50)
                print("Scroll Up")
 
            elif lm[4].y > lm[3].y and lm[8].y < lm[6].y:  # Scroll down
                pyautogui.scroll(-50)
                print("Scroll Down")
 
            # Fist detection to stop tracking
            fingers_folded = all(lm[i].y > lm[i - 2].y for i in [8, 12, 16, 20])
            if fingers_folded:
                tracking_enabled = False
                print("Tracking stopped (Fist detected)")
 
            # Move mouse
            pyautogui.moveTo(smooth_x, smooth_y)
 
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
 
    cv2.imshow("Hand Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break
 
cap.release()
cv2.destroyAllWindows()