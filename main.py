import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import screeninfo

pyautogui.FAILSAFE = False  # Disable safety for testing

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Get screen size
screen = screeninfo.get_monitors()[0]
screen_width, screen_height = screen.width, screen.height

# Camera setup
cap = cv2.VideoCapture(0)

# Tracking vars
neutral_x, neutral_y = None, None
smooth_factor = 5  # Higher = smoother but slower
move_scale = 2     # Higher = faster movement
dead_zone = 5      # Ignore tiny movements

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Index fingertip (landmark 8)
                index_x = int(hand_landmarks.landmark[8].x * frame.shape[1])
                index_y = int(hand_landmarks.landmark[8].y * frame.shape[0])

                # Calibration: set neutral position when first seen
                if neutral_x is None or neutral_y is None:
                    neutral_x, neutral_y = index_x, index_y

                # Calculate movement delta
                dx = index_x - neutral_x
                dy = index_y - neutral_y

                # Dead zone filtering
                if abs(dx) < dead_zone: dx = 0
                if abs(dy) < dead_zone: dy = 0

                # Smooth movement
                dx /= smooth_factor
                dy /= smooth_factor

                # Scale movement
                dx *= move_scale
                dy *= move_scale

                # Get current cursor pos & move
                curr_x, curr_y = pyautogui.position()
                pyautogui.moveTo(curr_x + dx, curr_y + dy)

                # Draw hand landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Index Finger Control", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

cap.release()
cv2.destroyAllWindows()
