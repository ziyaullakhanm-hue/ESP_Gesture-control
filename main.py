import cv2
import mediapipe as mp
import pyautogui
import math

# Disable PyAutoGUI fail-safe for testing
pyautogui.FAILSAFE = False

# Initialize Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# OpenCV video capture
cap = cv2.VideoCapture(0)

screen_width, screen_height = pyautogui.size()

# Function to calculate distance between two landmarks
def distance(point1, point2):
    return math.hypot(point2.x - point1.x, point2.y - point1.y)

# Neutral position
neutral_x, neutral_y = None, None

# Click debounce variables
left_click_done = False
right_click_done = False

with mp_hands.Hands(min_detection_confidence=0.7,
                    min_tracking_confidence=0.7,
                    max_num_hands=1) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process hands
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                # Fingertips
                index_tip = hand_landmarks.landmark[8]
                middle_tip = hand_landmarks.landmark[12]
                ring_tip = hand_landmarks.landmark[16]
                pinky_tip = hand_landmarks.landmark[20]
                thumb_tip = hand_landmarks.landmark[4]

                # Finger base (knuckles)
                index_mcp = hand_landmarks.landmark[5]
                middle_mcp = hand_landmarks.landmark[9]
                ring_mcp = hand_landmarks.landmark[13]
                pinky_mcp = hand_landmarks.landmark[17]

                # Gaps for clicks
                gap_index_thumb = distance(index_tip, thumb_tip)
                gap_middle_thumb = distance(middle_tip, thumb_tip)

                # Check if all fingers are open (tip higher than knuckle in y)
                all_open = (index_tip.y < index_mcp.y and
                            middle_tip.y < middle_mcp.y and
                            ring_tip.y < ring_mcp.y and
                            pinky_tip.y < pinky_mcp.y)

                if not all_open:  # Only move cursor if not all fingers open
                    avg_x = (index_tip.x + middle_tip.x) / 2
                    avg_y = (index_tip.y + middle_tip.y) / 2

                    if neutral_x is None or neutral_y is None:
                        neutral_x, neutral_y = avg_x, avg_y

                    dx = (avg_x - neutral_x) * screen_width
                    dy = (avg_y - neutral_y) * screen_height

                    current_mouse_x, current_mouse_y = pyautogui.position()
                    pyautogui.moveTo(current_mouse_x + dx, current_mouse_y + dy, duration=0.05)

                # Left click — index + thumb touching
                if gap_index_thumb < 0.05 and not left_click_done:
                    pyautogui.click(button='left')
                    left_click_done = True
                elif gap_index_thumb >= 0.05:
                    left_click_done = False

                # Right click — middle + thumb touching
                if gap_middle_thumb < 0.05 and not right_click_done:
                    pyautogui.click(button='right')
                    right_click_done = True
                elif gap_middle_thumb >= 0.05:
                    right_click_done = False

                # Draw landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('Gesture Control', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

cap.release()
cv2.destroyAllWindows()
