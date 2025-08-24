import cv2
import mediapipe as mp
import numpy as np

# ——— Setup MediaPipe Hands ———
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# ——— Open webcam ———
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for mirror view (optional)
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert to RGB for MediaPipe
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    left_pt = None
    right_pt = None

    # ——— Find wrist of each hand and draw landmarks ———
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks,
                                                   results.multi_handedness):
            label = hand_handedness.classification[0].label  # 'Left' or 'Right'
            # Use landmark 0 (wrist); you can choose another landmark if you like
            lm = hand_landmarks.landmark[0]
            x_px, y_px = int(lm.x * w), int(lm.y * h)

            if label == 'Left':
                left_pt = (x_px, y_px)
            elif label == 'Right':
                right_pt = (x_px, y_px)

            # Draw hand skeleton
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # ——— If both hands visible, compute warp ———
    if left_pt and right_pt:
        x1, y1 = left_pt
        x2, y2 = right_pt

        # Define the four corners of the rectangle (axis-aligned)
        top_left     = (min(x1, x2), min(y1, y2))
        top_right    = (max(x1, x2), min(y1, y2))
        bottom_right = (max(x1, x2), max(y1, y2))
        bottom_left  = (min(x1, x2), max(y1, y2))

        src_pts = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
        dst_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

        # Compute perspective transform & apply
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(frame, M, (w, h))

        # Show warped output
        cv2.imshow('Warped View', warped)

        # Optional: draw the source quad on the original for feedback
        cv2.polylines(frame, [src_pts.astype(int)], isClosed=True, color=(0,255,0), thickness=2)

    # ——— Display original feed ———
    cv2.imshow('Original View', frame)

    # Quit on ‘q’
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
