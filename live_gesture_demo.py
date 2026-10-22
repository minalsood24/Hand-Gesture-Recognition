
import cv2
import mediapipe as mp
import numpy as np
import joblib

from collections import deque, Counter
from feature_extraction import extract_features

# ---------------- SETTINGS ----------------

MODEL_FILE = "gesture_model.pkl"
CAMERA_ID = 0

CONFIDENCE_THRESHOLD = 0.65
SMOOTHING_FRAMES = 7
MIN_VOTES = 4

# ---------------- LOAD MODEL ----------------

model = joblib.load(MODEL_FILE)
print("Model loaded successfully")
print("Gestures:", model.classes_)

# ---------------- MEDIAPIPE ----------------

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Keep separate prediction histories for both hands
histories = {
    "Left": deque(maxlen=SMOOTHING_FRAMES),
    "Right": deque(maxlen=SMOOTHING_FRAMES)
}

# ---------------- CAMERA ----------------

cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():
    print("ERROR: Cannot open camera.")
    raise SystemExit

print("Camera started. Press Q to quit.")

# ---------------- MAIN LOOP ----------------

while True:
    success, frame = cap.read()

    if not success:
        print("Cannot read camera frame.")
        break

    # Mirror the camera for a natural selfie view
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(
            result.multi_hand_landmarks
        ):
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Identify the hand
            hand_label = "Hand"

            if result.multi_handedness:
                hand_label = (
                    result.multi_handedness[i]
                    .classification[0].label
                )

            if hand_label not in histories:
                histories[hand_label] = deque(
                    maxlen=SMOOTHING_FRAMES
                )

            # Extract the same 8 features used in training
            features = extract_features(
                hand_landmarks.landmark
            )

            X = np.asarray(features, dtype=float).reshape(1, -1)

            # Predict gesture and confidence
            probabilities = model.predict_proba(X)[0]
            best_index = int(np.argmax(probabilities))

            gesture = str(model.classes_[best_index])
            confidence = float(probabilities[best_index])

            history = histories[hand_label]

            if confidence >= CONFIDENCE_THRESHOLD:
                history.append(gesture)
            else:
                history.clear()

            # Stabilize prediction using majority voting
            stable_gesture = "Uncertain"

            if len(history) >= MIN_VOTES:
                most_common, votes = Counter(history).most_common(1)[0]

                if votes >= MIN_VOTES:
                    stable_gesture = most_common

            # Position label near the wrist
            wrist = hand_landmarks.landmark[0]
            x = int(wrist.x * frame.shape[1])
            y = int(wrist.y * frame.shape[0])

            label = (
                f"{hand_label}: {stable_gesture} "
                f"({confidence * 100:.1f}%)"
            )

            color = (
                (0, 255, 0)
                if stable_gesture != "Uncertain"
                else (0, 165, 255)
            )

            cv2.putText(
                frame,
                label,
                (max(10, x - 80), max(30, y - 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2
            )

    else:
        cv2.putText(
            frame,
            "Show your hand to the camera",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

    cv2.putText(
        frame,
        "Two-hand gesture recognition | Q: Quit",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.imshow("Live Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ---------------- CLEANUP ----------------

cap.release()
hands.close()
cv2.destroyAllWindows()