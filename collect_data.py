
import cv2
import mediapipe as mp
import csv
import os

from feature_extraction import extract_features, FEATURE_NAMES

# ---------------- SETTINGS ----------------

OUTPUT_FILE = "gesture_data.csv"
SAMPLES_PER_GESTURE = 500
SAMPLE_EVERY_N_FRAMES = 3

GESTURES = {
    "1": "Open Palm",
    "2": "Fist",
    "3": "Thumbs Up",
    "4": "Peace",
    "5": "OK Sign",
    "6": "Pointing"
}

# ---------------- MEDIAPIPE ----------------

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ---------------- CSV ----------------

file_exists = (
    os.path.exists(OUTPUT_FILE)
    and os.path.getsize(OUTPUT_FILE) > 0
)

csv_file = open(OUTPUT_FILE, "a", newline="")
writer = csv.writer(csv_file)

if not file_exists:
    writer.writerow(FEATURE_NAMES + ["gesture"])

# ---------------- CAMERA ----------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open camera.")
    csv_file.close()
    raise SystemExit

current_gesture = None
sample_count = 0
frame_count = 0

print("\nGESTURE DATA COLLECTION")
print("-----------------------")
for key, gesture in GESTURES.items():
    print(f"{key} - {gesture}")
print("Q - Quit")
print("\nPress a number to collect a gesture.")

# ---------------- MAIN LOOP ----------------

try:
    while True:
        success, frame = cap.read()

        if not success:
            print("Cannot read camera frame.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        frame_count += 1

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                if (
                    current_gesture is not None
                    and frame_count % SAMPLE_EVERY_N_FRAMES == 0
                    and sample_count < SAMPLES_PER_GESTURE
                ):
                    features = extract_features(
                        hand_landmarks.landmark
                    )

                    writer.writerow(features + [current_gesture])
                    sample_count += 1

                    if sample_count >= SAMPLES_PER_GESTURE:
                        break

        # ---------------- DISPLAY ----------------

        if current_gesture is None:
            status = "Press 1-6 to select gesture"
            color = (0, 255, 255)
        else:
            status = f"Collecting: {current_gesture}"
            color = (0, 255, 0)

        cv2.putText(
            frame, status, (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
        )

        cv2.putText(
            frame,
            f"Samples: {sample_count}/{SAMPLES_PER_GESTURE}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (255, 255, 255), 2
        )

        cv2.putText(
            frame,
            "Show either or both hands | Q: Quit",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (255, 255, 255), 2
        )

        cv2.imshow("Gesture Data Collection", frame)

        # ---------------- KEYBOARD ----------------

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if chr(key) in GESTURES:
            current_gesture = GESTURES[chr(key)]
            sample_count = 0
            frame_count = 0

            print(f"\nCollecting: {current_gesture}")

        if sample_count >= SAMPLES_PER_GESTURE:
            print(
                f"Completed {current_gesture}: "
                f"{sample_count} samples"
            )
            current_gesture = None
            sample_count = 0

finally:
    csv_file.close()
    cap.release()
    hands.close()
    cv2.destroyAllWindows()

print("\nData collection finished.")
print(f"Dataset saved to: {OUTPUT_FILE}")