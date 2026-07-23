import cv2
import mediapipe as mp
import os

GESTURE_NAME = "NOTHING"     # Change this for each letter/gesture
SAMPLES_TO_COLLECT = 200         # python collect_fingerspell_data.py
DATA_DIR = "fingerspell_dataset"

os.makedirs(f"{DATA_DIR}/{GESTURE_NAME}", exist_ok=True)

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)

print(f"Collecting data for: {GESTURE_NAME}")
print("RIGHT HAND ONLY")
print("----------------------------")
print("0-50   samples: Normal position")
print("50-100 samples: Move closer to camera")
print("100-150 samples: Move further from camera")
print("150-200 samples: Slight angle variation")
print("----------------------------")
print("Press SPACE to save image")
print("Press Q to quit")

sample_count = 0

while sample_count < SAMPLES_TO_COLLECT:
    ret, frame = cap.read()
    if not ret:
        break

    frame     = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result    = hands.process(rgb_frame)

    if sample_count < 50:
        variation = "Variation 1: Normal position"
    elif sample_count < 100:
        variation = "Variation 2: Move CLOSER to camera"
    elif sample_count < 150:
        variation = "Variation 3: Move FURTHER from camera"
    else:
        variation = "Variation 4: Slight angle change"

    # Draw landmarks on display frame only — NOT on saved image
    display_frame = frame.copy()

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.putText(display_frame, f"Gesture: {GESTURE_NAME}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(display_frame, f"Samples: {sample_count}/{SAMPLES_TO_COLLECT}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(display_frame, variation, (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(display_frame, "RIGHT HAND ONLY - Press SPACE", (10, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    if sample_count in [50, 100, 150]:
        cv2.putText(display_frame, "CHANGE POSITION NOW!", (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow("Fingerspell Collection", display_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        if result.multi_hand_landmarks:
            # Save clean frame — no drawings on it
            save_path = f"{DATA_DIR}/{GESTURE_NAME}/sample_{sample_count}.jpg"
            cv2.imwrite(save_path, frame)
            sample_count += 1
            print(f"Sample {sample_count} saved — {variation}")
        else:
            print("No hand detected. Show your right hand first.")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Done! Collected {sample_count} samples for {GESTURE_NAME}")
