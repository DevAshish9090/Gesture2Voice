import cv2
import mediapipe as mp
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

DATASET_DIR = "fingerspell_dataset"

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.2)

print("Extracting landmarks...")

X, y = [], []

for label in sorted(os.listdir(DATASET_DIR)):
    label_path = os.path.join(DATASET_DIR, label)
    if not os.path.isdir(label_path):
        continue

    images = [f for f in os.listdir(label_path) if f.endswith('.jpg')]
    count  = 0

    for img_file in images:
        img = cv2.imread(os.path.join(label_path, img_file))
        if img is None:
            continue

        result = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not result.multi_hand_landmarks:
            continue

        lm = result.multi_hand_landmarks[0].landmark
        wx, wy, wz = lm[0].x, lm[0].y, lm[0].z
        row = []
        for pt in lm:
            row.extend([pt.x - wx, pt.y - wy, pt.z - wz])

        X.append(row)
        y.append(label)
        count += 1

    print(f"  [{label}]  {count}/{len(images)} extracted")

hands.close()

print(f"\nTotal: {len(X)} samples across {len(set(y))} classes")

print("\nTraining model...")
X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc    = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {acc * 100:.1f}%")
print(classification_report(y_test, y_pred))

print("\nWebcam test — show any gesture. Press Q to quit.")

hands_live = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame  = cv2.flip(frame, 1)
    result = hands_live.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    label, conf = "No hand", 0.0

    if result.multi_hand_landmarks:
        mp_draw.draw_landmarks(frame, result.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

        lm = result.multi_hand_landmarks[0].landmark
        wx, wy, wz = lm[0].x, lm[0].y, lm[0].z
        row = []
        for pt in lm:
            row.extend([pt.x - wx, pt.y - wy, pt.z - wz])

        proba = model.predict_proba([row])[0]
        idx   = proba.argmax()
        label = model.classes_[idx]
        conf  = proba[idx] * 100

    cv2.putText(frame, f"Letter: {label}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (80, 255, 80), 3)
    cv2.putText(frame, f"Confidence: {conf:.1f}%", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 220, 80), 2)
    cv2.putText(frame, "Q to quit", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("Fingerspell Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
hands_live.close()
cv2.destroyAllWindows()
