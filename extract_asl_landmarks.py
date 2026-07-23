"""
extract_asl_landmarks.py
────────────────────────
Reads images from the Kaggle ASL Alphabet dataset, runs MediaPipe Hands
on each image, extracts 63 normalised landmark values (21 pts × x,y,z),
and saves everything to fingerspell_landmarks.csv.

Dataset: https://www.kaggle.com/datasets/grassknoted/asl-alphabet
Download and unzip so you have:  asl_alphabet_train/A/*.jpg  B/*.jpg … Z/*.jpg
Then run:  python extract_asl_landmarks.py
"""

import os
import cv2
import mediapipe as mp
import csv
import numpy as np
from pathlib import Path

# ── CONFIG ──────────────────────────────────────────────────────────────────
DATASET_PATH = "asl_alphabet_train"   # folder you get after unzipping Kaggle dataset
OUTPUT_CSV   = "fingerspell_landmarks.csv"
MAX_PER_CLASS = 1000                  # cap images per letter (keeps training fast)
VALID_LABELS  = [chr(i) for i in range(ord('A'), ord('Z') + 1)]  # A-Z only

# ── MEDIAPIPE SETUP ──────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,         # important! we're feeding still images
    max_num_hands=1,
    min_detection_confidence=0.4,   # lower threshold → more detections from dataset images
)

# ── EXTRACTION ───────────────────────────────────────────────────────────────
header = [f"{axis}{i}" for i in range(21) for axis in ["x", "y", "z"]] + ["label"]

total_rows = 0
skipped    = 0

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)

    for label_dir in sorted(os.listdir(DATASET_PATH)):
        label = label_dir.upper()
        if label not in VALID_LABELS:
            print(f"  Skipping class '{label_dir}' (not A-Z)")
            continue

        label_path = os.path.join(DATASET_PATH, label_dir)
        if not os.path.isdir(label_path):
            continue

        image_files = [f for f in os.listdir(label_path)
                       if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        count = 0
        for img_file in image_files:
            if count >= MAX_PER_CLASS:
                break

            img_path = os.path.join(label_path, img_file)
            img = cv2.imread(img_path)
            if img is None:
                skipped += 1
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result  = hands.process(img_rgb)

            if not result.multi_hand_landmarks:
                skipped += 1
                continue

            lm = result.multi_hand_landmarks[0].landmark

            # Normalise: subtract wrist (landmark 0) from all points
            wx, wy, wz = lm[0].x, lm[0].y, lm[0].z
            row = []
            for point in lm:
                row.extend([point.x - wx, point.y - wy, point.z - wz])
            row.append(label)

            writer.writerow(row)
            count += 1

        total_rows += count
        print(f"  [{label}]  {count} samples extracted")

hands.close()

print(f"\n✅  Done — {total_rows} rows saved to {OUTPUT_CSV}")
print(f"   Skipped (no hand detected / unreadable): {skipped}")
