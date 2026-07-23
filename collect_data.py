import cv2
import mediapipe as mp
import numpy as np
import os
import time

# Settings
GESTURE_NAME = "EMERGENCY"    # Change this for each gesture
SAMPLES_TO_COLLECT = 200  # 200 samples per gesture
FRAMES_PER_SAMPLE = 30    # 30 frames per sample
DATA_DIR = "dataset"      # Folder to save data     # python collect_data.py

# Create folder
os.makedirs(f"{DATA_DIR}/{GESTURE_NAME}", exist_ok=True)

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)  
#Minimum confidence required for initial hand detection.

cap = cv2.VideoCapture(0)  #0 - Default Camera

print(f"Collecting data for: {GESTURE_NAME}")
print("RIGHT HAND ONLY")
print("----------------------------")
print("0-50   samples: Normal position")
print("50-100 samples: Sit closer to camera")
print("100-150 samples: Sit further from camera")
print("150-200 samples: Slight angle variation")
print("----------------------------")
print("Press SPACE to collect a sample")
print("Press Q to quit")

sample_count = 0

while sample_count < SAMPLES_TO_COLLECT:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  #BGR_to_RGB
    result = hands.process(rgb_frame)

    # Show which variation we are in
    if sample_count < 50:
        variation = "Variation 1: Normal position"
    elif sample_count < 100:
        variation = "Variation 2: Move CLOSER to camera"
    elif sample_count < 150:
        variation = "Variation 3: Move FURTHER from camera"
    else:
        variation = "Variation 4: Slight angle change"

    # Show status on screen
    cv2.putText(frame, f"Gesture: {GESTURE_NAME}", (10, 30),  
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2) #(image, text, position, font, size, color, thickness)
    cv2.putText(frame, f"Samples: {sample_count}/{SAMPLES_TO_COLLECT}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, variation, (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, "RIGHT HAND ONLY - Press SPACE", (10, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Alert when variation changes
    if sample_count in [50, 100, 150]:
        cv2.putText(frame, "CHANGE POSITION NOW!", (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Data Collection", frame)

    key = cv2.waitKey(1) & 0xFF  # Waits 1 millisecond for key press.

    if key == ord(' '):  # if key is " " (Spacebar)  #"ord" converts space character into ASCII number.
        if result.multi_hand_landmarks: #if hand is detected
            sequence = []

            for frame_num in range(FRAMES_PER_SAMPLE):
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb_frame)

                if result.multi_hand_landmarks:
                    landmarks = []
                    for lm in result.multi_hand_landmarks[0].landmark:  #21 hand points
                        landmarks.append([lm.x, lm.y, lm.z])       #Loop runs 21 times
                    sequence.append(np.array(landmarks).flatten())
                else:
                    sequence.append(np.zeros(63))

                cv2.putText(frame, f"RECORDING... {frame_num+1}/30", (10, 180), #frame_num+1 bcz loop starts from 0 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Data Collection", frame)
                cv2.waitKey(1)

            sequence = np.array(sequence) # Convert sequence into NumPy array
            save_path = f"{DATA_DIR}/{GESTURE_NAME}/sample_{sample_count}.npy"
            np.save(save_path, sequence) # saves the NumPy array into a .npy file
            sample_count += 1
            print(f"Sample {sample_count} saved - {variation}")

        else:
            print("No hand detected. Show your right hand first.")

    elif key == ord('q'): 
        break    #If q pressed then exit 

cap.release()  #Releases webcam access.
cv2.destroyAllWindows()  #Closes all OpenCV windows.
print(f"Done! Collected {sample_count} samples for {GESTURE_NAME}")





""" 
dataset/
│
├── hello/
│   ├── sample_0.npy
│   ├── sample_1.npy
│   ├── sample_2.npy
│
├── thanks/
│   ├── sample_0.npy
│   ├── sample_1.npy

"""