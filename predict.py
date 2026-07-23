import cv2
import mediapipe as mp
import numpy as np
import pickle
import warnings
import time
import threading
from sentence_speech import build_sentence, speak

# Load model and labels
with open('gesture_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('gesture_labels.pkl', 'rb') as f:
    labels = pickle.load(f)

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Settings
SEQUENCE_LENGTH = 30
STABILITY_THRESHOLD = 8
CONFIDENCE_THRESHOLD = 0.5
RESET_TIMEOUT = 30

def normalize_landmarks(sequence):
    normalized = []
    for frame in sequence:
        landmarks = frame.reshape(21, 3)
        wrist = landmarks[0]
        landmarks = landmarks - wrist
        normalized.append(landmarks.flatten())
    return np.array(normalized)

# State
sequence = []
prediction_buffer = []
sentence = []
last_word_time = time.time()
is_speaking = False

cap = cv2.VideoCapture(0)
print("Gesture2Voice started.")
print("Press S to speak sentence")
print("Press R to reset sentence")
print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # Auto reset after 5 seconds of inactivity
    if len(sentence) > 0 and (time.time() - last_word_time) > RESET_TIMEOUT:
        sentence = []
        prediction_buffer = []
        print("Sentence auto-reset after inactivity")

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])
            landmarks = np.array(landmarks).flatten()
            sequence.append(landmarks)

            if len(sequence) > SEQUENCE_LENGTH:
                sequence.pop(0)

            if len(sequence) == SEQUENCE_LENGTH:
                normalized_seq = normalize_landmarks(np.array(sequence))
                input_data = normalized_seq.flatten().reshape(1, -1)

                prediction = model.predict(input_data)[0]
                confidence = model.predict_proba(input_data).max()

                if confidence >= CONFIDENCE_THRESHOLD:
                    predicted_word = labels[prediction]
                    prediction_buffer.append(predicted_word)

                    if len(prediction_buffer) > STABILITY_THRESHOLD:
                        prediction_buffer.pop(0)

                    if prediction_buffer.count(predicted_word) == STABILITY_THRESHOLD:
                        if len(sentence) == 0 or sentence[-1] != predicted_word:
                            sentence.append(predicted_word)
                            last_word_time = time.time()
                            print(f"Word added: {predicted_word}")

                    cv2.putText(frame, f"Detected: {predicted_word}", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                else:
                    cv2.putText(frame, "Low confidence", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "No hand detected", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        sequence = []

    # Show sentence on screen
    sentence_text = build_sentence(sentence)
    cv2.putText(frame, f"Sentence: {sentence_text}", (10, 380),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "S=Speak | R=Reset | Q=Quit", (10, 420),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("Gesture2Voice", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        if len(sentence) > 0 and not is_speaking:
            final_sentence = build_sentence(sentence)
            is_speaking = True
            def speak_thread():
                global is_speaking
                speak(final_sentence)
                is_speaking = False
            threading.Thread(target=speak_thread).start()
        else:
            print("Nothing to speak")

    elif key == ord('r'):
        sentence = []
        prediction_buffer = []
        print("Sentence reset")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()