import cv2
import mediapipe as mp

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Open Webcam
cap = cv2.VideoCapture(0)

print("Camera started. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not accessible")
        break

    # Flip frame (mirror effect)
    frame = cv2.flip(frame, 1)

    # Convert to RGB (MediaPipe needs RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame
    result = hands.process(rgb_frame)

    # If hand detected
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks on screen
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            print("Hand detected!")

    # Show the frame
    cv2.imshow("Gesture2Voice - Hand Detection", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()