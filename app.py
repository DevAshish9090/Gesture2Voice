import cv2
import mediapipe as mp
import numpy as np
import pickle
import warnings
import time
import threading
import os
import collections
from flask import Flask, render_template, Response
from flask_socketio import SocketIO
from sentence_speech import build_sentence, speak


warnings.filterwarnings('ignore')

app = Flask(__name__)  #creates Flask web application
app.config['SECRET_KEY'] = 'gesture2voice'
#app.config['NAME'] = 'Ashish'  #new 
#print(app.config) #new
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')#This creates a Socket.IO server attached to Flask.
mp_draw = mp.solutions.drawing_utils

# ── Load main gesture model ───────────────────────────────────────────────────
with open('gesture_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('gesture_labels.pkl', 'rb') as f:
    labels = pickle.load(f)

# ── Load fingerspell model ────────────────────────────────────────────────────
try:
    with open('fingerspell_model.pkl', 'rb') as f: # rb = read in binary 
        fs_model = pickle.load(f)
    with open('fingerspell_labels.pkl', 'rb') as f:
        fs_labels = pickle.load(f)
    FS_MODEL_LOADED = True   # Model is loaded 
    print(f"[Fingerspell] ✅ Model loaded — {len(fs_labels)} classes")
except FileNotFoundError:
    FS_MODEL_LOADED = False
    fs_model  = None
    fs_labels = []
    print("[Fingerspell] ⚠️  fingerspell_model.pkl not found")

# ── MediaPipe ─────────────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
hands    = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,  # at least 60% sure it sees a hand before it starts tracking it
    min_tracking_confidence=0.5,
    model_complexity=0
)

# ── Main app settings ─────────────────────────────────────────────────────────
SEQUENCE_LENGTH      = 30
STABILITY_THRESHOLD  = 4
CONFIDENCE_THRESHOLD = 0.7
RESET_TIMEOUT        = 30
EMERGENCY_PHRASE     = 'I need help immediately!'
EMERGENCY_REPEAT     = 0

# ── Fingerspell settings ──────────────────────────────────────────────────────
FS_STABILITY  = 3
FS_CONFIDENCE = 0.35
FS_COOLDOWN   = 20

FS_ACTION_DELETE  = 'DEL'
FS_ACTION_SPACE   = 'SPACE'
FS_ACTION_NOTHING = 'NOTHING'

# ── Shared state ──────────────────────────────────────────────────────────────
sequence          = []
prediction_buffer = []
sentence          = []
last_word_time    = time.time()
is_speaking       = False
selected_voice    = 'en'

# ONE shared frame used by BOTH /video_feed and /fingerspell_feed
latest_frame     = None
latest_raw_frame = None          # raw BGR frame for fingerspell detection
lock             = threading.Lock()

# Emergency state
emergency_active     = False
emergency_stop_event = threading.Event()  #an Event object that acts like a switch.

# Fingerspell state
fs_state = {
    'letter_history': collections.deque(maxlen=8),
    'current_word':   [],
    'sentence_words': [],
    'cooldown':       0,
    'last_action':    None,
}
fs_lock = threading.Lock()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def emergency_loop():
    while not emergency_stop_event.is_set():  #While True:
        try:
            speak(EMERGENCY_PHRASE, selected_voice)
        except Exception as e:
            print(f"[speak error] {e}")
        for _ in range(EMERGENCY_REPEAT * 2):
            if emergency_stop_event.is_set(): #False
                break
            time.sleep(0.5)

def start_emergency(): #trigger that kicks off your emergency
    global emergency_active #use the variable that already exists outside the function
    if emergency_active:
        return  #emergency already active , stop this fn 
    emergency_active = True  #else  = True 
    emergency_stop_event.clear()
    socketio.emit('emergency_triggered', {}) #Send data/event from backend to frontend instantly
    threading.Thread(target=emergency_loop, daemon=True).start()

def stop_emergency():
    global emergency_active #use the variable that already exists outside the function
    emergency_active = False    

    emergency_stop_event.set()

def normalize_landmarks(seq):
    normalized = []
    for frame in seq:       # lm[0] is the first landmark (wrist) 
        lm = frame.reshape(21, 3)   #takes those 63 numbers and reshapes them into a grid of 21 rows and 3 columns.
        normalized.append((lm - lm[0]).flatten()) #every point is measured relative to wrist.
    return np.array(normalized)  #The model expects a flat input.


def camera_loop():
    """
    Single camera loop — captures frames, runs main gesture detection,
    stores latest_frame (JPEG) and latest_raw_frame (BGR) for sharing.
    """
    global sequence, prediction_buffer, sentence, last_word_time
    global latest_frame, latest_raw_frame

    cap = cv2.VideoCapture(0)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue #skip the rest of the current iteration and move to the next iteration immediately

        frame_count += 1
        frame = cv2.flip(frame, 1)

        small  = cv2.resize(frame, (320, 240))
        rgb    = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)  #Mediapipe runs
        #Checks if the sentence contains words and no new word is detected for longer than RESET_TIMEOUT seconds.
        if len(sentence) > 0 and (time.time() - last_word_time) > RESET_TIMEOUT:   #time.time() gives the current timestamp in seconds
            sentence = []
            prediction_buffer = []
            socketio.emit('sentence_update', {'sentence': ''}) # tells the browser to clear its display.

        detected_word = ''
        confidence    = 0.0

        if result.multi_hand_landmarks:
            for hand_lm in result.multi_hand_landmarks: #loops through detected hands
                mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS) #hand_lm - contains landmark coordinates 

                lm_list   = [[lm.x, lm.y, lm.z] for lm in hand_lm.landmark]
                landmarks = np.array(lm_list).flatten()
                sequence.append(landmarks) #adds this frames 63 numbers to the sequence list
                if len(sequence) > SEQUENCE_LENGTH:
                    sequence.pop(0)

                if len(sequence) == SEQUENCE_LENGTH and frame_count % 3 == 0: #only run prediction when you have exactly 30 frames,  gestures don't change that fast so we  predict on every 3rd frame using frame_count % 3 == 0
                    norm       = normalize_landmarks(np.array(sequence))  # shape is (30, 63) — a grid of 30 rows and 63 columns
                    inp        = norm.flatten().reshape(1, -1) #reshapes into a 2D array with 1 row
                    pred       = model.predict(inp)[0] # feeds input to random forest
                    confidence = float(model.predict_proba(inp).max())
                    detected_word = str(labels[pred])
                    # stability + confidence gate
                    if confidence >= CONFIDENCE_THRESHOLD and not emergency_active:
                        prediction_buffer.append(detected_word) # short memory of the last 4 predictions
                        if len(prediction_buffer) > STABILITY_THRESHOLD:
                            prediction_buffer.pop(0)
                        if prediction_buffer.count(detected_word) == STABILITY_THRESHOLD: #STABILITY_THRESHOLD is set to 4
                            if len(sentence) == 0 or sentence[-1] != detected_word:  #Even if the gesture is stable, one more check happens.

                                sentence.append(detected_word)  # word is added to the sentence list permanently.
                                last_word_time = time.time() # resets the 30-second inactivity timer
                                socketio.emit('word_added', {    
                                    'word':     detected_word,
                                    'sentence': build_sentence(sentence)
                                })  # instantly pushes to the browser

                cv2.putText(frame, detected_word, (10, 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 255, 80), 2)
                cv2.putText(frame, f"{int(confidence*100)}%", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 220, 80), 2)
        else:  # No Hand Detected
            sequence = []

        if frame_count % 3 == 0:
            socketio.emit('prediction_update', {
                'word':       detected_word,
                'confidence': round(confidence, 2),
                'no_hand':    result.multi_hand_landmarks is None
            })

        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])  #this always runs # compresses the frame into JPEG first.
        with lock:
            latest_frame     = buf.tobytes() #pin the compressed JPEG for the browser
            latest_raw_frame = small.copy()   # store small 320x240 for fingerspell

    cap.release()


def generate_frames():
    """MJPEG stream for /video_feed — uses shared latest_frame."""
    while True:
        with lock:
            if latest_frame is None:
                time.sleep(0.02)
                continue #If no frame available,skip the rest of this loop and try again.
            frame = latest_frame
        yield (b'--frame\r\n'   #try using return later   
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # sends one webcam image frame to the browser during live video streaming
        time.sleep(0.02)


def generate_fingerspell_frames():
    """
    MJPEG stream for /fingerspell_feed.
    Reads the SAME latest_frame as the main feed — no second camera opened.
    """
    while True:
        with lock:
            if latest_frame is None:
                time.sleep(0.02)
                continue
            frame = latest_frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.02)


# ══════════════════════════════════════════════════════════════════════════════
# FINGERSPELL DETECTION — runs in its own thread, reads shared raw frame
# ══════════════════════════════════════════════════════════════════════════════
# This entire section has 3 functions that work together:

def _fs_predict(landmarks):
    if not FS_MODEL_LOADED: # If the fingerspell model failed to load at startup, immediately return nothing
        return None, 0.0
    wx, wy, wz = landmarks[0].x, landmarks[0].y, landmarks[0].z  # landmarks[0] is the wrist 
    row = []
    for lm in landmarks:
        row.extend([lm.x - wx, lm.y - wy, lm.z - wz]) #subtracts wrist position from every landmark. (NORMALIZATION)
    proba   = fs_model.predict_proba([row])[0] #probability
    top_idx = proba.argmax() # finds highest probability index.
    return fs_labels[top_idx], float(proba[top_idx])


def _fs_handle_confirmed(label): #label -> predicted letter
    with fs_lock:
        if fs_state['cooldown'] > 0: #prevents repeated spam. #would happen instantly because webcam sees same hand repeatedly.
            return None
        if label == fs_state['last_action']: #fs_state - idx of label
            if label in (FS_ACTION_SPACE, FS_ACTION_NOTHING):  #Another protection layer for space and nothing
                return None

        if label == FS_ACTION_NOTHING:
            fs_state['last_action'] = label
            return {'action': 'nothing'}

        if label == FS_ACTION_DELETE:
            if fs_state['current_word']: # checks the word isn't already empty
                fs_state['current_word'].pop()
            fs_state['cooldown']    = FS_COOLDOWN #sets cooldown to 20 frames
            fs_state['last_action'] = label
            fs_state['letter_history'].clear()
            return {'action': 'del', 'word': ''.join(fs_state['current_word'])} # joins the remaining letters into a string

        if label == FS_ACTION_SPACE:
            word = ''.join(fs_state['current_word'])
            if word:
                fs_state['sentence_words'].append(word)
                fs_state['current_word'] = [] #clears the letter buffer. Ready to start the next word:
            fs_state['cooldown']    = FS_COOLDOWN * 2
            fs_state['last_action'] = label
            fs_state['letter_history'].clear()
            return {
                'action':   's  fg  e4refdffpace',
                'word':     word,
                'sentence': ' '.join(fs_state['sentence_words']),
            }

        # Letter A–Z
        fs_state['current_word'].append(label)
        fs_state['cooldown']    = FS_COOLDOWN
        fs_state['last_action'] = label
        fs_state['letter_history'].clear()
        return {
            'action': 'letter',
            'letter': label,
            'word':   ''.join(fs_state['current_word']),
        }


def fingerspell_detection_loop():
    # runs forever in its own thread handling gesture recognition
    #just reads what camera_loop already captured
    """
    Separate thread that reads shared raw frames and runs
    fingerspell prediction — no camera opened here.
    """
    fs_hands = mp.solutions.hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    )

    frame_idx = 0 #its own frame counter

    while True:
        time.sleep(0.04)   # ~25 fps detection rate

        with lock: #prevents two threads from accessing/modifying shared data at the same time.
            raw = latest_raw_frame
        if raw is None:
            continue

        frame_idx += 1

        # Decrement cooldown
        with fs_lock:
            if fs_state['cooldown'] > 0:
                fs_state['cooldown'] -= 1

        rgb    = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        result = fs_hands.process(rgb)

        if result.multi_hand_landmarks:
            hand_lm = result.multi_hand_landmarks[0]
            label, conf = _fs_predict(hand_lm.landmark)

            if label and conf >= FS_CONFIDENCE:
                with fs_lock:
                    fs_state['letter_history'].append(label)
                    history = list(fs_state['letter_history'])

                stable_count = sum(1 for x in history if x == label)
                is_stable    = (len(history) >= FS_STABILITY and
                                len(set(history[-FS_STABILITY:])) == 1)

                socketio.emit('fs_prediction', {
                    'label':        label,
                    'confidence':   round(conf * 100, 1),
                    'stable_count': stable_count,
                    'is_stable':    is_stable,
                })  #Send live prediction to frontend

                if is_stable: # If prediction stable
                    result_data = _fs_handle_confirmed(label) #adds letters, delete etc
                    if result_data:
                        socketio.emit('fs_confirmed', result_data) #Send confirmed result to browser
            else: #Low confidence
                with fs_lock: 
                    fs_state['letter_history'].clear()
                socketio.emit('fs_prediction',
                              {'label': None, 'confidence': 0,
                               'stable_count': 0, 'is_stable': False})
        else: #No hand detected
            with fs_lock:
                fs_state['letter_history'].clear()
            socketio.emit('fs_prediction',
                          {'label': None, 'confidence': 0,
                           'stable_count': 0, 'is_stable': False})


# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/') # @ - decorator 
def landing():
    return render_template('landing.html')  #RESPONSE
    

@app.route('/app')
def index():
    return render_template('index.html')#RESPONSE

@app.route('/video_feed') #this sends a continuous stream of JPEG images
def video_feed(): 
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/fingerspell')
def fingerspell():
    return render_template('fingerspell.html')

@app.route('/fingerspell_feed')
def fingerspell_feed():
    return Response(generate_fingerspell_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ══════════════════════════════════════════════════════════════════════════════
# SOCKETIO EVENTS — original
# ══════════════════════════════════════════════════════════════════════════════
#Triggered when: user clicks the Speak button on the browser → app.js emits 'speak'.
@socketio.on('speak')
def handle_speak(data):
    global is_speaking
    if len(sentence) > 0 and not is_speaking:
        fs = build_sentence(sentence)
        is_speaking = True
        def _t(): #reset speaking status afterward
            global is_speaking
            speak(fs, selected_voice)
            is_speaking = False
        threading.Thread(target=_t).start()

@socketio.on('reset')
def handle_reset():
    global sentence, prediction_buffer
    sentence = [] #reset sentences 
    prediction_buffer = [] #reset words
    socketio.emit('sentence_update', {'sentence': ''}) # tells browser to clear its display.

@socketio.on('set_voice') #user picks an accent from the dropdown
def handle_voice(data):
    global selected_voice # updates the global. Next time speak() is called it uses this new voice.
    selected_voice = data['voice']

@socketio.on('emergency_speak')
def handle_emergency_speak():
    start_emergency()

@socketio.on('dismiss_emergency')
def handle_dismiss():
    stop_emergency()
    socketio.emit('emergency_dismissed', {})

@socketio.on('speak_text')
def handle_speak_text(data):
    threading.Thread(target=speak, args=(data['text'], selected_voice)).start()


# ══════════════════════════════════════════════════════════════════════════════
# SOCKETIO EVENTS — fingerspell
# ══════════════════════════════════════════════════════════════════════════════

@socketio.on('fs_backspace')
def fs_backspace():
    with fs_lock:
        if fs_state['current_word']:
            fs_state['current_word'].pop()
        socketio.emit('fs_confirmed', {
            'action': 'del',
            'word':   ''.join(fs_state['current_word']),
        })

@socketio.on('fs_add_space')
def fs_add_space():
    with fs_lock:
        word = ''.join(fs_state['current_word'])
        if word:
            fs_state['sentence_words'].append(word)
            fs_state['current_word'] = []
        socketio.emit('fs_confirmed', {
            'action':   'space',
            'word':     word,
            'sentence': ' '.join(fs_state['sentence_words']),
        })

@socketio.on('fs_clear_word')
def fs_clear_word():
    with fs_lock:
        fs_state['current_word'] = []
        socketio.emit('fs_confirmed', {'action': 'del', 'word': ''})

@socketio.on('fs_clear_all')
def fs_clear_all():
    with fs_lock:
        fs_state['current_word']   = []
        fs_state['sentence_words'] = []
        fs_state['letter_history'].clear()
        fs_state['last_action']    = None
        socketio.emit('fs_cleared', {})

@socketio.on('fs_undo_word')
def fs_undo_word():
    with fs_lock:
        if fs_state['sentence_words']:
            fs_state['sentence_words'].pop()
        socketio.emit('fs_confirmed', {
            'action':   'space',
            'word':     '',
            'sentence': ' '.join(fs_state['sentence_words']),
        })

@socketio.on('fs_speak') 
def fs_speak(data=None):
    with fs_lock:
        words   = list(fs_state['sentence_words'])
        current = ''.join(fs_state['current_word'])
        if current:
            words.append(current)
    if not words:
        return
    sentence_str = ' '.join(words).capitalize() + '.'
    voice        = (data or {}).get('voice', 'en-us')
    threading.Thread(target=speak, args=(sentence_str, voice), daemon=True).start()
    socketio.emit('fs_speaking', {'sentence': sentence_str})


# ══════════════════════════════════════════════════════════════════════════════
# STARTUP
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    threading.Thread(target=camera_loop, daemon=True).start()
    threading.Thread(target=fingerspell_detection_loop, daemon=True).start()
    socketio.run(app, debug=False)