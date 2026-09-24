"""
Smart Kitchen MERGED ALL FEATURES: YOLO + Gaze + Face Recognition + Emotion + Professional Gaze Mode
- One camera only
- YOLO object tracking sends OBJECT:... and OBJECT_ROT:...;ANGLE=... commands to C# on port 5000
- Gaze tracking sends GAZE_LEFT / GAZE_CENTER / GAZE_RIGHT + samples to C# on port 5001
- Face recognition + emotion sends profile/emotion to C# Bluetooth/Profile socket on port 5055
- Keeps gaze reports: hits, focus seconds, dominant AOI, confidence, ratios
- Sends live GAZE_SAMPLE data to C# so Marker 50 can reveal Eye Gaze Heatmap on the GUI

Install:
    py -3.10 -m pip install opencv-python mediapipe numpy ultralytics face-recognition deepface tf-keras

Files/folders:
    yolov8n.pt beside this file
    people/ folder beside this file, with images named as user names:
        people/Belal Hesham.jpg
        people/Ahmed Elsharkawy.png

Run order:
    1) Run the C# Smart Kitchen app first.
    2) Run this file:
       py -3.10 smart_kitchen_merged_all_features.py
    Optional gaze-only mode:
    py -3.10 smart_kitchen_merged_all_features.py --professional-gaze

Controls:
    Q = quit and save gaze reports
    A = auto gaze calibration sequence CENTER -> LEFT -> RIGHT -> UP -> DOWN (eyebrow + eye assisted)
    C = save current gaze as CENTER
    L = save current gaze as LEFT edge of GUI
    R = save current gaze as RIGHT edge of GUI
    U = save current gaze as TOP edge of GUI
    D = save current gaze as BOTTOM edge of GUI
    P = reload people folder
"""

import csv
import os
import re
import socket
import math
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except Exception as ex:
    face_recognition = None
    FACE_RECOGNITION_AVAILABLE = False
    print("[FACE WARNING] face_recognition not available:", ex)

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except Exception as ex:
    DeepFace = None
    DEEPFACE_AVAILABLE = False
    print("[EMOTION WARNING] deepface not available:", ex)


# =========================
# General camera settings
# =========================
CAMERA_INDEX = 0
FRAME_WIDTH = 960
FRAME_HEIGHT = 540
USE_DSHOW_ON_WINDOWS = True

# =========================
# Socket settings
# =========================
HOST = "127.0.0.1"

YOLO_PORT = 5000       # C# connects as client to receive OBJECT:...
GAZE_PORT = 5001       # C# connects as client to receive GAZE_...
PROFILE_PORT = 5055    # Python connects as client to C# profile/bluetooth server
SIGNUP_COMMAND_PORT = 5056  # C# sends marker-based signup commands to Python

SOCKET_WAIT_SECONDS = 2.0

# =========================
# YOLO settings
# =========================
MODEL_PATH = "yolov8n.pt"
CONF_THRESHOLD = 0.35
OBJECT_SEND_COOLDOWN = 2.0
OBJECT_ROT_SEND_COOLDOWN = 0.12

OBJECT_TO_COMMAND = {
    "spoon": "OBJECT:Spoon",
    "knife": "OBJECT:Knife",
    "cup": "OBJECT:Pot",
    "bowl": "OBJECT:Pot",
    "bottle": "OBJECT:Oil",
    "apple": "OBJECT:Tomato",
    "orange": "OBJECT:Tomato",
    "carrot": "OBJECT:Tomato",
    "broccoli": "OBJECT:Tomato",
    "pizza": "OBJECT:Rice",
    "sandwich": "OBJECT:Chicken",
}

# =========================
# Face + emotion settings
# =========================
PEOPLE_FOLDER = "people"
FACE_ROLES_FILE = "people_roles.txt"
FACE_PROCESS_EVERY_N_FRAMES = 12
EMOTION_PROCESS_EVERY_N_FRAMES = 24
FACE_SEND_COOLDOWN_SECONDS = 3.0
FACE_TOLERANCE = 0.50

known_face_encodings = []
known_face_names = []
known_face_roles = {}
last_detected_user = "Unknown"
last_emotion = "Neutral"
last_face_send_time = 0.0
signup_candidate_frame = None
signup_lock = threading.Lock()
signup_requested = False
signup_name = ""
signup_role = ""
signup_countdown_until = 0.0
signup_status = "Idle"
unknown_notice_sent = False

# =========================
# Gaze settings
# =========================
SMOOTHING_WINDOW = 2
STABLE_FRAMES_REQUIRED = 1
DWELL_SECONDS = 0.04
GAZE_SEND_COOLDOWN_SECONDS = 0.15
MIN_CONFIDENCE = 0.03
MIN_EYE_OPEN_SCORE = 0.080
SAMPLE_SEND_INTERVAL = 0.05
LEFT_TRIGGER_MARGIN = 0.035
RIGHT_TRIGGER_MARGIN = 0.050
AUTO_CAL_SECONDS_PER_POINT = 2.6
AUTO_CAL_START_DELAY_SECONDS = 1.5

LOG_FILE = "gaze_hits_log.csv"
SUMMARY_FILE = "gaze_summary_report.csv"

AOI_MAP = {
    "LEFT": "Left Screen Area",
    "CENTER": "Center Screen Area",
    "RIGHT": "Right Screen Area",
    "UP": "Top Screen Area",
    "DOWN": "Bottom Screen Area",
    "UP_LEFT": "Top Left Screen Area",
    "UP_RIGHT": "Top Right Screen Area",
    "DOWN_LEFT": "Bottom Left Screen Area",
    "DOWN_RIGHT": "Bottom Right Screen Area",
    "POINT": "Free Screen Point",
    "NONE": "No Face / No Gaze",
}

GAZE_COMMAND_MAP = {
    "LEFT": "GAZE_LEFT",
    "CENTER": "GAZE_CENTER",
    "RIGHT": "GAZE_RIGHT",
    "UP": "GAZE_UP",
    "DOWN": "GAZE_DOWN",
    "UP_LEFT": "GAZE_UP_LEFT",
    "UP_RIGHT": "GAZE_UP_RIGHT",
    "DOWN_LEFT": "GAZE_DOWN_LEFT",
    "DOWN_RIGHT": "GAZE_DOWN_RIGHT",
    "POINT": "GAZE_POINT",
}

# MediaPipe FaceMesh iris + eye indices
LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133
RIGHT_EYE_OUTER = 362
RIGHT_EYE_INNER = 263
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374
LEFT_IRIS = [468, 469, 470, 471]
RIGHT_IRIS = [473, 474, 475, 476]

LEFT_EYE_UPPER = [159, 160, 161, 158, 157, 173]
LEFT_EYE_LOWER = [145, 144, 163, 153, 154, 155]
RIGHT_EYE_UPPER = [386, 385, 384, 387, 388, 466]
RIGHT_EYE_LOWER = [374, 373, 390, 380, 381, 382]

HORIZONTAL_GAIN = 1.35
VERTICAL_GAIN = 1.45
FREE_SCREEN_X_GAIN = 1.18
FREE_SCREEN_Y_GAIN = 1.45

# Eyebrow landmarks help vertical gaze, especially looking up.
LEFT_BROW = [70, 63, 105, 66, 107]
RIGHT_BROW = [336, 296, 334, 293, 300]
FACE_TOP = 10
CHIN = 152

NOSE_TIP = 1
FACE_LEFT = 234
FACE_RIGHT = 454

mp_face_mesh = mp.solutions.face_mesh


# =========================
# Shared helpers
# =========================
def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def open_camera():

    backends = [
        (0, cv2.CAP_MSMF),
        (0, cv2.CAP_DSHOW),
        (0, cv2.CAP_ANY),
        (1, cv2.CAP_MSMF),
        (1, cv2.CAP_DSHOW),
    ]

    cap = None

    for index, backend in backends:
        print(f"[CAMERA] Trying index {index} backend {backend}")

        test = cv2.VideoCapture(index, backend)

        if test.isOpened():
            print(f"[CAMERA] Opened camera {index}")
            cap = test
            break

        test.release()

    if cap is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    return cap


# =========================
# Socket server for YOLO/Gaze
# =========================
class OptionalSocketServer:
    def __init__(self, name: str, host: str, port: int, wait_seconds: float = 2.0, read_context: bool = False):
        self.name = name
        self.host = host
        self.port = port
        self.wait_seconds = wait_seconds
        self.read_context = read_context
        self.server = None
        self.client = None
        self.recv_buffer = ""
        self.user_name = "GUI User"
        self.device_name = "Unknown Device"
        self.login_status = "0"

    def start(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((self.host, self.port))
        self.server.listen(1)
        self.server.settimeout(self.wait_seconds)

        print(f"[{self.name}] Waiting for C# on {self.host}:{self.port} ...")

        try:
            self.client, addr = self.server.accept()
            self.client.setblocking(False)
            print(f"[{self.name}] C# connected: {addr}")
        except socket.timeout:
            print(f"[{self.name}] C# not connected now. Continuing camera only...")
        except Exception as ex:
            print(f"[{self.name}] Socket start error: {ex}")

    def poll_gui_context(self) -> str:
        if self.client is None or not self.read_context:
            return self.user_name

        try:
            while True:
                data = self.client.recv(4096)
                if not data:
                    break
                self.recv_buffer += data.decode("utf-8", errors="ignore")
                while "\n" in self.recv_buffer:
                    line, self.recv_buffer = self.recv_buffer.split("\n", 1)
                    self._parse_gui_context(line.strip())
        except BlockingIOError:
            pass
        except Exception as ex:
            print(f"[{self.name}] GUI context read error: {ex}")

        return self.user_name

    def _parse_gui_context(self, line: str):
        if not line:
            return

        parts = line.replace("\r", "").split(";")
        for part in parts:
            if "=" not in part:
                continue

            key, value = part.split("=", 1)
            key = key.strip().upper()
            value = value.strip()

            if key == "USER_NAME" and value:
                self.user_name = value
            elif key == "DEVICE" and value:
                self.device_name = value
            elif key == "LOGIN" and value:
                self.login_status = value

    def send_line(self, message: str) -> bool:
        if self.client is None:
            return False

        try:
            self.client.sendall((message + "\n").encode("utf-8"))
            return True
        except Exception as ex:
            print(f"[{self.name}] Send error: {ex}")
            self.client = None
            return False

    def close(self):
        for obj in [self.client, self.server]:
            try:
                if obj:
                    obj.close()
            except Exception:
                pass


# =========================
# Face + emotion
# =========================
def normalize_role(role: str) -> str:
    role = (role or "").strip().title()
    if role not in ["Chef", "Client"]:
        role = "Chef"
    return role


def load_face_roles() -> Dict[str, str]:
    roles = {}
    if not os.path.exists(FACE_ROLES_FILE):
        return roles

    try:
        with open(FACE_ROLES_FILE, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                name, role = line.split("=", 1)
                name = name.strip()
                if name:
                    roles[name] = normalize_role(role)
    except Exception as ex:
        print("[FACE ROLE] Load error:", ex)

    return roles


def save_face_role(user_name: str, role: str) -> None:
    user_name = sanitize_user_filename(user_name)
    role = normalize_role(role)
    if not user_name:
        return

    roles = load_face_roles()
    roles[user_name] = role

    try:
        with open(FACE_ROLES_FILE, "w", encoding="utf-8") as f:
            f.write("# Face role mapping. Existing people without a role default to Chef.\n")
            for name in sorted(roles.keys()):
                f.write(f"{name}={roles[name]}\n")
        print(f"[FACE ROLE] Saved: {user_name} -> {role}")
    except Exception as ex:
        print("[FACE ROLE] Save error:", ex)


def get_face_role(user_name: str) -> str:
    if not user_name or user_name == "Unknown":
        return ""
    return known_face_roles.get(user_name, "Chef")


def load_known_faces():
    global known_face_roles
    known_face_encodings.clear()
    known_face_names.clear()
    known_face_roles = load_face_roles()

    if not FACE_RECOGNITION_AVAILABLE:
        print("[FACE] Disabled because face_recognition is not installed.")
        return

    if not os.path.exists(PEOPLE_FOLDER):
        os.makedirs(PEOPLE_FOLDER)
        print("[FACE] people folder created. Add images inside it.")
        return

    for file_name in os.listdir(PEOPLE_FOLDER):
        if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(PEOPLE_FOLDER, file_name)

            try:
                image = face_recognition.load_image_file(path)
                encodings = face_recognition.face_encodings(image)

                if len(encodings) > 0:
                    user_name = os.path.splitext(file_name)[0]
                    known_face_encodings.append(encodings[0])
                    known_face_names.append(user_name)
                    if user_name not in known_face_roles:
                        known_face_roles[user_name] = "Chef"
                    print("[FACE] Loaded:", user_name, "Role=", known_face_roles.get(user_name, "Chef"))
                else:
                    print("[FACE] No face found in:", file_name)
            except Exception as ex:
                print("[FACE] Error loading", file_name, ":", ex)

    print("[FACE] Known users:", ", ".join(known_face_names) if known_face_names else "None")


def identify_face(frame) -> str:
    if not FACE_RECOGNITION_AVAILABLE or len(known_face_encodings) == 0:
        return "Unknown"

    try:
        small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        locations = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, locations)

        detected_name = "Unknown"

        for face_encoding in encodings:
            matches = face_recognition.compare_faces(
                known_face_encodings,
                face_encoding,
                tolerance=FACE_TOLERANCE
            )

            distances = face_recognition.face_distance(
                known_face_encodings,
                face_encoding
            )

            if len(distances) > 0:
                best_match = int(np.argmin(distances))
                if matches[best_match]:
                    detected_name = known_face_names[best_match]

        return detected_name

    except Exception as ex:
        print("[FACE] Identify error:", ex)
        return "Unknown"


def sanitize_user_filename(user_name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9 _-]+", "", user_name or "").strip()
    safe = re.sub(r"\s+", " ", safe)
    return safe


def register_new_face(frame, user_name: str, role: str) -> str:
    global known_face_encodings, known_face_names

    if not FACE_RECOGNITION_AVAILABLE:
        print("[FACE SIGNUP] Cannot register because face_recognition is not available.")
        return "Unknown"

    try:
        user_name = sanitize_user_filename(user_name)
        role = normalize_role(role)

        if not user_name:
            print("[FACE SIGNUP] Cancelled: empty user name.")
            return "Unknown"

        if not os.path.exists(PEOPLE_FOLDER):
            os.makedirs(PEOPLE_FOLDER)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb)

        if len(locations) == 0:
            print("[FACE SIGNUP] No face found in the captured frame.")
            return "Unknown"

        encodings = face_recognition.face_encodings(rgb, locations)

        if len(encodings) == 0:
            print("[FACE SIGNUP] Could not encode the face.")
            return "Unknown"

        base_name = user_name
        file_path = os.path.join(PEOPLE_FOLDER, f"{base_name}.jpg")
        counter = 2

        while os.path.exists(file_path):
            file_path = os.path.join(PEOPLE_FOLDER, f"{base_name}_{counter}.jpg")
            user_name = f"{base_name}_{counter}"
            counter += 1

        cv2.imwrite(file_path, frame)
        save_face_role(user_name, role)

        print(f"[FACE SIGNUP] Registered new {role}: {user_name}")
        print(f"[FACE SIGNUP] Saved image: {file_path}")

        # Reload the people folder after saving, so the new user is recognized immediately.
        load_known_faces()

        return user_name

    except Exception as ex:
        print("[FACE SIGNUP ERROR]", ex)
        return "Unknown"


def send_signup_status_to_csharp(stage: str, message: str = "", seconds: int = 0, role: str = "", name: str = "") -> bool:
    payload = (
        f"Detected=False;"
        f"Signup=True;"
        f"Stage={stage};"
        f"Message={message};"
        f"Seconds={seconds};"
        f"Role={role};"
        f"Name={name};"
        f"DeviceName=FaceRecognition;"
        f"Address=Camera;"
        f"Timestamp={datetime.utcnow().isoformat()}"
    )

    try:
        with socket.create_connection((HOST, PROFILE_PORT), timeout=1.2) as sock:
            sock.sendall((payload + "\n").encode("utf-8"))
        print("[SIGNUP] Sent to C#:", payload)
        return True
    except Exception as ex:
        print("[SIGNUP] Socket error:", ex)
        return False


def send_signup_error(message: str) -> None:
    send_signup_status_to_csharp("ERROR", message)


def signup_command_server_loop():
    global signup_requested, signup_name, signup_role, signup_countdown_until, signup_status

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        server.bind((HOST, SIGNUP_COMMAND_PORT))
        server.listen(1)
        server.settimeout(1.0)
        print(f"[SIGNUP COMMAND] Listening on {HOST}:{SIGNUP_COMMAND_PORT}")
    except Exception as ex:
        print("[SIGNUP COMMAND] Server start error:", ex)
        return

    while True:
        try:
            client, addr = server.accept()
        except socket.timeout:
            continue
        except Exception as ex:
            print("[SIGNUP COMMAND] Accept error:", ex)
            continue

        with client:
            try:
                data = client.recv(4096).decode("utf-8", errors="ignore").strip()
                if not data:
                    continue

                print("[SIGNUP COMMAND] Received:", data)
                parts = data.replace("\r", "").split(";")
                main = parts[0].strip().upper()
                values = {}
                for part in parts[1:]:
                    if "=" in part:
                        k, v = part.split("=", 1)
                        values[k.strip().upper()] = v.strip()

                with signup_lock:
                    if main == "SIGNUP_START":
                        signup_name = sanitize_user_filename(values.get("NAME", ""))
                        signup_role = ""
                        signup_requested = True
                        signup_status = "NAME_ENTERED"
                        signup_countdown_until = 0.0
                        send_signup_status_to_csharp("NAME_ENTERED", "Name saved. Use marker 61 for Chef or marker 62 for Client.", 0, signup_role, signup_name)

                    elif main == "SIGNUP_ROLE":
                        role_value = values.get("ROLE", "").strip().title()
                        if role_value not in ["Chef", "Client"]:
                            role_value = "Chef"
                        signup_role = role_value
                        signup_status = "ROLE_SELECTED"
                        send_signup_status_to_csharp("ROLE_SELECTED", f"{signup_role} selected. Use marker 63 to start 8 second capture countdown.", 0, signup_role, signup_name)

                    elif main == "SIGNUP_CAPTURE":
                        if not signup_name:
                            send_signup_error("Please use marker 60 and enter the user name first.")
                        elif not signup_role:
                            send_signup_error("Please choose Chef with marker 61 or Client with marker 62 first.")
                        else:
                            seconds = int(values.get("SECONDS", "8") or "8")
                            signup_countdown_until = time.time() + max(1, seconds)
                            signup_status = "COUNTDOWN"
                            send_signup_status_to_csharp("COUNTDOWN", "Hold still. Capturing face soon.", seconds, signup_role, signup_name)

                    elif main == "SIGNUP_CANCEL":
                        signup_requested = False
                        signup_name = ""
                        signup_role = ""
                        signup_countdown_until = 0.0
                        signup_status = "CANCELLED"
                        send_signup_status_to_csharp("CANCELLED", "Signup cancelled.")

                client.sendall(b"OK\n")
            except Exception as ex:
                print("[SIGNUP COMMAND] Handle error:", ex)


def start_signup_command_server():
    t = threading.Thread(target=signup_command_server_loop, daemon=True)
    t.start()



def detect_emotion(frame) -> str:
    global last_emotion

    if not DEEPFACE_AVAILABLE:
        return last_emotion

    try:
        result = DeepFace.analyze(
            img_path=frame,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend="opencv",
            silent=True
        )

        emotion = result[0]["dominant_emotion"].lower()

        if emotion == "happy":
            last_emotion = "Happy"
        elif emotion == "sad":
            last_emotion = "Sad"
        elif emotion == "angry":
            last_emotion = "Angry"
        elif emotion == "surprise":
            last_emotion = "Surprised"
        else:
            last_emotion = "Neutral"

        return last_emotion

    except Exception as ex:
        print("[EMOTION] Error:", ex)
        return last_emotion


def send_face_to_csharp(user_name: str, emotion: str, role: str = "") -> bool:
    role = normalize_role(role) if role else get_face_role(user_name)
    payload = (
        f"Detected=True;"
        f"Profile={user_name};"
        f"Role={role};"
        f"Emotion={emotion};"
        f"DeviceName=FaceRecognition;"
        f"Address=Camera;"
        f"Timestamp={datetime.utcnow().isoformat()}"
    )

    try:
        with socket.create_connection((HOST, PROFILE_PORT), timeout=1.2) as sock:
            sock.sendall((payload + "\n").encode("utf-8"))
            try:
                response = sock.recv(1024).decode("utf-8", errors="ignore").strip()
                if response:
                    print("[FACE] C# reply:", response)
            except Exception:
                pass

        print("[FACE] Sent to C#:", payload)
        return True

    except Exception as ex:
        print("[FACE] Socket error:", ex)
        return False



# =========================
# Gesture + Yellow Laser Control (MERGED into same camera)
# =========================
try:
    from dollarpy import Point, Template, Recognizer
    DOLLARPY_AVAILABLE = True
except Exception as ex:
    Point = Template = Recognizer = None
    DOLLARPY_AVAILABLE = False
    print("[GESTURE WARNING] dollarpy not available:", ex)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

MENU_NAMES = ["Overview", "Recipes", "Calories", "Steps", "Health", "Tips"]


def make_gesture_point(x, y, stroke_id=1):
    try:
        return Point(x, y, stroke_id)
    except TypeError:
        return Point(x, y)


def build_line_template(name, start, end, count=56):
    x1, y1 = start
    x2, y2 = end
    points = []
    for i in range(count):
        t = i / (count - 1)
        x = x1 + (x2 - x1) * t
        y = y1 + (y2 - y1) * t
        points.append(make_gesture_point(x, y))
    return Template(name, points)


def build_templates_for_direction(name, is_right=True):
    if is_right:
        starts_and_ends = [
            ((0.08, 0.50), (0.92, 0.50)), ((0.08, 0.46), (0.92, 0.46)),
            ((0.08, 0.54), (0.92, 0.54)), ((0.10, 0.42), (0.90, 0.48)),
            ((0.10, 0.58), (0.90, 0.52)), ((0.12, 0.40), (0.88, 0.50)),
            ((0.12, 0.60), (0.88, 0.50)), ((0.18, 0.50), (0.82, 0.50)),
            ((0.20, 0.45), (0.80, 0.45)), ((0.20, 0.55), (0.80, 0.55)),
        ]
    else:
        starts_and_ends = [
            ((0.92, 0.50), (0.08, 0.50)), ((0.92, 0.46), (0.08, 0.46)),
            ((0.92, 0.54), (0.08, 0.54)), ((0.90, 0.42), (0.10, 0.48)),
            ((0.90, 0.58), (0.10, 0.52)), ((0.88, 0.40), (0.12, 0.50)),
            ((0.88, 0.60), (0.12, 0.50)), ((0.82, 0.50), (0.18, 0.50)),
            ((0.80, 0.45), (0.20, 0.45)), ((0.80, 0.55), (0.20, 0.55)),
        ]
    return [build_line_template(name, s, e) for s, e in starts_and_ends]


class GestureLaserController:
    """DollarPy swipe gestures + yellow pointer, using the SAME frame/camera as YOLO/Gaze/Face."""
    def __init__(self, command_socket: OptionalSocketServer):
        self.socket = command_socket
        self.enabled = DOLLARPY_AVAILABLE
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.65,
        )
        self.recognizer = None
        if self.enabled:
            right_templates = build_templates_for_direction("RIGHT", is_right=True)
            left_templates = build_templates_for_direction("LEFT", is_right=False)
            self.recognizer = Recognizer(right_templates + left_templates)

        self.trajectory = []
        self.last_seen_time = 0.0
        self.last_move_time = 0.0
        self.last_action_time = 0.0
        self.last_point = None
        self.last_laser_send_time = 0.0
        self.status_text = "Gesture/Laser ready"
        self.status_color = (0, 255, 255)

        self.MIN_POINTS = 8
        self.MIN_HORIZONTAL_DISTANCE = 0.09
        self.MAX_VERTICAL_DRIFT = 0.26
        self.MIN_SCORE = 0.58
        self.COOLDOWN = 0.55
        self.GESTURE_END_TIMEOUT = 0.16
        self.STILLNESS_TIMEOUT = 0.11
        self.MIN_POINT_STEP = 0.0055
        self.MAX_TRAJECTORY_POINTS = 80
        self.MIN_DIRECTION_DOMINANCE = 1.45
        self.MAX_OPPOSITE_SHARE = 0.40
        self.YELLOW_MIN_AREA = 180
        self.LASER_SEND_COOLDOWN = 0.08

    def close(self):
        try:
            self.hands.close()
        except Exception:
            pass

    def send_command(self, command: str) -> bool:
        sent = self.socket.send_line(command)
        print("[GESTURE/LASER] Sent:" if sent else "[GESTURE/LASER] C# not connected:", command)
        return sent

    def reset_trajectory(self):
        self.trajectory = []
        self.last_point = None
        self.last_move_time = 0.0

    def add_point_if_moved(self, x, y, now):
        if self.last_point is None:
            self.trajectory.append((x, y))
            self.last_point = (x, y)
            self.last_move_time = now
            return
        dx = x - self.last_point[0]
        dy = y - self.last_point[1]
        dist = math.hypot(dx, dy)
        if dist >= self.MIN_POINT_STEP:
            self.trajectory.append((x, y))
            self.last_point = (x, y)
            self.last_move_time = now
            if len(self.trajectory) > self.MAX_TRAJECTORY_POINTS:
                self.trajectory[:] = self.trajectory[-self.MAX_TRAJECTORY_POINTS:]

    def analyze_path(self, points_xy):
        total_right = total_left = total_up = total_down = 0.0
        for i in range(1, len(points_xy)):
            dx = points_xy[i][0] - points_xy[i - 1][0]
            dy = points_xy[i][1] - points_xy[i - 1][1]
            if dx > 0: total_right += dx
            else: total_left += -dx
            if dy > 0: total_down += dy
            else: total_up += -dy
        x_start, y_start = points_xy[0]
        x_end, y_end = points_xy[-1]
        return {
            "net_dx": x_end - x_start,
            "net_dy": y_end - y_start,
            "total_right": total_right,
            "total_left": total_left,
            "total_up": total_up,
            "total_down": total_down,
            "vertical_total": total_up + total_down,
            "horizontal_total": total_right + total_left,
        }

    def path_matches_direction(self, gesture_name, info):
        net_dx = info["net_dx"]
        vertical_total = info["vertical_total"]
        if gesture_name == "RIGHT":
            forward = info["total_right"]
            opposite = info["total_left"]
            if net_dx <= 0: return False
        elif gesture_name == "LEFT":
            forward = info["total_left"]
            opposite = info["total_right"]
            if net_dx >= 0: return False
        else:
            return False
        if forward <= 0: return False
        dominance = forward / max(opposite, 1e-6)
        opposite_share = opposite / max(forward + opposite, 1e-6)
        if dominance < self.MIN_DIRECTION_DOMINANCE: return False
        if opposite_share > self.MAX_OPPOSITE_SHARE: return False
        if vertical_total > self.MAX_VERTICAL_DRIFT: return False
        return True

    def process_gesture(self, points_xy):
        if not self.enabled or self.recognizer is None or len(points_xy) < self.MIN_POINTS:
            return None, 0.0
        info = self.analyze_path(points_xy)
        if abs(info["net_dx"]) < self.MIN_HORIZONTAL_DISTANCE:
            return None, 0.0
        if info["vertical_total"] > self.MAX_VERTICAL_DRIFT:
            return None, 0.0
        dollar_points = [make_gesture_point(x, y) for x, y in points_xy]
        try:
            result = self.recognizer.recognize(dollar_points)
        except Exception as e:
            print("[GESTURE] DollarPy recognition error:", e)
            return None, 0.0
        if not result:
            return None, 0.0
        name, score = None, 0.0
        if isinstance(result, tuple) and len(result) >= 2:
            name, score = result[0], float(result[1])
        else:
            try:
                name = getattr(result, "name", None)
                score = float(getattr(result, "score", 0.0))
            except Exception:
                return None, 0.0
        if name not in ("RIGHT", "LEFT") or score < self.MIN_SCORE:
            return None, score
        if not self.path_matches_direction(name, info):
            return None, score
        return name, score

    def detect_yellow_laser(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_yellow = (18, 120, 120)
        upper_yellow = (40, 255, 255)
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < self.YELLOW_MIN_AREA:
            return None
        moment = cv2.moments(largest)
        if moment["m00"] == 0:
            return None
        cx = int(moment["m10"] / moment["m00"])
        cy = int(moment["m01"] / moment["m00"])
        return cx, cy, area

    def draw_laser_camera_hint(self, frame, lx, ly, area, menu_index=None):
        h, w = frame.shape[:2]
        cv2.circle(frame, (lx, ly), 24, (0, 255, 255), 3)
        cv2.circle(frame, (lx, ly), 7, (0, 255, 255), -1)
        cv2.line(frame, (max(0, lx - 45), ly), (min(w - 1, lx + 45), ly), (0, 255, 255), 2)
        cv2.line(frame, (lx, max(0, ly - 45)), (lx, min(h - 1, ly + 45)), (0, 255, 255), 2)
        target = MENU_NAMES[menu_index] if menu_index is not None and 0 <= menu_index < len(MENU_NAMES) else "center / none"
        cv2.putText(frame, f"YELLOW MENU: {target}", (25, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2)

    def process(self, frame):
        h, w = frame.shape[:2]
        now = time.time()
        laser = self.detect_yellow_laser(frame)
        laser_active = laser is not None
        hand_visible = False

        if laser_active:
            self.reset_trajectory()
            lx, ly, area = laser
            cx_frame = w / 2.0
            cy_frame = h / 2.0
            dx = lx - cx_frame
            dy = ly - cy_frame
            dist = math.sqrt(dx * dx + dy * dy)
            angle = math.degrees(math.atan2(dy, dx))
            if angle < 0:
                angle += 360
            menu_index = None
            if dist > 45:
                normalized = angle - 240
                while normalized < 0: normalized += 360
                while normalized >= 360: normalized -= 360
                menu_index = max(0, min(5, int(normalized / 60)))
            px = lx / w
            py = ly / h
            if now - self.last_laser_send_time > self.LASER_SEND_COOLDOWN:
                # Always send the raw laser position so the C# GUI can draw the dot directly on the signup keyboard.
                self.send_command(f"LASER_MENU;X={px:.4f};Y={py:.4f};Area={area:.1f}")
                # Also send menu slice data when the dashboard circular menu is being used.
                if menu_index is not None:
                    self.send_command(f"LASER_MENU_INDEX;INDEX={menu_index};PX={px:.4f};PY={py:.4f};Area={area:.1f}")
                self.last_laser_send_time = now
            self.status_text = "Yellow pointer active - hand skeleton paused"
            self.status_color = (0, 255, 255)
            self.draw_laser_camera_hint(frame, lx, ly, area, menu_index)
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)
            if results.multi_hand_landmarks:
                hand_visible = True
                hand = results.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                tip = hand.landmark[8]
                self.add_point_if_moved(tip.x, tip.y, now)
                self.last_seen_time = now

        if not laser_active:
            for i in range(1, len(self.trajectory)):
                x1 = int(self.trajectory[i - 1][0] * w)
                y1 = int(self.trajectory[i - 1][1] * h)
                x2 = int(self.trajectory[i][0] * w)
                y2 = int(self.trajectory[i][1] * h)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)

        should_finalize = False
        if (not laser_active) and self.trajectory:
            if hand_visible and self.last_move_time and (now - self.last_move_time > self.STILLNESS_TIMEOUT):
                should_finalize = True
            elif (not hand_visible) and self.last_seen_time and (now - self.last_seen_time > self.GESTURE_END_TIMEOUT):
                should_finalize = True

        if should_finalize:
            gesture_name, score = self.process_gesture(self.trajectory)
            if gesture_name and (now - self.last_action_time > self.COOLDOWN):
                if gesture_name == "RIGHT":
                    self.send_command("NEXT_STEP")
                    self.status_text = f"RIGHT = NEXT_STEP ({score:.2f})"
                    self.status_color = (0, 255, 0)
                elif gesture_name == "LEFT":
                    self.send_command("PREVIOUS_STEP")
                    self.status_text = f"LEFT = PREVIOUS_STEP ({score:.2f})"
                    self.status_color = (0, 255, 0)
                self.last_action_time = now
            else:
                self.status_text = "Gesture not recognized"
                self.status_color = (0, 0, 255)
            self.reset_trajectory()

        cv2.putText(frame, self.status_text, (25, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.status_color, 2)
        return laser_active


# =========================
# YOLO
# =========================
class YoloTracker:
    def __init__(self, socket_server: OptionalSocketServer):
        self.socket = socket_server
        self.last_sent: Dict[str, float] = {}
        self.last_rot_sent: Dict[str, float] = {}

        print("[YOLO] Loading model...")
        if not os.path.exists(MODEL_PATH):
            print(f"[YOLO WARNING] Model not found at: {MODEL_PATH}")

        self.model = YOLO(MODEL_PATH)
        print("[YOLO] Model loaded")

    def process(self, frame) -> List[Tuple[str, float, int, int, int, int]]:
        results = self.model(frame, verbose=False)
        detected_items = []

        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < CONF_THRESHOLD:
                    continue

                cls_id = int(box.cls[0])
                label = self.model.names[cls_id].lower()

                if label not in OBJECT_TO_COMMAND:
                    continue

                command = OBJECT_TO_COMMAND[label]
                now = time.time()

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # YOLO boxes do not provide real physical rotation like a TUIO marker.
                # So we create a "virtual rotation" from the object position around the camera center.
                # Move the fruit/object around the camera view to change quantity exactly like marker angle.
                h, w = frame.shape[:2]
                obj_cx = (x1 + x2) / 2.0
                obj_cy = (y1 + y2) / 2.0
                dx = obj_cx - (w / 2.0)
                dy = obj_cy - (h / 2.0)
                angle_rad = math.atan2(dy, dx) + (math.pi / 2.0)
                while angle_rad < 0:
                    angle_rad += 2.0 * math.pi
                while angle_rad >= 2.0 * math.pi:
                    angle_rad -= 2.0 * math.pi
                angle_deg = angle_rad * 180.0 / math.pi

                object_name = command.replace("OBJECT:", "")
                rot_command = f"OBJECT_ROT:{object_name};ANGLE={angle_rad:.4f};DEG={angle_deg:.1f}"

                if command not in self.last_sent or now - self.last_sent[command] > OBJECT_SEND_COOLDOWN:
                    sent = self.socket.send_line(command)
                    print("[YOLO] Sent:" if sent else "[YOLO] Detected but C# not connected:", command)
                    self.last_sent[command] = now

                if command not in self.last_rot_sent or now - self.last_rot_sent.get(command, 0) > OBJECT_ROT_SEND_COOLDOWN:
                    sent_rot = self.socket.send_line(rot_command)
                    print("[YOLO ROT] Sent:" if sent_rot else "[YOLO ROT] Detected but C# not connected:", rot_command)
                    self.last_rot_sent[command] = now

                detected_items.append((label, conf, x1, y1, x2, y2))

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{label} {conf:.2f} rot:{angle_deg:.0f}",
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2
                )

        return detected_items


# =========================
# Gaze
# =========================
def lm_xy(landmarks, idx, w, h):
    lm = landmarks[idx]
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)


def iris_center(landmarks, indices, w, h):
    pts = np.array([lm_xy(landmarks, i, w, h) for i in indices], dtype=np.float32)
    return pts.mean(axis=0)


def ratio_between(value, a, b):
    lo, hi = min(a, b), max(a, b)
    if hi - lo < 1.0:
        return 0.5
    return clamp((value - lo) / (hi - lo), 0.0, 1.0)


def eye_open_score(landmarks, w, h):
    l_top, l_bottom = lm_xy(landmarks, LEFT_EYE_TOP, w, h), lm_xy(landmarks, LEFT_EYE_BOTTOM, w, h)
    l_outer, l_inner = lm_xy(landmarks, LEFT_EYE_OUTER, w, h), lm_xy(landmarks, LEFT_EYE_INNER, w, h)
    r_top, r_bottom = lm_xy(landmarks, RIGHT_EYE_TOP, w, h), lm_xy(landmarks, RIGHT_EYE_BOTTOM, w, h)
    r_outer, r_inner = lm_xy(landmarks, RIGHT_EYE_OUTER, w, h), lm_xy(landmarks, RIGHT_EYE_INNER, w, h)

    l = np.linalg.norm(l_top - l_bottom) / max(np.linalg.norm(l_outer - l_inner), 1.0)
    r = np.linalg.norm(r_top - r_bottom) / max(np.linalg.norm(r_outer - r_inner), 1.0)
    return float((l + r) / 2.0)


def eye_vertical_ratio(face, iris, upper_indices, lower_indices, brow_indices, w, h):
    """
    Vertical gaze using BOTH iris + eyelids + eyebrow.
    Eyelids alone can make the point stay down. The eyebrow gives a higher
    reference line, so looking up becomes easier to detect.
    """
    upper_pts = np.array([lm_xy(face, idx, w, h) for idx in upper_indices], dtype=np.float32)
    lower_pts = np.array([lm_xy(face, idx, w, h) for idx in lower_indices], dtype=np.float32)
    brow_pts = np.array([lm_xy(face, idx, w, h) for idx in brow_indices], dtype=np.float32)

    all_eye_pts = np.vstack([upper_pts, lower_pts])
    eye_top_y = float(np.min(all_eye_pts[:, 1]))
    eye_bottom_y = float(np.max(all_eye_pts[:, 1]))
    eye_h = eye_bottom_y - eye_top_y
    if eye_h < 3.0:
        return 0.5

    # 1) normal iris-in-eye ratio
    eye_top_y -= eye_h * 0.55
    eye_bottom_y += eye_h * 0.45
    eyelid_ratio = clamp((float(iris[1]) - eye_top_y) / max(eye_bottom_y - eye_top_y, 1.0), 0.0, 1.0)

    # 2) eyebrow-to-lower-eye ratio. This improves UP movement.
    brow_y = float(np.mean(brow_pts[:, 1]))
    lower_y = float(np.mean(lower_pts[:, 1]))
    brow_to_lower = max(lower_y - brow_y, eye_h * 1.8)
    brow_ratio = clamp((float(iris[1]) - (brow_y + eye_h * 0.10)) / brow_to_lower, 0.0, 1.0)

    # More weight to eyebrow channel, because user asked for stronger UP detection.
    return clamp((eyelid_ratio * 0.45) + (brow_ratio * 0.55), 0.0, 1.0)


def estimate_gaze(face, w, h):
    l_outer, l_inner = lm_xy(face, LEFT_EYE_OUTER, w, h), lm_xy(face, LEFT_EYE_INNER, w, h)
    r_outer, r_inner = lm_xy(face, RIGHT_EYE_OUTER, w, h), lm_xy(face, RIGHT_EYE_INNER, w, h)

    l_iris = iris_center(face, LEFT_IRIS, w, h)
    r_iris = iris_center(face, RIGHT_IRIS, w, h)

    x_left = ratio_between(l_iris[0], l_outer[0], l_inner[0])
    x_right = ratio_between(r_iris[0], r_outer[0], r_inner[0])
    raw_x = float((x_left + x_right) / 2.0)

    y_left = eye_vertical_ratio(face, l_iris, LEFT_EYE_UPPER, LEFT_EYE_LOWER, LEFT_BROW, w, h)
    y_right = eye_vertical_ratio(face, r_iris, RIGHT_EYE_UPPER, RIGHT_EYE_LOWER, RIGHT_BROW, w, h)
    raw_y = float((y_left + y_right) / 2.0)

    face_l = lm_xy(face, FACE_LEFT, w, h)
    face_r = lm_xy(face, FACE_RIGHT, w, h)
    nose = lm_xy(face, NOSE_TIP, w, h)
    face_top = lm_xy(face, FACE_TOP, w, h)
    chin = lm_xy(face, CHIN, w, h)
    nose_ratio = ratio_between(nose[0], face_l[0], face_r[0])
    head_y_ratio = ratio_between(nose[1], face_top[1], chin[1])

    # Small compensation for head motion. Keep it small so eye movement is still dominant.
    yaw_bias = (nose_ratio - 0.5) * 0.08
    pitch_bias = (head_y_ratio - 0.50) * 0.10

    x = clamp(0.5 + ((raw_x - yaw_bias) - 0.5) * HORIZONTAL_GAIN, 0.0, 1.0)
    y = clamp(0.5 + ((raw_y - pitch_bias) - 0.5) * VERTICAL_GAIN, 0.0, 1.0)

    return x, raw_x, y, nose_ratio, eye_open_score(face, w, h), l_iris, r_iris


@dataclass
class Calibrator:
    """
    Smart auto-calibration.
    It starts by itself, shows one target at a time, collects only valid eye samples,
    and maps raw eye movement to full GUI coordinates.
    """
    center_x: float = 0.50
    center_y: float = 0.50
    left_x: float = 0.18
    right_x: float = 0.82
    up_y: float = 0.18
    down_y: float = 0.82
    auto_stage: str = "IDLE"
    stage_until: float = 0.0
    stage_started: float = 0.0
    values_x: list = field(default_factory=list)
    values_y: list = field(default_factory=list)
    done_once: bool = False

    def _axis_to_screen(self, value, low_value, high_value, center_value, min_span=0.18):
        span = high_value - low_value

        # Preserve orientation. If right-left or down-up is negative, the mapping still works.
        if abs(span) < min_span:
            sign = 1.0 if span >= 0 else -1.0
            low_value = center_value - sign * (min_span / 2.0)
            high_value = center_value + sign * (min_span / 2.0)
            span = high_value - low_value

        return clamp((value - low_value) / span, 0.0, 1.0)

    def _robust_value(self, values, fallback):
        if not values:
            return fallback
        arr = np.array(values, dtype=np.float32)
        # Remove first part of every stage because the user may still be moving his eyes.
        if len(arr) > 6:
            arr = arr[len(arr)//4:]
        return float(np.median(arr))

    def set_center(self, x, y=None):
        self.center_x = clamp(x, 0.02, 0.98)
        if y is not None:
            self.center_y = clamp(y, 0.02, 0.98)

        # Safe wide defaults before edge stages finish.
        self.left_x = clamp(self.center_x - 0.34, 0.02, 0.98)
        self.right_x = clamp(self.center_x + 0.34, 0.02, 0.98)
        self.up_y = clamp(self.center_y - 0.34, 0.02, 0.98)
        self.down_y = clamp(self.center_y + 0.34, 0.02, 0.98)

    def set_left(self, x):
        self.left_x = clamp(x, 0.02, 0.98)

    def set_right(self, x):
        self.right_x = clamp(x, 0.02, 0.98)

    def set_up(self, y):
        self.up_y = clamp(y, 0.02, 0.98)

    def set_down(self, y):
        self.down_y = clamp(y, 0.02, 0.98)

    def screen_point(self, x, y):
        sx = self._axis_to_screen(x, self.left_x, self.right_x, self.center_x, min_span=0.20)
        sy = self._axis_to_screen(y, self.up_y, self.down_y, self.center_y, min_span=0.18)

        # Expand movement around the screen center. This makes small eye movement
        # reach left/right/up/down without needing to move the whole head.
        sx = clamp(0.5 + (sx - 0.5) * FREE_SCREEN_X_GAIN, 0.0, 1.0)
        sy = clamp(0.5 + (sy - 0.5) * FREE_SCREEN_Y_GAIN, 0.0, 1.0)

        # tiny dead-zone around center to stop jitter when the user looks at the middle
        if abs(sx - 0.5) < 0.018:
            sx = 0.5
        if abs(sy - 0.5) < 0.018:
            sy = 0.5
        return sx, sy

    def classify_free_2d(self, sx, sy):
        h = "CENTER"
        v = "CENTER"

        if sx < 0.25:
            h = "LEFT"
        elif sx > 0.75:
            h = "RIGHT"

        if sy < 0.25:
            v = "UP"
        elif sy > 0.75:
            v = "DOWN"

        if h == "CENTER" and v == "CENTER":
            return "CENTER"
        if h == "CENTER":
            return v
        if v == "CENTER":
            return h
        return v + "_" + h

    def classify_zone_3(self, sx):
        if sx < 0.333:
            return "LEFT"
        if sx > 0.666:
            return "RIGHT"
        return "CENTER"

    def confidence(self, sx, sy):
        edge_strength = max(abs(sx - 0.5), abs(sy - 0.5)) * 2.0
        return clamp(0.30 + edge_strength * 0.70, 0.10, 1.0)

    def start_auto(self):
        self.auto_stage = "CENTER"
        self.stage_started = time.time()
        self.stage_until = self.stage_started + AUTO_CAL_SECONDS_PER_POINT
        self.values_x = []
        self.values_y = []
        print("[CAL] Auto calibration started. Follow the target: CENTER -> LEFT -> RIGHT -> UP -> DOWN")

    def progress(self):
        if self.auto_stage == "IDLE":
            return 1.0
        total = max(self.stage_until - self.stage_started, 0.001)
        return clamp((time.time() - self.stage_started) / total, 0.0, 1.0)

    def target_point(self, w, h):
        pad_x = int(w * 0.16)
        pad_y = int(h * 0.18)
        pts = {
            "CENTER": (w // 2, h // 2),
            "LEFT": (pad_x, h // 2),
            "RIGHT": (w - pad_x, h // 2),
            "UP": (w // 2, pad_y),
            "DOWN": (w // 2, h - pad_y),
        }
        return pts.get(self.auto_stage, (w // 2, h // 2))

    def update_auto(self, x, y):
        if self.auto_stage == "IDLE":
            return False

        self.values_x.append(x)
        self.values_y.append(y)

        if time.time() < self.stage_until:
            return True

        med_x = self._robust_value(self.values_x, x)
        med_y = self._robust_value(self.values_y, y)

        if self.auto_stage == "CENTER":
            self.set_center(med_x, med_y)
            self.auto_stage = "LEFT"
            print("[CAL] Center saved. Look at LEFT target.")
        elif self.auto_stage == "LEFT":
            self.set_left(med_x)
            self.auto_stage = "RIGHT"
            print("[CAL] Left saved. Look at RIGHT target.")
        elif self.auto_stage == "RIGHT":
            self.set_right(med_x)
            self.auto_stage = "UP"
            print("[CAL] Right saved. Look at TOP target.")
        elif self.auto_stage == "UP":
            self.set_up(med_y)
            self.auto_stage = "DOWN"
            print("[CAL] Up saved. Look at BOTTOM target.")
        elif self.auto_stage == "DOWN":
            self.set_down(med_y)
            self.auto_stage = "IDLE"
            self.done_once = True
            print(
                f"[CAL] Done. center=({self.center_x:.3f},{self.center_y:.3f}) "
                f"X left={self.left_x:.3f} right={self.right_x:.3f} | "
                f"Y up={self.up_y:.3f} down={self.down_y:.3f}"
            )
            self.values_x = []
            self.values_y = []
            return False

        self.values_x = []
        self.values_y = []
        self.stage_started = time.time()
        self.stage_until = self.stage_started + AUTO_CAL_SECONDS_PER_POINT
        return True

def init_logs():
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "timestamp", "user", "direction", "aoi", "ratio_x", "raw_x", "ratio_y",
            "screen_x", "screen_y", "confidence", "eye_open_score", "sent_to_csharp"
        ])


def append_log(user, direction, x, raw_x, y, screen_x, screen_y, conf, open_score, sent):
    aoi = AOI_MAP.get(direction, "Free Screen Point")
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            now_text(), user, direction, aoi, f"{x:.4f}", f"{raw_x:.4f}",
            f"{y:.4f}", f"{screen_x:.4f}", f"{screen_y:.4f}",
            f"{conf:.3f}", f"{open_score:.3f}", sent
        ])


def write_summary(user, hits, focus_seconds, started_at):
    total = sum(hits.values())
    dominant_key = None if total == 0 else max(hits.items(), key=lambda kv: kv[1])[0]
    dominant = "No gaze hits yet" if dominant_key is None else AOI_MAP.get(dominant_key, dominant_key)

    with open(SUMMARY_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Smart Kitchen All-In-One Gaze Summary Report"])
        w.writerow(["Generated At", now_text()])
        w.writerow(["User Name", user])
        w.writerow(["Session Duration Seconds", round(time.time() - started_at, 2)])
        w.writerow([])
        w.writerow(["Control Zone", "Hits", "Hit Percentage", "Focus Seconds"])
        for key in ["LEFT", "CENTER", "RIGHT", "UP", "DOWN", "UP_LEFT", "UP_RIGHT", "DOWN_LEFT", "DOWN_RIGHT"]:
            pct = 0 if total == 0 else round(hits[key] * 100 / total, 2)
            w.writerow([AOI_MAP.get(key, key), hits[key], f"{pct}%", round(focus_seconds[key], 2)])
        w.writerow([])
        w.writerow(["Total Valid Gaze Hits", total])
        w.writerow(["Most Looked AOI", dominant])
        w.writerow(["Adaptive Rule", "YOLO + Gaze + Face Recognition + Emotion run together on the same camera frame."])


def send_gaze(gaze_socket, zone, free_direction, ratio_x, ratio_y, screen_x, screen_y, confidence):
    gaze_socket.poll_gui_context()
    command = GAZE_COMMAND_MAP.get(zone, "GAZE_CENTER")
    msg = (
        f"{command};Mode=ZONE;Zone={zone};Direction={free_direction};"
        f"RatioX={ratio_x:.4f};RatioY={ratio_y:.4f};"
        f"ScreenX={screen_x:.4f};ScreenY={screen_y:.4f};"
        f"Confidence={confidence:.3f}"
    )
    return gaze_socket.send_line(msg)


def send_gaze_sample(gaze_socket, zone, free_direction, ratio_x, ratio_y, screen_x, screen_y, confidence):
    gaze_socket.poll_gui_context()
    z = zone or "NONE"
    d = free_direction or "NONE"
    msg = (
        f"GAZE_SAMPLE;Mode=FREE;Zone={z};Direction={d};"
        f"RatioX={ratio_x:.4f};RatioY={ratio_y:.4f};"
        f"ScreenX={screen_x:.4f};ScreenY={screen_y:.4f};"
        f"Confidence={confidence:.3f}"
    )
    return gaze_socket.send_line(msg)


# =========================
# Overlay
# =========================
def draw_overlay(frame, direction, stable, x, raw_x, y, conf, hits, cal, gaze_sent,
                 auto_stage, detected_items, face_user, emotion, l_iris=None, r_iris=None):
    h, w = frame.shape[:2]

    cv2.rectangle(frame, (0, 0), (w, 108), (0, 0, 0), -1)
    cv2.line(frame, (w // 3, 0), (w // 3, h), (90, 90, 90), 1)
    cv2.line(frame, (2 * w // 3, 0), (2 * w // 3, h), (90, 90, 90), 1)

    cv2.putText(frame, "Ingredients", (35, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2)
    cv2.putText(frame, "Dashboard", (w // 3 + 35, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2)
    cv2.putText(frame, "Nutrition", (2 * w // 3 + 35, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2)

    cv2.putText(frame, f"YOLO objects: {len(detected_items)}", (35, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
    cv2.putText(frame, f"Face: {face_user} | Emotion: {emotion}", (w // 3 + 35, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 0), 1)

    zone = stable or direction
    if zone:
        if zone == "LEFT":
            x1, x2 = 0, w // 3
        elif zone == "RIGHT":
            x1, x2 = 2 * w // 3, w
        else:
            x1, x2 = w // 3, 2 * w // 3

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, 108), (x2, h), (0, 255, 255), -1)
        cv2.addWeighted(overlay, 0.10, frame, 0.90, 0, frame)


    cv2.rectangle(frame, (10, h - 194), (w - 10, h - 10), (25, 25, 25), -1)
    cv2.putText(
        frame,
        f"Gaze:{direction or 'NO_FACE'} Stable:{stable or '-'} EyeX:{x:.3f} EyeY:{y:.3f} ScreenX:{cal.screen_point(x,y)[0]:.2f} ScreenY:{cal.screen_point(x,y)[1]:.2f} Conf:{conf:.2f} Sent:{gaze_sent}",
        (25, h - 156),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (0, 255, 255),
        2
    )
    cv2.putText(
        frame,
        f"Control Hits Ingredients:{hits['LEFT']} Dashboard:{hits['CENTER']} Nutrition:{hits['RIGHT']} Total:{sum(hits.values())}",
        (25, h - 122),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (0, 255, 0),
        2
    )
    cv2.putText(
        frame,
        f"Auto gaze calibration: X L={cal.left_x:.3f} R={cal.right_x:.3f} | Y U={cal.up_y:.3f} D={cal.down_y:.3f} | A recalibrate | Q quit",
        (25, h - 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (230, 230, 230),
        1
    )

    object_names = ", ".join([f"{name}:{score:.2f}" for name, score, *_ in detected_items[:5]]) or "None"
    cv2.putText(
        frame,
        f"Detected Objects: {object_names}",
        (25, h - 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (220, 220, 220),
        1
    )

    cv2.putText(
        frame,
        f"User Recognition: {face_user} | Mood: {emotion}",
        (25, h - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 220, 0),
        1
    )

    if auto_stage != "IDLE":
        tx, ty = cal.target_point(w, h)
        progress = cal.progress()
        cv2.circle(frame, (tx, ty), 42, (0, 255, 255), 4)
        cv2.circle(frame, (tx, ty), 10, (0, 255, 255), -1)
        cv2.line(frame, (tx - 65, ty), (tx + 65, ty), (0, 255, 255), 2)
        cv2.line(frame, (tx, ty - 65), (tx, ty + 65), (0, 255, 255), 2)
        bar_w = int(w * 0.45)
        bar_x = (w - bar_w) // 2
        bar_y = 130
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 18), (70, 70, 70), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * progress), bar_y + 18), (0, 255, 255), -1)
        cv2.putText(
            frame,
            f"AUTO CALIBRATION: LOOK AT THE {auto_stage} TARGET",
            (max(20, w // 2 - 365), 118),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            (0, 255, 255),
            3
        )
        cv2.putText(
            frame,
            "Keep your head steady - do not press anything",
            (max(20, w // 2 - 250), bar_y + 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2
        )


# =========================
# Main
# =========================
def main():
    global last_detected_user, last_emotion, last_face_send_time, signup_candidate_frame, unknown_notice_sent, signup_requested, signup_name, signup_role, signup_countdown_until, signup_status

    init_logs()
    load_known_faces()
    start_signup_command_server()

    yolo_socket = OptionalSocketServer("YOLO SOCKET", HOST, YOLO_PORT, SOCKET_WAIT_SECONDS, read_context=False)
    gaze_socket = OptionalSocketServer("GAZE SOCKET", HOST, GAZE_PORT, SOCKET_WAIT_SECONDS, read_context=True)

    yolo_socket.start()
    gaze_socket.start()

    user = gaze_socket.poll_gui_context()

    yolo_tracker = YoloTracker(yolo_socket)
    gesture_laser = GestureLaserController(yolo_socket)

    cap = open_camera()
    if not cap.isOpened():
        print("[ERROR] Camera not opened. Check camera permissions / index.")
        yolo_socket.close()
        gaze_socket.close()
        return

    print("[CAMERA] Opened successfully.")
    print("[RUNNING] YOLO + Gesture + Yellow Laser + Gaze + Face Recognition + Emotion are using ONE camera together.")

    hits = defaultdict(int)
    focus_seconds = defaultdict(float)
    x_values = deque(maxlen=SMOOTHING_WINDOW)
    y_values = deque(maxlen=SMOOTHING_WINDOW)
    zone_buffer = deque(maxlen=STABLE_FRAMES_REQUIRED)

    cal = Calibrator()
    auto_cal_request_time = time.time() + AUTO_CAL_START_DELAY_SECONDS
    print('[GAZE] Auto calibration will start by itself. Keep your face centered and follow the big target.')

    started_at = time.time()
    last_frame_time = time.time()
    last_gaze_sent_time = 0.0
    last_sample_sent_time = 0.0
    current_stable = None
    stable_started_at = None
    last_counted = None
    gaze_sent = False
    frame_index = 0

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.75,
    ) as face_mesh:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[CAMERA] Frame error")
                break

            frame_index += 1
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            now = time.time()
            if (not cal.done_once) and cal.auto_stage == "IDLE" and now >= auto_cal_request_time:
                cal.start_auto()
            dt = now - last_frame_time
            last_frame_time = now

            # 1) YOLO object tracking on the same frame
            detected_items = yolo_tracker.process(frame)

            # 1.5) Hand gestures + yellow laser on the SAME camera frame.
            # If yellow pointer is visible, hand skeleton pauses automatically.
            gesture_laser.process(frame)

            # 2) Face recognition, unknown notice, and marker-based signup support.
            if frame_index % FACE_PROCESS_EVERY_N_FRAMES == 0:
                detected_user = identify_face(frame)
                last_detected_user = detected_user

                if detected_user == "Unknown":
                    signup_candidate_frame = frame.copy()
                    if not unknown_notice_sent:
                        send_signup_status_to_csharp(
                            "UNKNOWN",
                            "We cannot recognize the user. Use marker 60 to sign up."
                        )
                        unknown_notice_sent = True
                else:
                    signup_candidate_frame = None
                    unknown_notice_sent = False

            if frame_index % EMOTION_PROCESS_EVERY_N_FRAMES == 0:
                last_emotion = detect_emotion(frame)

            if last_detected_user != "Unknown" and (now - last_face_send_time) >= FACE_SEND_COOLDOWN_SECONDS:
                send_face_to_csharp(last_detected_user, last_emotion)
                last_face_send_time = now

            # 3) Gaze tracking on the same frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = face_mesh.process(rgb)
            rgb.flags.writeable = True

            free_direction = None
            zone = None
            conf = 0.0
            x = raw_x = y = 0.5
            screen_x = screen_y = 0.5
            open_score = 0.0
            l_iris = r_iris = None

            if results.multi_face_landmarks:
                face = results.multi_face_landmarks[0].landmark
                x0, raw_x, raw_y, nose_ratio, open_score, l_iris, r_iris = estimate_gaze(face, w, h)

                if open_score >= MIN_EYE_OPEN_SCORE:
                    x_values.append(x0)
                    y_values.append(raw_y)
                    x = float(np.median(x_values))
                    y = float(np.median(y_values))
                    cal.update_auto(x, y)
                    screen_x, screen_y = cal.screen_point(x, y)
                    free_direction = cal.classify_free_2d(screen_x, screen_y)
                    zone = cal.classify_zone_3(screen_x)
                    conf = cal.confidence(screen_x, screen_y)
                else:
                    zone_buffer.clear()
                    current_stable = None
                    stable_started_at = None
            else:
                x_values.clear()
                y_values.clear()
                zone_buffer.clear()
                current_stable = None
                stable_started_at = None

            if cal.auto_stage == "IDLE" and (now - last_sample_sent_time) >= SAMPLE_SEND_INTERVAL:
                send_gaze_sample(gaze_socket, zone, free_direction, x, y, screen_x, screen_y, conf)
                last_sample_sent_time = now
                user = gaze_socket.poll_gui_context()

            if zone and conf >= MIN_CONFIDENCE and cal.auto_stage == "IDLE":
                zone_buffer.append(zone)

                if len(zone_buffer) == STABLE_FRAMES_REQUIRED and len(set(zone_buffer)) == 1:
                    stable = zone_buffer[-1]

                    if stable != current_stable:
                        current_stable = stable
                        stable_started_at = now
                        last_counted = None
                    else:
                        focus_seconds[stable] += dt

                    dwell_ok = stable_started_at is not None and (now - stable_started_at) >= DWELL_SECONDS
                    cooldown_ok = (now - last_gaze_sent_time) >= GAZE_SEND_COOLDOWN_SECONDS

                    if dwell_ok and cooldown_ok and stable != last_counted:
                        user = gaze_socket.poll_gui_context()
                        gaze_sent = send_gaze(gaze_socket, stable, free_direction, x, y, screen_x, screen_y, conf)
                        user = gaze_socket.poll_gui_context()
                        hits[stable] += 1
                        append_log(user, free_direction or stable, x, raw_x, y, screen_x, screen_y, conf, open_score, gaze_sent)
                        write_summary(user, hits, focus_seconds, started_at)
                        last_gaze_sent_time = now
                        last_counted = stable
                else:
                    current_stable = None
                    stable_started_at = None
                    last_counted = None

            # 6) Marker-based signup countdown/capture controlled by Demo.cs
            with signup_lock:
                countdown_ready = (
                    signup_status == "COUNTDOWN" and
                    signup_candidate_frame is not None and
                    signup_countdown_until > 0 and
                    time.time() >= signup_countdown_until
                )
                capture_name = signup_name
                capture_role = signup_role

            if countdown_ready:
                new_user = register_new_face(signup_candidate_frame, capture_name, capture_role)

                if new_user != "Unknown":
                    last_detected_user = new_user
                    signup_candidate_frame = None
                    unknown_notice_sent = False
                    last_face_send_time = 0.0

                    with signup_lock:
                        signup_requested = False
                        signup_name = ""
                        signup_role = ""
                        signup_countdown_until = 0.0
                        signup_status = "DONE"

                    send_signup_status_to_csharp("CAPTURED", "Face captured and account created. Signing in now.", 0, capture_role, new_user)
                    send_face_to_csharp(last_detected_user, last_emotion, capture_role)
                else:
                    with signup_lock:
                        signup_countdown_until = 0.0
                        signup_status = "ERROR"
                    send_signup_error("Capture failed. Please stand in front of the camera and use marker 63 again.")

            draw_overlay(
                frame=frame,
                direction=free_direction,
                stable=current_stable,
                x=x,
                raw_x=raw_x,
                y=y,
                conf=conf,
                hits=hits,
                cal=cal,
                gaze_sent=gaze_sent,
                auto_stage=cal.auto_stage,
                detected_items=detected_items,
                face_user=last_detected_user,
                emotion=last_emotion,
                l_iris=l_iris,
                r_iris=r_iris
            )

            if last_detected_user == "Unknown" and signup_candidate_frame is not None:
                cv2.rectangle(frame, (18, 112), (920, 170), (0, 0, 0), -1)
                with signup_lock:
                    local_status = signup_status
                    local_name = signup_name
                    local_role = signup_role
                    local_until = signup_countdown_until
                if local_status == "COUNTDOWN" and local_until > 0:
                    left = max(0, int(local_until - time.time()) + 1)
                    signup_text = f"Signup: {local_name} ({local_role}) - capture in {left}s"
                elif local_name and local_role:
                    signup_text = f"Signup: {local_name} ({local_role}) - use marker 63 to capture"
                elif local_name:
                    signup_text = f"Signup: {local_name} - use marker 61 Chef or 62 Client"
                else:
                    signup_text = "Unknown Face - use marker 60 to sign up"
                cv2.putText(
                    frame,
                    signup_text,
                    (30, 148),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 255),
                    2
                )

            cv2.imshow("Smart Kitchen ALL-IN-ONE Camera", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("a"):
                cal.start_auto()
            elif key == ord("c") and free_direction:
                cal.set_center(x, y)
                print(f"[CAL] CENTER X={x:.3f} Y={y:.3f}")
            elif key == ord("l") and free_direction:
                cal.set_left(x)
                print(f"[CAL] LEFT EDGE X={x:.3f}")
            elif key == ord("r") and free_direction:
                cal.set_right(x)
                print(f"[CAL] RIGHT EDGE X={x:.3f}")
            elif key == ord("u") and free_direction:
                cal.set_up(y)
                print(f"[CAL] TOP EDGE Y={y:.3f}")
            elif key == ord("d") and free_direction:
                cal.set_down(y)
                print(f"[CAL] BOTTOM EDGE Y={y:.3f}")
            elif key == ord("n"):
                print("[SIGNUP] Keyboard signup is disabled. Use markers 60, 61, 62, and 63 from Demo.cs.")
            elif key == ord("p"):
                load_known_faces()
                last_detected_user = "Unknown"
                signup_candidate_frame = None

    user = gaze_socket.poll_gui_context()
    write_summary(user, hits, focus_seconds, started_at)

    gesture_laser.close()
    cap.release()
    cv2.destroyAllWindows()
    yolo_socket.close()
    gaze_socket.close()

    print(f"[DONE] Saved {LOG_FILE} and {SUMMARY_FILE}")




# =========================================================
# MERGED FILE: Professional Gaze-Only Mode
# This keeps the second Python file inside this single file.
# Normal run uses main() above (YOLO + laser + face + emotion + gaze).
# Run with --professional-gaze or --gaze-only to start the old professional gaze tracker only.
# =========================================================
PROFESSIONAL_GAZE_ONLY_CODE = '"""\nSmart Kitchen Ultra Gaze Tracking\n- High-stability webcam gaze detection using MediaPipe FaceMesh iris landmarks\n- Adaptive GUI control through TCP socket to C# TuioDemo.cs\n- Reports: hits, focus seconds, dominant AOI, confidence, ratios\n\nRun order:\n1) Run C# Smart Kitchen app first OR run this first; both sides reconnect.\n2) python smart_kitchen_gaze_tracking_ultra.py\n\nNote:\n- No username input is required. Python receives USER_NAME from the C# GUI.\n\nInstall:\n    py -3.10 -m pip install opencv-python mediapipe numpy\n\nControls:\n    Q = quit and save reports\n    A = auto calibration sequence CENTER -> LEFT -> RIGHT\n    C = save current gaze as CENTER\n    L = save current gaze as LEFT\n    R = save current gaze as RIGHT\n\nSocket commands sent to C#:\n    GAZE_LEFT;RatioX=...;RatioY=...;Confidence=...\n    GAZE_CENTER;RatioX=...;RatioY=...;Confidence=...\n    GAZE_RIGHT;RatioX=...;RatioY=...;Confidence=...\n"""\n\nimport csv\nimport socket\nimport time\nfrom collections import defaultdict, deque\nfrom dataclasses import dataclass, field\nfrom datetime import datetime\n\nimport cv2\nimport mediapipe as mp\nimport numpy as np\n\nHOST = "127.0.0.1"\nPORT = 5001\nCAMERA_INDEX = 0\nFRAME_WIDTH = 960\nFRAME_HEIGHT = 540\n\nSMOOTHING_WINDOW = 5\nSTABLE_FRAMES_REQUIRED = 2\nDWELL_SECONDS = 0.08\nSEND_COOLDOWN_SECONDS = 0.15\nMIN_CONFIDENCE = 0.03\nMIN_EYE_OPEN_SCORE = 0.115\nSAMPLE_SEND_INTERVAL = 0.05\n\n# Easier LEFT detection for the left GUI panel (Scanned Items).\n# Smaller margin => left triggers sooner. 0.035 is a good balance.\nLEFT_TRIGGER_MARGIN = 0.035\nRIGHT_TRIGGER_MARGIN = 0.050\nAUTO_CAL_SECONDS_PER_POINT = 2.0\n\nLOG_FILE = "gaze_hits_log.csv"\nSUMMARY_FILE = "gaze_summary_report.csv"\n\nAOI_MAP = {\n    "LEFT": "Ingredients Control Zone",\n    "CENTER": "Dashboard Control Zone",\n    "RIGHT": "Nutrition Control Zone",\n}\n\nCOMMAND_MAP = {\n    "LEFT": "GAZE_LEFT",\n    "CENTER": "GAZE_CENTER",\n    "RIGHT": "GAZE_RIGHT",\n}\n\n# MediaPipe FaceMesh iris + eye indices\nLEFT_EYE_OUTER = 33\nLEFT_EYE_INNER = 133\nRIGHT_EYE_OUTER = 362\nRIGHT_EYE_INNER = 263\nLEFT_EYE_TOP = 159\nLEFT_EYE_BOTTOM = 145\nRIGHT_EYE_TOP = 386\nRIGHT_EYE_BOTTOM = 374\nLEFT_IRIS = [468, 469, 470, 471]\nRIGHT_IRIS = [473, 474, 475, 476]\nNOSE_TIP = 1\nFACE_LEFT = 234\nFACE_RIGHT = 454\n\nmp_face_mesh = mp.solutions.face_mesh\n\n\ndef now_text():\n    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")\n\n\ndef clamp(v, lo, hi):\n    return max(lo, min(hi, v))\n\n\ndef lm_xy(landmarks, idx, w, h):\n    lm = landmarks[idx]\n    return np.array([lm.x * w, lm.y * h], dtype=np.float32)\n\n\ndef iris_center(landmarks, indices, w, h):\n    pts = np.array([lm_xy(landmarks, i, w, h) for i in indices], dtype=np.float32)\n    return pts.mean(axis=0)\n\n\ndef ratio_between(value, a, b):\n    lo, hi = min(a, b), max(a, b)\n    if hi - lo < 1.0:\n        return 0.5\n    return clamp((value - lo) / (hi - lo), 0.0, 1.0)\n\n\ndef eye_open_score(landmarks, w, h):\n    l_top, l_bottom = lm_xy(landmarks, LEFT_EYE_TOP, w, h), lm_xy(landmarks, LEFT_EYE_BOTTOM, w, h)\n    l_outer, l_inner = lm_xy(landmarks, LEFT_EYE_OUTER, w, h), lm_xy(landmarks, LEFT_EYE_INNER, w, h)\n    r_top, r_bottom = lm_xy(landmarks, RIGHT_EYE_TOP, w, h), lm_xy(landmarks, RIGHT_EYE_BOTTOM, w, h)\n    r_outer, r_inner = lm_xy(landmarks, RIGHT_EYE_OUTER, w, h), lm_xy(landmarks, RIGHT_EYE_INNER, w, h)\n\n    l = np.linalg.norm(l_top - l_bottom) / max(np.linalg.norm(l_outer - l_inner), 1.0)\n    r = np.linalg.norm(r_top - r_bottom) / max(np.linalg.norm(r_outer - r_inner), 1.0)\n    return float((l + r) / 2.0)\n\n\ndef estimate_gaze(face, w, h):\n    l_outer, l_inner = lm_xy(face, LEFT_EYE_OUTER, w, h), lm_xy(face, LEFT_EYE_INNER, w, h)\n    r_outer, r_inner = lm_xy(face, RIGHT_EYE_OUTER, w, h), lm_xy(face, RIGHT_EYE_INNER, w, h)\n    l_top, l_bottom = lm_xy(face, LEFT_EYE_TOP, w, h), lm_xy(face, LEFT_EYE_BOTTOM, w, h)\n    r_top, r_bottom = lm_xy(face, RIGHT_EYE_TOP, w, h), lm_xy(face, RIGHT_EYE_BOTTOM, w, h)\n\n    l_iris = iris_center(face, LEFT_IRIS, w, h)\n    r_iris = iris_center(face, RIGHT_IRIS, w, h)\n\n    x_left = ratio_between(l_iris[0], l_outer[0], l_inner[0])\n    x_right = ratio_between(r_iris[0], r_outer[0], r_inner[0])\n    raw_x = float((x_left + x_right) / 2.0)\n\n    y_left = ratio_between(l_iris[1], l_top[1], l_bottom[1])\n    y_right = ratio_between(r_iris[1], r_top[1], r_bottom[1])\n    raw_y = float((y_left + y_right) / 2.0)\n\n    # Head-yaw compensation: when user turns head, iris ratio shifts falsely.\n    face_l = lm_xy(face, FACE_LEFT, w, h)\n    face_r = lm_xy(face, FACE_RIGHT, w, h)\n    nose = lm_xy(face, NOSE_TIP, w, h)\n    nose_ratio = ratio_between(nose[0], face_l[0], face_r[0])\n    yaw_bias = (nose_ratio - 0.5) * 0.10\n    x = clamp(raw_x - yaw_bias, 0.0, 1.0)\n\n    return x, raw_x, raw_y, nose_ratio, eye_open_score(face, w, h), l_iris, r_iris\n\n\n@dataclass\nclass Calibrator:\n    center_x: float = 0.50\n    left_x: float = 0.35\n    right_x: float = 0.65\n    left_threshold: float = 0.43\n    right_threshold: float = 0.57\n    auto_stage: str = "IDLE"\n    stage_until: float = 0.0\n    values: list = field(default_factory=list)\n\n    def recompute_thresholds(self):\n        # Professional calibrated thresholds.\n        # LEFT is intentionally a bit easier because the Scanned Items panel is at the far-left edge\n        # and webcam gaze often underestimates far-left eye movement.\n        calibrated_left = (self.left_x + self.center_x) / 2.0\n        calibrated_right = (self.right_x + self.center_x) / 2.0\n\n        assisted_left = self.center_x - LEFT_TRIGGER_MARGIN\n        assisted_right = self.center_x + RIGHT_TRIGGER_MARGIN\n\n        # Use the threshold that makes LEFT easier, but keep a valid center band.\n        self.left_threshold = clamp(max(calibrated_left, assisted_left), 0.25, 0.49)\n        self.right_threshold = clamp(min(calibrated_right, assisted_right), 0.51, 0.75)\n\n        if self.right_threshold - self.left_threshold < 0.035:\n            mid = (self.right_threshold + self.left_threshold) / 2.0\n            self.left_threshold = clamp(mid - 0.020, 0.25, 0.49)\n            self.right_threshold = clamp(mid + 0.020, 0.51, 0.75)\n\n    def set_center(self, x):\n        self.center_x = clamp(x, 0.25, 0.75)\n        self.recompute_thresholds()\n\n    def set_left(self, x):\n        self.left_x = clamp(x, 0.05, self.center_x - 0.03)\n        self.recompute_thresholds()\n\n    def set_right(self, x):\n        self.right_x = clamp(x, self.center_x + 0.03, 0.95)\n        self.recompute_thresholds()\n\n    def classify(self, x):\n        if x < self.left_threshold:\n            return "LEFT"\n        if x > self.right_threshold:\n            return "RIGHT"\n        return "CENTER"\n\n    def confidence(self, direction, x):\n        if direction == "CENTER":\n            dist = min(abs(x - self.left_threshold), abs(x - self.right_threshold))\n            return clamp(dist / 0.055, 0.0, 1.0)\n        if direction == "LEFT":\n            return clamp((self.left_threshold - x) / 0.080, 0.0, 1.0)\n        return clamp((x - self.right_threshold) / 0.080, 0.0, 1.0)\n\n    def start_auto(self):\n        self.auto_stage = "CENTER"\n        self.stage_until = time.time() + AUTO_CAL_SECONDS_PER_POINT\n        self.values = []\n        print("[CAL] Auto calibration started: look CENTER")\n\n    def update_auto(self, x):\n        if self.auto_stage == "IDLE":\n            return False\n        self.values.append(x)\n        if time.time() < self.stage_until:\n            return True\n\n        med = float(np.median(self.values)) if self.values else x\n        if self.auto_stage == "CENTER":\n            self.set_center(med)\n            self.auto_stage = "LEFT"\n            print("[CAL] Center saved. Now look LEFT")\n        elif self.auto_stage == "LEFT":\n            self.set_left(med)\n            self.auto_stage = "RIGHT"\n            print("[CAL] Left saved. Now look RIGHT")\n        elif self.auto_stage == "RIGHT":\n            self.set_right(med)\n            self.auto_stage = "IDLE"\n            print(f"[CAL] Done. thresholds: L<{self.left_threshold:.3f} R>{self.right_threshold:.3f}")\n            self.values = []\n            return False\n\n        self.values = []\n        self.stage_until = time.time() + AUTO_CAL_SECONDS_PER_POINT\n        return True\n\n\nclass CSharpSocket:\n    def __init__(self):\n        self.server = None\n        self.client = None\n        self.recv_buffer = ""\n        self.user_name = "GUI User"\n        self.device_name = "Unknown Device"\n        self.login_status = "0"\n\n    def wait_client(self):\n        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n        self.server.bind((HOST, PORT))\n        self.server.listen(1)\n        print(f"[SOCKET] Waiting for C# on {HOST}:{PORT} ...")\n        self.client, addr = self.server.accept()\n        self.client.setblocking(False)\n        print("[SOCKET] C# connected:", addr)\n\n    def poll_gui_context(self):\n        if self.client is None:\n            return self.user_name\n        try:\n            while True:\n                data = self.client.recv(4096)\n                if not data:\n                    break\n                self.recv_buffer += data.decode("utf-8", errors="ignore")\n                while "\\n" in self.recv_buffer:\n                    line, self.recv_buffer = self.recv_buffer.split("\\n", 1)\n                    self._parse_gui_context(line.strip())\n        except BlockingIOError:\n            pass\n        except Exception as ex:\n            print("[GUI CONTEXT READ ERROR]", ex)\n        return self.user_name\n\n    def _parse_gui_context(self, line):\n        # Expected from C#: USER_NAME=Belal;DEVICE=Manual login;LOGIN=1\n        if not line:\n            return\n        parts = line.replace("\\r", "").split(";")\n        for part in parts:\n            if "=" not in part:\n                continue\n            key, value = part.split("=", 1)\n            key = key.strip().upper()\n            value = value.strip()\n            if key == "USER_NAME" and value:\n                self.user_name = value\n            elif key == "DEVICE" and value:\n                self.device_name = value\n            elif key == "LOGIN" and value:\n                self.login_status = value\n\n    def send(self, direction, ratio_x, ratio_y, confidence):\n        if self.client is None:\n            return False\n        self.poll_gui_context()\n        msg = f"{COMMAND_MAP[direction]};RatioX={ratio_x:.4f};RatioY={ratio_y:.4f};Confidence={confidence:.3f}\\n"\n        try:\n            self.client.sendall(msg.encode("utf-8"))\n            return True\n        except Exception as ex:\n            print("[SOCKET ERROR]", ex)\n            return False\n\n    def send_sample(self, direction, ratio_x, ratio_y, confidence):\n        if self.client is None:\n            return False\n        self.poll_gui_context()\n        d = direction or "NONE"\n        msg = f"GAZE_SAMPLE;Direction={d};RatioX={ratio_x:.4f};RatioY={ratio_y:.4f};Confidence={confidence:.3f}\\n"\n        try:\n            self.client.sendall(msg.encode("utf-8"))\n            return True\n        except Exception:\n            return False\n\n    def close(self):\n        for obj in [self.client, self.server]:\n            try:\n                if obj:\n                    obj.close()\n            except Exception:\n                pass\n\n\ndef init_logs():\n    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:\n        csv.writer(f).writerow([\n            "timestamp", "user", "direction", "aoi", "ratio_x", "raw_x", "ratio_y",\n            "confidence", "eye_open_score", "sent_to_csharp"\n        ])\n\n\ndef append_log(user, direction, x, raw_x, y, conf, open_score, sent):\n    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:\n        csv.writer(f).writerow([\n            now_text(), user, direction, AOI_MAP[direction], f"{x:.4f}", f"{raw_x:.4f}",\n            f"{y:.4f}", f"{conf:.3f}", f"{open_score:.3f}", sent\n        ])\n\n\ndef write_summary(user, hits, focus_seconds, started_at):\n    total = sum(hits.values())\n    dominant = "No gaze hits yet" if total == 0 else AOI_MAP[max(hits.items(), key=lambda kv: kv[1])[0]]\n    with open(SUMMARY_FILE, "w", newline="", encoding="utf-8") as f:\n        w = csv.writer(f)\n        w.writerow(["Smart Kitchen Gaze Summary Report"])\n        w.writerow(["Generated At", now_text()])\n        w.writerow(["User Name", user])\n        w.writerow(["Session Duration Seconds", round(time.time() - started_at, 2)])\n        w.writerow([])\n        w.writerow(["Control Zone", "Hits", "Hit Percentage", "Focus Seconds"])\n        for key in ["LEFT", "CENTER", "RIGHT"]:\n            pct = 0 if total == 0 else round(hits[key] * 100 / total, 2)\n            w.writerow([AOI_MAP[key], hits[key], f"{pct}%", round(focus_seconds[key], 2)])\n        w.writerow([])\n        w.writerow(["Total Valid Gaze Hits", total])\n        w.writerow(["Most Looked AOI", dominant])\n        w.writerow(["Adaptive Rule", "Only three zones control the GUI; detailed GUI attention is logged by the C# professional report."])\n\n\ndef draw_overlay(frame, direction, stable, x, raw_x, y, conf, hits, cal, sent, auto_stage, l_iris=None, r_iris=None):\n    h, w = frame.shape[:2]\n    cv2.rectangle(frame, (0, 0), (w, 76), (0, 0, 0), -1)\n    cv2.line(frame, (w // 3, 0), (w // 3, h), (90, 90, 90), 1)\n    cv2.line(frame, (2 * w // 3, 0), (2 * w // 3, h), (90, 90, 90), 1)\n    cv2.putText(frame, "Ingredients", (35, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)\n    cv2.putText(frame, "Dashboard", (w // 3 + 35, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)\n    cv2.putText(frame, "Nutrition", (2 * w // 3 + 35, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)\n\n    zone = stable or direction\n    if zone:\n        if zone == "LEFT":\n            x1, x2 = 0, w // 3\n        elif zone == "RIGHT":\n            x1, x2 = 2 * w // 3, w\n        else:\n            x1, x2 = w // 3, 2 * w // 3\n        overlay = frame.copy()\n        cv2.rectangle(overlay, (x1, 76), (x2, h), (0, 255, 255), -1)\n        cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)\n\n    if l_iris is not None:\n        cv2.circle(frame, tuple(l_iris.astype(int)), 4, (0, 255, 255), -1)\n    if r_iris is not None:\n        cv2.circle(frame, tuple(r_iris.astype(int)), 4, (0, 255, 255), -1)\n\n    cv2.rectangle(frame, (10, h - 158), (w - 10, h - 10), (25, 25, 25), -1)\n    cv2.putText(frame, f"Gaze:{direction or \'NO_FACE\'} Stable:{stable or \'-\'} X:{x:.3f} Raw:{raw_x:.3f} Y:{y:.3f} Conf:{conf:.2f} Sent:{sent}",\n                (25, h - 120), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 255, 255), 2)\n    cv2.putText(frame, f"Control Hits Ingredients:{hits[\'LEFT\']} Dashboard:{hits[\'CENTER\']} Nutrition:{hits[\'RIGHT\']} Total:{sum(hits.values())}",\n                (25, h - 84), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 255, 0), 2)\n    cv2.putText(frame, f"Thresholds L<{cal.left_threshold:.3f} C={cal.center_x:.3f} R>{cal.right_threshold:.3f} | A auto | C/L/R manual | Q quit",\n                (25, h - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)\n    if auto_stage != "IDLE":\n        cv2.putText(frame, f"AUTO CALIBRATION: LOOK {auto_stage}", (w // 2 - 260, 118),\n                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 3)\n\n\ndef main():\n    init_logs()\n\n    sock = CSharpSocket()\n    sock.wait_client()\n    user = sock.poll_gui_context()\n\n    cap = cv2.VideoCapture(CAMERA_INDEX)\n    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)\n    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)\n    if not cap.isOpened():\n        print("[ERROR] Camera not opened. Check camera permissions / index.")\n        sock.close()\n        return\n\n    hits = defaultdict(int)\n    focus_seconds = defaultdict(float)\n    x_values = deque(maxlen=SMOOTHING_WINDOW)\n    y_values = deque(maxlen=SMOOTHING_WINDOW)\n    direction_buffer = deque(maxlen=STABLE_FRAMES_REQUIRED)\n    cal = Calibrator()\n    cal.start_auto()\n\n    started_at = time.time()\n    last_frame_time = time.time()\n    last_sent_time = 0.0\n    last_sample_sent_time = 0.0\n    current_stable = None\n    stable_started_at = None\n    last_counted = None\n    sent = False\n\n    with mp_face_mesh.FaceMesh(\n        max_num_faces=1,\n        refine_landmarks=True,\n        min_detection_confidence=0.75,\n        min_tracking_confidence=0.75,\n    ) as face_mesh:\n        while True:\n            ok, frame = cap.read()\n            if not ok:\n                break\n            frame = cv2.flip(frame, 1)\n            h, w = frame.shape[:2]\n            now = time.time()\n            dt = now - last_frame_time\n            last_frame_time = now\n\n            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)\n            rgb.flags.writeable = False\n            results = face_mesh.process(rgb)\n            rgb.flags.writeable = True\n\n            direction = None\n            conf = 0.0\n            x = raw_x = y = 0.5\n            open_score = 0.0\n            l_iris = r_iris = None\n\n            if results.multi_face_landmarks:\n                face = results.multi_face_landmarks[0].landmark\n                x0, raw_x, raw_y, nose_ratio, open_score, l_iris, r_iris = estimate_gaze(face, w, h)\n                if open_score >= MIN_EYE_OPEN_SCORE:\n                    x_values.append(x0)\n                    y_values.append(raw_y)\n                    x = float(np.median(x_values))\n                    y = float(np.median(y_values))\n                    cal.update_auto(x)\n                    direction = cal.classify(x)\n                    conf = cal.confidence(direction, x)\n                else:\n                    direction_buffer.clear()\n                    current_stable = None\n                    stable_started_at = None\n            else:\n                x_values.clear()\n                y_values.clear()\n                direction_buffer.clear()\n                current_stable = None\n                stable_started_at = None\n\n            if cal.auto_stage == "IDLE" and (now - last_sample_sent_time) >= SAMPLE_SEND_INTERVAL:\n                sock.send_sample(direction, x, y, conf)\n                last_sample_sent_time = now\n                user = sock.poll_gui_context()\n\n            if direction and conf >= MIN_CONFIDENCE and cal.auto_stage == "IDLE":\n                direction_buffer.append(direction)\n                if len(direction_buffer) == STABLE_FRAMES_REQUIRED and len(set(direction_buffer)) == 1:\n                    stable = direction_buffer[-1]\n                    if stable != current_stable:\n                        current_stable = stable\n                        stable_started_at = now\n                        last_counted = None\n                    else:\n                        focus_seconds[stable] += dt\n\n                    dwell_ok = stable_started_at is not None and (now - stable_started_at) >= DWELL_SECONDS\n                    cooldown_ok = (now - last_sent_time) >= SEND_COOLDOWN_SECONDS\n                    if dwell_ok and cooldown_ok and stable != last_counted:\n                        user = sock.poll_gui_context()\n                        sent = sock.send(stable, x, y, conf)\n                        user = sock.poll_gui_context()\n                        hits[stable] += 1\n                        append_log(user, stable, x, raw_x, y, conf, open_score, sent)\n                        write_summary(user, hits, focus_seconds, started_at)\n                        last_sent_time = now\n                        last_counted = stable\n                else:\n                    current_stable = None\n                    stable_started_at = None\n                    last_counted = None\n\n            draw_overlay(frame, direction, current_stable, x, raw_x, y, conf, hits, cal, sent, cal.auto_stage, l_iris, r_iris)\n            cv2.imshow("Smart Kitchen Ultra Gaze Tracking", frame)\n\n            key = cv2.waitKey(1) & 0xFF\n            if key == ord(\'q\'):\n                break\n            if key == ord(\'a\'):\n                cal.start_auto()\n            elif key == ord(\'c\') and direction:\n                cal.set_center(x)\n                print(f"[CAL] CENTER={x:.3f}")\n            elif key == ord(\'l\') and direction:\n                cal.set_left(x)\n                print(f"[CAL] LEFT={x:.3f}")\n            elif key == ord(\'r\') and direction:\n                cal.set_right(x)\n                print(f"[CAL] RIGHT={x:.3f}")\n\n    user = sock.poll_gui_context()\n    write_summary(user, hits, focus_seconds, started_at)\n    cap.release()\n    cv2.destroyAllWindows()\n    sock.close()\n    print(f"[DONE] Saved {LOG_FILE} and {SUMMARY_FILE}")\n\n\nif __name__ == "__main__":\n    main()\n'


def run_professional_gaze_only():
    """Run the original smart_kitchen_professional_gaze.py code from this merged file."""
    namespace = {
        "__name__": "__merged_professional_gaze__",
        "__file__": __file__,
    }
    exec(PROFESSIONAL_GAZE_ONLY_CODE, namespace)
    namespace["main"]()


def print_run_help():
    print("Smart Kitchen merged file")
    print("Normal full mode:")
    print("  py -3.10 smart_kitchen_merged_all_features.py")
    print("Professional gaze-only mode:")
    print("  py -3.10 smart_kitchen_merged_all_features.py --professional-gaze")
    print("  py -3.10 smart_kitchen_merged_all_features.py --gaze-only")


if __name__ == "__main__":
    import sys
    args = [a.strip().lower() for a in sys.argv[1:]]
    if "--help" in args or "-h" in args:
        print_run_help()
    elif "--professional-gaze" in args or "--gaze-only" in args:
        run_professional_gaze_only()
    else:
        main()
