"""
Smart Kitchen Ultra Gaze Tracking
- High-stability webcam gaze detection using MediaPipe FaceMesh iris landmarks
- Adaptive GUI control through TCP socket to C# TuioDemo.cs
- Reports: hits, focus seconds, dominant AOI, confidence, ratios

Run order:
1) Run C# Smart Kitchen app first OR run this first; both sides reconnect.
2) python smart_kitchen_gaze_tracking_ultra.py

Note:
- No username input is required. Python receives USER_NAME from the C# GUI.

Install:
    py -3.10 -m pip install opencv-python mediapipe numpy

Controls:
    Q = quit and save reports
    A = auto calibration sequence CENTER -> LEFT -> RIGHT
    C = save current gaze as CENTER
    L = save current gaze as LEFT
    R = save current gaze as RIGHT

Socket commands sent to C#:
    GAZE_LEFT;RatioX=...;RatioY=...;Confidence=...
    GAZE_CENTER;RatioX=...;RatioY=...;Confidence=...
    GAZE_RIGHT;RatioX=...;RatioY=...;Confidence=...
"""

import csv
import socket
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np

HOST = "127.0.0.1"
PORT = 5001
CAMERA_INDEX = 0
FRAME_WIDTH = 960
FRAME_HEIGHT = 540

SMOOTHING_WINDOW = 5
STABLE_FRAMES_REQUIRED = 2
DWELL_SECONDS = 0.08
SEND_COOLDOWN_SECONDS = 0.15
MIN_CONFIDENCE = 0.03
MIN_EYE_OPEN_SCORE = 0.115
SAMPLE_SEND_INTERVAL = 0.05

# Easier LEFT detection for the left GUI panel (Scanned Items).
# Smaller margin => left triggers sooner. 0.035 is a good balance.
LEFT_TRIGGER_MARGIN = 0.035
RIGHT_TRIGGER_MARGIN = 0.050
AUTO_CAL_SECONDS_PER_POINT = 2.0

LOG_FILE = "gaze_hits_log.csv"
SUMMARY_FILE = "gaze_summary_report.csv"

AOI_MAP = {
    "LEFT": "Ingredients Control Zone",
    "CENTER": "Dashboard Control Zone",
    "RIGHT": "Nutrition Control Zone",
}

COMMAND_MAP = {
    "LEFT": "GAZE_LEFT",
    "CENTER": "GAZE_CENTER",
    "RIGHT": "GAZE_RIGHT",
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
NOSE_TIP = 1
FACE_LEFT = 234
FACE_RIGHT = 454

mp_face_mesh = mp.solutions.face_mesh


def now_text():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


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


def estimate_gaze(face, w, h):
    l_outer, l_inner = lm_xy(face, LEFT_EYE_OUTER, w, h), lm_xy(face, LEFT_EYE_INNER, w, h)
    r_outer, r_inner = lm_xy(face, RIGHT_EYE_OUTER, w, h), lm_xy(face, RIGHT_EYE_INNER, w, h)
    l_top, l_bottom = lm_xy(face, LEFT_EYE_TOP, w, h), lm_xy(face, LEFT_EYE_BOTTOM, w, h)
    r_top, r_bottom = lm_xy(face, RIGHT_EYE_TOP, w, h), lm_xy(face, RIGHT_EYE_BOTTOM, w, h)

    l_iris = iris_center(face, LEFT_IRIS, w, h)
    r_iris = iris_center(face, RIGHT_IRIS, w, h)

    x_left = ratio_between(l_iris[0], l_outer[0], l_inner[0])
    x_right = ratio_between(r_iris[0], r_outer[0], r_inner[0])
    raw_x = float((x_left + x_right) / 2.0)

    y_left = ratio_between(l_iris[1], l_top[1], l_bottom[1])
    y_right = ratio_between(r_iris[1], r_top[1], r_bottom[1])
    raw_y = float((y_left + y_right) / 2.0)

    # Head-yaw compensation: when user turns head, iris ratio shifts falsely.
    face_l = lm_xy(face, FACE_LEFT, w, h)
    face_r = lm_xy(face, FACE_RIGHT, w, h)
    nose = lm_xy(face, NOSE_TIP, w, h)
    nose_ratio = ratio_between(nose[0], face_l[0], face_r[0])
    yaw_bias = (nose_ratio - 0.5) * 0.10
    x = clamp(raw_x - yaw_bias, 0.0, 1.0)

    return x, raw_x, raw_y, nose_ratio, eye_open_score(face, w, h), l_iris, r_iris


@dataclass
class Calibrator:
    center_x: float = 0.50
    left_x: float = 0.35
    right_x: float = 0.65
    left_threshold: float = 0.43
    right_threshold: float = 0.57
    auto_stage: str = "IDLE"
    stage_until: float = 0.0
    values: list = field(default_factory=list)

    def recompute_thresholds(self):
        # Professional calibrated thresholds.
        # LEFT is intentionally a bit easier because the Scanned Items panel is at the far-left edge
        # and webcam gaze often underestimates far-left eye movement.
        calibrated_left = (self.left_x + self.center_x) / 2.0
        calibrated_right = (self.right_x + self.center_x) / 2.0

        assisted_left = self.center_x - LEFT_TRIGGER_MARGIN
        assisted_right = self.center_x + RIGHT_TRIGGER_MARGIN

        # Use the threshold that makes LEFT easier, but keep a valid center band.
        self.left_threshold = clamp(max(calibrated_left, assisted_left), 0.25, 0.49)
        self.right_threshold = clamp(min(calibrated_right, assisted_right), 0.51, 0.75)

        if self.right_threshold - self.left_threshold < 0.035:
            mid = (self.right_threshold + self.left_threshold) / 2.0
            self.left_threshold = clamp(mid - 0.020, 0.25, 0.49)
            self.right_threshold = clamp(mid + 0.020, 0.51, 0.75)

    def set_center(self, x):
        self.center_x = clamp(x, 0.25, 0.75)
        self.recompute_thresholds()

    def set_left(self, x):
        self.left_x = clamp(x, 0.05, self.center_x - 0.03)
        self.recompute_thresholds()

    def set_right(self, x):
        self.right_x = clamp(x, self.center_x + 0.03, 0.95)
        self.recompute_thresholds()

    def classify(self, x):
        if x < self.left_threshold:
            return "LEFT"
        if x > self.right_threshold:
            return "RIGHT"
        return "CENTER"

    def confidence(self, direction, x):
        if direction == "CENTER":
            dist = min(abs(x - self.left_threshold), abs(x - self.right_threshold))
            return clamp(dist / 0.055, 0.0, 1.0)
        if direction == "LEFT":
            return clamp((self.left_threshold - x) / 0.080, 0.0, 1.0)
        return clamp((x - self.right_threshold) / 0.080, 0.0, 1.0)

    def start_auto(self):
        self.auto_stage = "CENTER"
        self.stage_until = time.time() + AUTO_CAL_SECONDS_PER_POINT
        self.values = []
        print("[CAL] Auto calibration started: look CENTER")

    def update_auto(self, x):
        if self.auto_stage == "IDLE":
            return False
        self.values.append(x)
        if time.time() < self.stage_until:
            return True

        med = float(np.median(self.values)) if self.values else x
        if self.auto_stage == "CENTER":
            self.set_center(med)
            self.auto_stage = "LEFT"
            print("[CAL] Center saved. Now look LEFT")
        elif self.auto_stage == "LEFT":
            self.set_left(med)
            self.auto_stage = "RIGHT"
            print("[CAL] Left saved. Now look RIGHT")
        elif self.auto_stage == "RIGHT":
            self.set_right(med)
            self.auto_stage = "IDLE"
            print(f"[CAL] Done. thresholds: L<{self.left_threshold:.3f} R>{self.right_threshold:.3f}")
            self.values = []
            return False

        self.values = []
        self.stage_until = time.time() + AUTO_CAL_SECONDS_PER_POINT
        return True


class CSharpSocket:
    def __init__(self):
        self.server = None
        self.client = None
        self.recv_buffer = ""
        self.user_name = "GUI User"
        self.device_name = "Unknown Device"
        self.login_status = "0"

    def wait_client(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((HOST, PORT))
        self.server.listen(1)
        print(f"[SOCKET] Waiting for C# on {HOST}:{PORT} ...")
        self.client, addr = self.server.accept()
        self.client.setblocking(False)
        print("[SOCKET] C# connected:", addr)

    def poll_gui_context(self):
        if self.client is None:
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
            print("[GUI CONTEXT READ ERROR]", ex)
        return self.user_name

    def _parse_gui_context(self, line):
        # Expected from C#: USER_NAME=Belal;DEVICE=Manual login;LOGIN=1
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

    def send(self, direction, ratio_x, ratio_y, confidence):
        if self.client is None:
            return False
        self.poll_gui_context()
        msg = f"{COMMAND_MAP[direction]};RatioX={ratio_x:.4f};RatioY={ratio_y:.4f};Confidence={confidence:.3f}\n"
        try:
            self.client.sendall(msg.encode("utf-8"))
            return True
        except Exception as ex:
            print("[SOCKET ERROR]", ex)
            return False

    def send_sample(self, direction, ratio_x, ratio_y, confidence):
        if self.client is None:
            return False
        self.poll_gui_context()
        d = direction or "NONE"
        msg = f"GAZE_SAMPLE;Direction={d};RatioX={ratio_x:.4f};RatioY={ratio_y:.4f};Confidence={confidence:.3f}\n"
        try:
            self.client.sendall(msg.encode("utf-8"))
            return True
        except Exception:
            return False

    def close(self):
        for obj in [self.client, self.server]:
            try:
                if obj:
                    obj.close()
            except Exception:
                pass


def init_logs():
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "timestamp", "user", "direction", "aoi", "ratio_x", "raw_x", "ratio_y",
            "confidence", "eye_open_score", "sent_to_csharp"
        ])


def append_log(user, direction, x, raw_x, y, conf, open_score, sent):
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            now_text(), user, direction, AOI_MAP[direction], f"{x:.4f}", f"{raw_x:.4f}",
            f"{y:.4f}", f"{conf:.3f}", f"{open_score:.3f}", sent
        ])


def write_summary(user, hits, focus_seconds, started_at):
    total = sum(hits.values())
    dominant = "No gaze hits yet" if total == 0 else AOI_MAP[max(hits.items(), key=lambda kv: kv[1])[0]]
    with open(SUMMARY_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Smart Kitchen Gaze Summary Report"])
        w.writerow(["Generated At", now_text()])
        w.writerow(["User Name", user])
        w.writerow(["Session Duration Seconds", round(time.time() - started_at, 2)])
        w.writerow([])
        w.writerow(["Control Zone", "Hits", "Hit Percentage", "Focus Seconds"])
        for key in ["LEFT", "CENTER", "RIGHT"]:
            pct = 0 if total == 0 else round(hits[key] * 100 / total, 2)
            w.writerow([AOI_MAP[key], hits[key], f"{pct}%", round(focus_seconds[key], 2)])
        w.writerow([])
        w.writerow(["Total Valid Gaze Hits", total])
        w.writerow(["Most Looked AOI", dominant])
        w.writerow(["Adaptive Rule", "Only three zones control the GUI; detailed GUI attention is logged by the C# professional report."])


def draw_overlay(frame, direction, stable, x, raw_x, y, conf, hits, cal, sent, auto_stage, l_iris=None, r_iris=None):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 76), (0, 0, 0), -1)
    cv2.line(frame, (w // 3, 0), (w // 3, h), (90, 90, 90), 1)
    cv2.line(frame, (2 * w // 3, 0), (2 * w // 3, h), (90, 90, 90), 1)
    cv2.putText(frame, "Ingredients", (35, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(frame, "Dashboard", (w // 3 + 35, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(frame, "Nutrition", (2 * w // 3 + 35, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    zone = stable or direction
    if zone:
        if zone == "LEFT":
            x1, x2 = 0, w // 3
        elif zone == "RIGHT":
            x1, x2 = 2 * w // 3, w
        else:
            x1, x2 = w // 3, 2 * w // 3
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, 76), (x2, h), (0, 255, 255), -1)
        cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)

    if l_iris is not None:
        cv2.circle(frame, tuple(l_iris.astype(int)), 4, (0, 255, 255), -1)
    if r_iris is not None:
        cv2.circle(frame, tuple(r_iris.astype(int)), 4, (0, 255, 255), -1)

    cv2.rectangle(frame, (10, h - 158), (w - 10, h - 10), (25, 25, 25), -1)
    cv2.putText(frame, f"Gaze:{direction or 'NO_FACE'} Stable:{stable or '-'} X:{x:.3f} Raw:{raw_x:.3f} Y:{y:.3f} Conf:{conf:.2f} Sent:{sent}",
                (25, h - 120), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 255, 255), 2)
    cv2.putText(frame, f"Control Hits Ingredients:{hits['LEFT']} Dashboard:{hits['CENTER']} Nutrition:{hits['RIGHT']} Total:{sum(hits.values())}",
                (25, h - 84), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 255, 0), 2)
    cv2.putText(frame, f"Thresholds L<{cal.left_threshold:.3f} C={cal.center_x:.3f} R>{cal.right_threshold:.3f} | A auto | C/L/R manual | Q quit",
                (25, h - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
    if auto_stage != "IDLE":
        cv2.putText(frame, f"AUTO CALIBRATION: LOOK {auto_stage}", (w // 2 - 260, 118),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 3)


def main():
    init_logs()

    sock = CSharpSocket()
    sock.wait_client()
    user = sock.poll_gui_context()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    if not cap.isOpened():
        print("[ERROR] Camera not opened. Check camera permissions / index.")
        sock.close()
        return

    hits = defaultdict(int)
    focus_seconds = defaultdict(float)
    x_values = deque(maxlen=SMOOTHING_WINDOW)
    y_values = deque(maxlen=SMOOTHING_WINDOW)
    direction_buffer = deque(maxlen=STABLE_FRAMES_REQUIRED)
    cal = Calibrator()
    cal.start_auto()

    started_at = time.time()
    last_frame_time = time.time()
    last_sent_time = 0.0
    last_sample_sent_time = 0.0
    current_stable = None
    stable_started_at = None
    last_counted = None
    sent = False

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.75,
    ) as face_mesh:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            now = time.time()
            dt = now - last_frame_time
            last_frame_time = now

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = face_mesh.process(rgb)
            rgb.flags.writeable = True

            direction = None
            conf = 0.0
            x = raw_x = y = 0.5
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
                    cal.update_auto(x)
                    direction = cal.classify(x)
                    conf = cal.confidence(direction, x)
                else:
                    direction_buffer.clear()
                    current_stable = None
                    stable_started_at = None
            else:
                x_values.clear()
                y_values.clear()
                direction_buffer.clear()
                current_stable = None
                stable_started_at = None

            if cal.auto_stage == "IDLE" and (now - last_sample_sent_time) >= SAMPLE_SEND_INTERVAL:
                sock.send_sample(direction, x, y, conf)
                last_sample_sent_time = now
                user = sock.poll_gui_context()

            if direction and conf >= MIN_CONFIDENCE and cal.auto_stage == "IDLE":
                direction_buffer.append(direction)
                if len(direction_buffer) == STABLE_FRAMES_REQUIRED and len(set(direction_buffer)) == 1:
                    stable = direction_buffer[-1]
                    if stable != current_stable:
                        current_stable = stable
                        stable_started_at = now
                        last_counted = None
                    else:
                        focus_seconds[stable] += dt

                    dwell_ok = stable_started_at is not None and (now - stable_started_at) >= DWELL_SECONDS
                    cooldown_ok = (now - last_sent_time) >= SEND_COOLDOWN_SECONDS
                    if dwell_ok and cooldown_ok and stable != last_counted:
                        user = sock.poll_gui_context()
                        sent = sock.send(stable, x, y, conf)
                        user = sock.poll_gui_context()
                        hits[stable] += 1
                        append_log(user, stable, x, raw_x, y, conf, open_score, sent)
                        write_summary(user, hits, focus_seconds, started_at)
                        last_sent_time = now
                        last_counted = stable
                else:
                    current_stable = None
                    stable_started_at = None
                    last_counted = None

            draw_overlay(frame, direction, current_stable, x, raw_x, y, conf, hits, cal, sent, cal.auto_stage, l_iris, r_iris)
            cv2.imshow("Smart Kitchen Ultra Gaze Tracking", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('a'):
                cal.start_auto()
            elif key == ord('c') and direction:
                cal.set_center(x)
                print(f"[CAL] CENTER={x:.3f}")
            elif key == ord('l') and direction:
                cal.set_left(x)
                print(f"[CAL] LEFT={x:.3f}")
            elif key == ord('r') and direction:
                cal.set_right(x)
                print(f"[CAL] RIGHT={x:.3f}")

    user = sock.poll_gui_context()
    write_summary(user, hits, focus_seconds, started_at)
    cap.release()
    cv2.destroyAllWindows()
    sock.close()
    print(f"[DONE] Saved {LOG_FILE} and {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
