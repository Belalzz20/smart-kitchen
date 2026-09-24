import cv2
import time
import socket
import math
import mediapipe as mp
from dollarpy import Point, Template, Recognizer

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

HOST = "127.0.0.1"
PORT = 5000


# =========================
# Socket Server
# =========================
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST, PORT))
server_socket.listen(1)

print(f"Waiting for C# connection on {HOST}:{PORT} ...")
client_socket, client_address = server_socket.accept()
print("C# connected from:", client_address)


def send_command(command: str) -> None:
    try:
        client_socket.sendall((command + "\n").encode("utf-8"))
        print("Sent:", command)
    except Exception as e:
        print("Socket send error:", e)


# =========================
# DollarPy helpers
# =========================
# Gesture recognition stays based on DollarPy dynamic gesture recognition.
def make_point(x, y, stroke_id=1):
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
        points.append(make_point(x, y))

    return Template(name, points)


def build_templates_for_direction(name, is_right=True):
    if is_right:
        starts_and_ends = [
            ((0.08, 0.50), (0.92, 0.50)),
            ((0.08, 0.46), (0.92, 0.46)),
            ((0.08, 0.54), (0.92, 0.54)),
            ((0.10, 0.42), (0.90, 0.48)),
            ((0.10, 0.58), (0.90, 0.52)),
            ((0.12, 0.40), (0.88, 0.50)),
            ((0.12, 0.60), (0.88, 0.50)),
            ((0.18, 0.50), (0.82, 0.50)),
            ((0.20, 0.45), (0.80, 0.45)),
            ((0.20, 0.55), (0.80, 0.55)),
        ]
    else:
        starts_and_ends = [
            ((0.92, 0.50), (0.08, 0.50)),
            ((0.92, 0.46), (0.08, 0.46)),
            ((0.92, 0.54), (0.08, 0.54)),
            ((0.90, 0.42), (0.10, 0.48)),
            ((0.90, 0.58), (0.10, 0.52)),
            ((0.88, 0.40), (0.12, 0.50)),
            ((0.88, 0.60), (0.12, 0.50)),
            ((0.82, 0.50), (0.18, 0.50)),
            ((0.80, 0.45), (0.20, 0.45)),
            ((0.80, 0.55), (0.20, 0.55)),
        ]

    return [build_line_template(name, s, e) for s, e in starts_and_ends]


# Templates are defined in the same coordinates the user sees on screen.
# We flip the frame first, then track/process on that flipped frame.
right_templates = build_templates_for_direction("RIGHT", is_right=True)
left_templates = build_templates_for_direction("LEFT", is_right=False)
recognizer = Recognizer(right_templates + left_templates)


# =========================
# MediaPipe setup
# =========================
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.65,
    min_tracking_confidence=0.65,
)


# =========================
# Gesture state
# =========================
trajectory = []
last_seen_time = 0.0
last_move_time = 0.0
last_action_time = 0.0
last_point = None

status_text = "Waiting..."
status_color = (0, 255, 255)

# Relaxed values so recognition works more consistently.
MIN_POINTS = 8
MIN_HORIZONTAL_DISTANCE = 0.09
MAX_VERTICAL_DRIFT = 0.26
MIN_SCORE = 0.58
COOLDOWN = 0.55
GESTURE_END_TIMEOUT = 0.16
STILLNESS_TIMEOUT = 0.11
MIN_POINT_STEP = 0.0055
MAX_TRAJECTORY_POINTS = 80
MIN_DIRECTION_DOMINANCE = 1.45
MAX_OPPOSITE_SHARE = 0.40

# =========================
# Yellow Laser / Yellow Pen Cap control for C# circular menu
# =========================
# Use a clearly yellow object (yellow pen cap / yellow tape / neon marker).
# Yellow is more stable than red because red can appear in skin/lips/background.
YELLOW_MIN_AREA = 180
LASER_SEND_COOLDOWN = 0.08
last_laser_send_time = 0.0
MENU_NAMES = ["Overview", "Recipes", "Calories", "Steps", "Health", "Tips"]


def draw_laser_pointer_overlay(frame, lx, ly, area, menu_index=None, dist=0.0, angle=0.0):
    """Draws a clear visible pointer for the yellow cap/laser on the camera screen."""
    h, w = frame.shape[:2]

    # Main pointer: big circle + filled center
    cv2.circle(frame, (lx, ly), 24, (0, 255, 255), 3)
    cv2.circle(frame, (lx, ly), 7, (0, 255, 255), -1)

    # Crosshair lines through the laser point
    cv2.line(frame, (max(0, lx - 45), ly), (min(w - 1, lx + 45), ly), (0, 255, 255), 2)
    cv2.line(frame, (lx, max(0, ly - 45)), (lx, min(h - 1, ly + 45)), (0, 255, 255), 2)

    # Center reference and line from center to pointer
    center_x = w // 2
    center_y = h // 2
    cv2.circle(frame, (center_x, center_y), 8, (255, 255, 255), 2)
    cv2.line(frame, (center_x, center_y), (lx, ly), (0, 255, 255), 2)

    # Approximate 6-zone guide in the camera window
    guide_radius = min(w, h) // 4
    cv2.circle(frame, (center_x, center_y), guide_radius, (90, 90, 90), 1)
    for i in range(6):
        a = math.radians(240 + i * 60)
        ex = int(center_x + math.cos(a) * guide_radius)
        ey = int(center_y + math.sin(a) * guide_radius)
        cv2.line(frame, (center_x, center_y), (ex, ey), (90, 90, 90), 1)

    # Text label near the point
    if menu_index is not None and 0 <= menu_index < len(MENU_NAMES):
        menu_text = f"Target: {MENU_NAMES[menu_index]}"
    else:
        menu_text = "Target: center / none"

    label_x = min(max(10, lx + 28), w - 300)
    label_y = min(max(65, ly - 18), h - 90)

    cv2.rectangle(frame, (label_x - 8, label_y - 34), (label_x + 285, label_y + 58), (20, 20, 20), -1)
    cv2.putText(frame, "YELLOW POINTER", (label_x, label_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2)
    cv2.putText(frame, menu_text, (label_x, label_y + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2)
    cv2.putText(frame, f"X:{lx} Y:{ly} Area:{area:.0f}", (label_x, label_y + 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1)


# =========================
# Gesture functions
# =========================
def reset_trajectory():
    global trajectory, last_point, last_move_time
    trajectory = []
    last_point = None
    last_move_time = 0.0


def add_point_if_moved(x, y, now):
    global last_point, trajectory, last_move_time

    if last_point is None:
        trajectory.append((x, y))
        last_point = (x, y)
        last_move_time = now
        return

    dx = x - last_point[0]
    dy = y - last_point[1]
    dist = math.hypot(dx, dy)

    if dist >= MIN_POINT_STEP:
        trajectory.append((x, y))
        last_point = (x, y)
        last_move_time = now

        if len(trajectory) > MAX_TRAJECTORY_POINTS:
            trajectory[:] = trajectory[-MAX_TRAJECTORY_POINTS:]


def analyze_path(points_xy):
    total_right = 0.0
    total_left = 0.0
    total_up = 0.0
    total_down = 0.0

    for i in range(1, len(points_xy)):
        dx = points_xy[i][0] - points_xy[i - 1][0]
        dy = points_xy[i][1] - points_xy[i - 1][1]

        if dx > 0:
            total_right += dx
        else:
            total_left += -dx

        if dy > 0:
            total_down += dy
        else:
            total_up += -dy

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


def path_matches_direction(gesture_name, info):
    net_dx = info["net_dx"]
    vertical_total = info["vertical_total"]

    if gesture_name == "RIGHT":
        forward = info["total_right"]
        opposite = info["total_left"]
        if net_dx <= 0:
            return False
    elif gesture_name == "LEFT":
        forward = info["total_left"]
        opposite = info["total_right"]
        if net_dx >= 0:
            return False
    else:
        return False

    if forward <= 0:
        return False

    dominance = forward / max(opposite, 1e-6)
    opposite_share = opposite / max(forward + opposite, 1e-6)

    if dominance < MIN_DIRECTION_DOMINANCE:
        return False

    if opposite_share > MAX_OPPOSITE_SHARE:
        return False

    if vertical_total > MAX_VERTICAL_DRIFT:
        return False

    return True


def process_gesture(points_xy):
    if len(points_xy) < MIN_POINTS:
        return None, 0.0

    info = analyze_path(points_xy)

    if abs(info["net_dx"]) < MIN_HORIZONTAL_DISTANCE:
        return None, 0.0

    if info["vertical_total"] > MAX_VERTICAL_DRIFT:
        return None, 0.0

    dollar_points = [make_point(x, y) for x, y in points_xy]

    try:
        result = recognizer.recognize(dollar_points)
    except Exception as e:
        print("DollarPy recognition error:", e)
        return None, 0.0

    if not result:
        return None, 0.0

    name = None
    score = 0.0

    if isinstance(result, tuple) and len(result) >= 2:
        name, score = result[0], float(result[1])
    else:
        try:
            name = getattr(result, "name", None)
            score = float(getattr(result, "score", 0.0))
        except Exception:
            return None, 0.0

    if name not in ("RIGHT", "LEFT"):
        return None, 0.0

    if score < MIN_SCORE:
        return None, score

    # DollarPy is still the recognizer.
    # The checks below only reject noisy paths that move heavily in the opposite direction.
    if not path_matches_direction(name, info):
        return None, score

    return name, score



# =========================
# Yellow Laser / Yellow Pen Cap detection
# =========================
def detect_yellow_laser(frame):
    """
    Detects the largest yellow blob in the camera frame.
    Returns: ((cx, cy, area), mask) or (None, mask)
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Yellow HSV range.
    # If your yellow cap is not detected, try lowering S/V from 120 to 90.
    lower_yellow = (18, 120, 120)
    upper_yellow = (40, 255, 255)

    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, mask

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)

    if area < YELLOW_MIN_AREA:
        return None, mask

    moment = cv2.moments(largest)
    if moment["m00"] == 0:
        return None, mask

    cx = int(moment["m10"] / moment["m00"])
    cy = int(moment["m01"] / moment["m00"])

    return (cx, cy, area), mask


# =========================
# Camera
# =========================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera error")
    hands.close()
    client_socket.close()
    server_socket.close()
    raise SystemExit

print("Gesture Control Ready")
print("Using DollarPy dynamic gesture recognition")
print("Swipe RIGHT = NEXT_STEP")
print("Swipe LEFT  = PREVIOUS_STEP")
print("Keyboard test: N = NEXT_STEP | P = PREVIOUS_STEP | Q = quit")


# =========================
# Main loop
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip first so text and motion on screen are not mirrored.
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    now = time.time()

    # First detect the yellow cap/laser.
    # If yellow is visible, we STOP MediaPipe hand skeleton for this frame
    # so the laser pointer and hand gesture system do not fight each other.
    laser, yellow_mask = detect_yellow_laser(frame)
    laser_active = laser is not None

    hand_visible = False

    if laser_active:
        # Clear any old hand trajectory while the yellow laser is controlling the menu.
        reset_trajectory()

        lx, ly, area = laser

        # Convert the yellow pointer position in the camera frame to menu slice index.
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
            # Same order as C# circularMenuItems:
            # 0 Overview, 1 Recipes, 2 Calories, 3 Steps, 4 Health, 5 Tips
            start_base = 240
            normalized = angle - start_base
            while normalized < 0:
                normalized += 360
            while normalized >= 360:
                normalized -= 360

            menu_index = int(normalized / 60)
            menu_index = max(0, min(5, menu_index))

        # Send the yellow pointer position to C# so the pointer is drawn on the GUI, not on the camera window.
        if menu_index is not None and now - last_laser_send_time > LASER_SEND_COOLDOWN:
            px = lx / w
            py = ly / h
            send_command(f"LASER_MENU_INDEX;INDEX={menu_index};PX={px:.4f};PY={py:.4f};Area={area:.1f}")
            last_laser_send_time = now

        status_text = "Yellow pointer active - hand skeleton paused"
        status_color = (0, 255, 255)

    else:
        # Yellow disappeared -> hand skeleton and swipe gestures work again.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            hand_visible = True
            hand = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            tip = hand.landmark[8]
            x, y = tip.x, tip.y

            add_point_if_moved(x, y, now)
            last_seen_time = now

    if not laser_active:
        for i in range(1, len(trajectory)):
            x1 = int(trajectory[i - 1][0] * w)
            y1 = int(trajectory[i - 1][1] * h)
            x2 = int(trajectory[i][0] * w)
            y2 = int(trajectory[i][1] * h)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)

    should_finalize = False
    if (not laser_active) and trajectory:
        if hand_visible and last_move_time and (now - last_move_time > STILLNESS_TIMEOUT):
            should_finalize = True
        elif (not hand_visible) and last_seen_time and (now - last_seen_time > GESTURE_END_TIMEOUT):
            should_finalize = True

    if should_finalize:
        gesture_name, score = process_gesture(trajectory)

        if gesture_name and (now - last_action_time > COOLDOWN):
            if gesture_name == "RIGHT":
                send_command("NEXT_STEP")
                status_text = f"RIGHT = NEXT_STEP ({score:.2f})"
                status_color = (0, 255, 0)
            elif gesture_name == "LEFT":
                send_command("PREVIOUS_STEP")
                status_text = f"LEFT = PREVIOUS_STEP ({score:.2f})"
                status_color = (0, 255, 0)

            last_action_time = now
        else:
            status_text = "Gesture not recognized"
            status_color = (0, 0, 255)

        reset_trajectory()

    # Clean camera window: no text overlays. The yellow pointer is drawn inside the C# GUI.
    cv2.imshow("Gesture Control", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('n'):
        send_command("NEXT_STEP")
        status_text = "NEXT_STEP (keyboard)"
        status_color = (0, 255, 0)
    elif key == ord('p'):
        send_command("PREVIOUS_STEP")
        status_text = "PREVIOUS_STEP (keyboard)"
        status_color = (0, 255, 0)
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
client_socket.close()
server_socket.close()
