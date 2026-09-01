import os
import time
from collections import deque

import cv2
import numpy as np
import mediapipe as mp

# ============================================================
# EMOTION MONITOR - ATTENTION + EMOTION DASHBOARD
# ------------------------------------------------------------
# IMPORTANT:
# 1. This file is standalone. It does NOT depend on attention.py
#    or eye_tracker.py.
# 2. It uses MediaPipe Face Landmarker + iris geometry.
# 3. Attention is calibrated automatically from your normal
#    camera-facing position. This fixes the "looking at camera
#    but 30-60%" problem caused by treating raw gaze blendshape
#    magnitudes as penalties.
#
# Controls:
#   Q / ESC : exit
#   F       : show/hide face points
#   R       : recalibrate attention
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "face_landmarker.task")

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        "\nMissing model:\n"
        f"{MODEL_PATH}\n\n"
        "Put face_landmarker.task inside the models folder."
    )

# ---------------- MediaPipe ----------------

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_faces=2,
    min_face_detection_confidence=0.50,
    min_face_presence_confidence=0.50,
    min_tracking_confidence=0.50,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
)

landmarker = FaceLandmarker.create_from_options(options)

# ---------------- Settings ----------------

WINDOW = "Emotion Monitor"
CAMERA_W = 960
CAMERA_H = 540

EMOTIONS = [
    "Happy",
    "Sad",
    "Angry",
    "Surprised",
    "Fear",
    "Disgust",
    "Neutral",
]

# MediaPipe Face Mesh / Face Landmarker landmark indices.
# Iris:
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

# Eye corners + upper/lower eyelid.
LEFT_EYE_IN = 133
LEFT_EYE_OUT = 33
LEFT_EYE_UP = 159
LEFT_EYE_DOWN = 145

RIGHT_EYE_IN = 362
RIGHT_EYE_OUT = 263
RIGHT_EYE_UP = 386
RIGHT_EYE_DOWN = 374

# Nose / face landmarks for head orientation.
NOSE = 1
CHIN = 152
LEFT_FACE_SIDE = 234
RIGHT_FACE_SIDE = 454

# ---------------- Utility ----------------

def clamp(x, lo=0.0, hi=1.0):
    return float(max(lo, min(hi, x)))


def lerp(old, new, alpha):
    return old + alpha * (new - old)


def dist(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def pt(face, idx):
    p = face[idx]
    return np.array([float(p.x), float(p.y)], dtype=np.float32)


def mean_pt(face, indices):
    return np.mean([pt(face, i) for i in indices], axis=0)


def safe_mean(values):
    values = [float(v) for v in values if np.isfinite(v)]
    return float(np.mean(values)) if values else 0.0


def normalize_name(name):
    return name.replace("_", "").replace("-", "").lower()


def feature(features, *names):
    vals = []
    for name in names:
        vals.append(float(features.get(normalize_name(name), 0.0)))
    return safe_mean(vals)


def get_blendshapes(result, face_index):
    data = {}
    if not result.face_blendshapes:
        return data
    if face_index >= len(result.face_blendshapes):
        return data

    for c in result.face_blendshapes[face_index]:
        name = getattr(c, "category_name", None)
        score = getattr(c, "score", None)
        if name is not None and score is not None:
            data[normalize_name(name)] = float(score)

    return data


# ============================================================
# FACE SELECTION
# ============================================================

def face_box(face):
    xs = [p.x for p in face]
    ys = [p.y for p in face]
    return min(xs), min(ys), max(xs), max(ys)


def choose_face(result):
    """Prefer a large, central face; avoid background faces."""
    if not result.face_landmarks:
        return None

    best = None
    best_score = -1e9

    for i, face in enumerate(result.face_landmarks):
        x1, y1, x2, y2 = face_box(face)
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        area = width * height

        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        center_distance = np.hypot(cx - 0.5, cy - 0.50)

        # Area matters, but center position also matters.
        score = area * 2.0 - center_distance * 0.25

        if score > best_score:
            best_score = score
            best = i

    return best


# ============================================================
# IRIS-BASED GAZE
# ============================================================

def eye_ratio(face, iris_indices, inner_idx, outer_idx, upper_idx, lower_idx):
    """
    Returns:
      horizontal: 0 = outer edge, 1 = inner edge
      vertical:   0 = upper lid, 1 = lower lid

    These are normalized to each eye, so camera distance and face
    size have very little effect.
    """
    iris = mean_pt(face, iris_indices)

    inner = pt(face, inner_idx)
    outer = pt(face, outer_idx)
    upper = pt(face, upper_idx)
    lower = pt(face, lower_idx)

    eye_vec = inner - outer
    eye_len2 = float(np.dot(eye_vec, eye_vec))

    vert_vec = lower - upper
    vert_len2 = float(np.dot(vert_vec, vert_vec))

    if eye_len2 < 1e-8 or vert_len2 < 1e-8:
        return 0.5, 0.5, 0.0

    horizontal = float(np.dot(iris - outer, eye_vec) / eye_len2)
    vertical = float(np.dot(iris - upper, vert_vec) / vert_len2)

    # Eye opening in normalized units.
    opening = float(np.linalg.norm(vert_vec))

    return horizontal, vertical, opening


def get_gaze_geometry(face):
    lh, lv, lopen = eye_ratio(
        face,
        LEFT_IRIS,
        LEFT_EYE_IN,
        LEFT_EYE_OUT,
        LEFT_EYE_UP,
        LEFT_EYE_DOWN,
    )

    rh, rv, ropen = eye_ratio(
        face,
        RIGHT_IRIS,
        RIGHT_EYE_IN,
        RIGHT_EYE_OUT,
        RIGHT_EYE_UP,
        RIGHT_EYE_DOWN,
    )

    return (
        safe_mean([lh, rh]),
        safe_mean([lv, rv]),
        safe_mean([lopen, ropen]),
    )


# ============================================================
# HEAD ORIENTATION
# ============================================================

def head_orientation(face):
    """
    Lightweight, stable orientation estimate.

    We intentionally do NOT use raw solvePnP Euler angles here.
    The old implementation could jump to ±170 degrees because
    webcam geometry is not a calibrated 3-D camera.

    yaw  : approximately -1..+1
    pitch: approximately -1..+1
    """
    left = pt(face, LEFT_FACE_SIDE)
    right = pt(face, RIGHT_FACE_SIDE)
    nose = pt(face, NOSE)
    chin = pt(face, CHIN)

    face_center = (left + right) * 0.5
    face_width = dist(left, right)

    if face_width < 1e-6:
        return 0.0, 0.0

    yaw = float((nose[0] - face_center[0]) / (face_width * 0.42))

    vertical_center = (face_center[1] + chin[1]) * 0.5
    face_height = max(dist(face_center, chin), 1e-6)

    pitch = float((nose[1] - vertical_center) / (face_height * 0.50))

    return float(np.clip(yaw, -1.0, 1.0)), float(np.clip(pitch, -1.0, 1.0))


# ============================================================
# ATTENTION CALIBRATION
# ============================================================

class AttentionTracker:
    """
    Attention = similarity to the user's calibrated camera-facing
    eye position, not "inverse sum of gaze blendshapes".

    This is the critical fix.

    Calibration collects normal looking-at-camera samples.
    The center is then a personalized reference.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.calibrating = True
        self.samples = []
        self.center_h = 0.5
        self.center_v = 0.5
        self.center_yaw = 0.0
        self.center_pitch = 0.0

        self.raw_history = deque(maxlen=7)
        self.gaze_history = deque(maxlen=5)
        self.score = 100.0
        self.last_valid_score = 100.0
        self.closed_since = None

    def add_calibration_sample(self, h, v, yaw, pitch):
        self.samples.append((h, v, yaw, pitch))

        # About 1.5 seconds at normal webcam FPS.
        if len(self.samples) >= 35:
            arr = np.array(self.samples, dtype=np.float32)

            # Median is much safer than mean against one bad frame.
            self.center_h = float(np.median(arr[:, 0]))
            self.center_v = float(np.median(arr[:, 1]))
            self.center_yaw = float(np.median(arr[:, 2]))
            self.center_pitch = float(np.median(arr[:, 3]))

            self.calibrating = False
            self.raw_history.clear()
            self.gaze_history.clear()
            self.score = 100.0
            self.last_valid_score = 100.0

    def calculate(self, h, v, yaw, pitch, eyes_open, now):
        if self.calibrating:
            self.add_calibration_sample(h, v, yaw, pitch)
            return 100.0

        dh = abs(h - self.center_h)
        dv = abs(v - self.center_v)
        dyaw = abs(yaw - self.center_yaw)
        dpitch = abs(pitch - self.center_pitch)

        # Ignore tiny landmark/webcam noise, but make genuine vertical
        # looking-away movement count. This is especially important for
        # "looking down at notes/phone" versus simply blinking.
        dh = max(0.0, dh - 0.040)
        dv = max(0.0, dv - 0.030)
        dyaw = max(0.0, dyaw - 0.065)
        dpitch = max(0.0, dpitch - 0.055)

        # Eye direction is primary.
        # Vertical gaze is intentionally a little more sensitive than
        # horizontal gaze because looking down is a common distraction.
        horizontal_distance = dh / 0.25
        vertical_distance = dv / 0.16

        eye_distance = np.sqrt(
            horizontal_distance ** 2 +
            vertical_distance ** 2
        )

        head_distance = np.sqrt(
            (dyaw / 0.48) ** 2 +
            (dpitch / 0.36) ** 2
        )

        eye_attention = 100.0 * np.exp(-0.95 * eye_distance * eye_distance)
        head_attention = 100.0 * np.exp(-0.72 * head_distance * head_distance)

        raw = 0.70 * eye_attention + 0.30 * head_attention

        # Explicit sustained-look-down/side penalty.
        # Small deviations remain focused; clear deviations cannot stay
        # at 80-90% merely because the face is still visible.
        if dv > 0.075:
            raw -= min(38.0, (dv - 0.075) * 260.0)

        if dh > 0.105:
            raw -= min(30.0, (dh - 0.105) * 180.0)

        # Normal blinks should not tank the score.
        if not eyes_open:
            if self.closed_since is None:
                self.closed_since = now

            closed_time = now - self.closed_since
            if closed_time < 0.28:
                raw = self.last_valid_score
            else:
                raw *= 0.30
        else:
            self.closed_since = None

        raw = float(np.clip(raw, 0.0, 100.0))
        self.raw_history.append(raw)

        median_raw = float(np.median(self.raw_history))
        self.score = lerp(self.score, median_raw, 0.34)
        self.score = float(np.clip(self.score, 0.0, 100.0))

        if eyes_open:
            self.last_valid_score = self.score

        return self.score

    def gaze_text(self, h, v, yaw, pitch):
        if self.calibrating:
            return "Calibrating..."

        dh = h - self.center_h
        dv = v - self.center_v

        H = 0.075
        V = 0.055

        if abs(dh) < H and abs(dv) < V:
            label = "Looking Center"
        elif abs(dh) > abs(dv):
            label = "Looking Right" if dh < 0 else "Looking Left"
        else:
            label = "Looking Down" if dv > 0 else "Looking Up"

        self.gaze_history.append(label)

        # Require most of the recent frames to agree. This keeps the
        # label stable without hiding a genuine sustained look-down.
        if len(self.gaze_history) >= 3:
            counts = {}
            for item in self.gaze_history:
                counts[item] = counts.get(item, 0) + 1
            label = max(counts, key=counts.get)

        return label


# ============================================================
# EMOTION DETECTION
# ============================================================

def emotion_evidence(f):
    """
    Expression classification from MediaPipe blendshapes.

    This is intentionally evidence-based rather than assigning
    arbitrary emotion percentages. Each expression has positive
    cues and contradiction penalties.

    MediaPipe blendshapes are expression features, NOT a medical
    or psychological emotion diagnosis.
    """

    smile = feature(f, "mouthSmileLeft", "mouthSmileRight")
    dimple = feature(f, "mouthDimpleLeft", "mouthDimpleRight")
    cheek = feature(f, "cheekSquintLeft", "cheekSquintRight")

    brow_inner = feature(f, "browInnerUp")
    brow_down = feature(f, "browDownLeft", "browDownRight")
    brow_outer = feature(f, "browOuterUpLeft", "browOuterUpRight")

    frown = feature(f, "mouthFrownLeft", "mouthFrownRight")
    press = feature(f, "mouthPressLeft", "mouthPressRight")
    stretch = feature(f, "mouthStretchLeft", "mouthStretchRight")
    funnel = feature(f, "mouthFunnel")
    pucker = feature(f, "mouthPucker")
    jaw = feature(f, "jawOpen")

    eye_wide = feature(f, "eyeWideLeft", "eyeWideRight")
    eye_squint = feature(f, "eyeSquintLeft", "eyeSquintRight")

    sneer = feature(f, "noseSneerLeft", "noseSneerRight")
    upper_lip = feature(
        f,
        "mouthUpperLipRaiseLeft",
        "mouthUpperLipRaiseRight",
    )

    # Stronger, normalized evidence values.
    happy = (
        2.00 * smile +
        0.55 * dimple +
        0.30 * cheek +
        0.15 * eye_squint
        - 0.90 * frown
        - 0.55 * brow_down
    )

    sad = (
        1.30 * brow_inner +
        1.35 * frown +
        0.55 * press +
        0.20 * pucker
        - 1.00 * smile
    )

    angry = (
        1.65 * brow_down +
        0.85 * press +
        0.55 * frown +
        0.35 * sneer
        - 0.80 * smile
        - 0.35 * brow_inner
    )

    surprised = (
        1.35 * eye_wide +
        1.25 * jaw +
        0.90 * brow_inner +
        0.55 * brow_outer +
        0.40 * stretch
    )

    fear = (
        1.00 * eye_wide +
        1.10 * brow_inner +
        0.70 * stretch +
        0.40 * jaw +
        0.25 * funnel
        - 0.55 * smile
        - 0.25 * brow_down
    )

    disgust = (
        1.70 * sneer +
        1.00 * upper_lip +
        0.55 * funnel +
        0.35 * brow_down +
        0.25 * press
        - 0.70 * smile
    )

    return {
        "Happy": max(0.0, happy),
        "Sad": max(0.0, sad),
        "Angry": max(0.0, angry),
        "Surprised": max(0.0, surprised),
        "Fear": max(0.0, fear),
        "Disgust": max(0.0, disgust),
    }


def emotion_probabilities(features, old_probs):
    """
    Stable facial-expression classifier built from MediaPipe blendshapes.

    The important change is temporal evidence accumulation:
    an emotion must remain stronger than its competitors for several
    frames before it becomes the displayed label. This greatly reduces
    the old Happy/Neutral lock-in and frame-to-frame flicker.
    """
    evidence = emotion_evidence(features)

    # Boost expression-specific evidence slightly so subtle Sad/Angry/
    # Surprise/Fear/Disgust movements are not drowned by Neutral.
    weights = {
        "Happy": 1.00,
        "Sad": 1.16,
        "Angry": 1.12,
        "Surprised": 1.18,
        "Fear": 1.12,
        "Disgust": 1.20,
    }

    scores = np.array([
        max(0.0, evidence[name] - 0.055) * weights[name]
        for name in EMOTIONS[:-1]
    ], dtype=np.float32)

    peak = float(np.max(scores)) if len(scores) else 0.0

    # Neutral is intentionally strong only when expression evidence is weak.
    neutral_score = max(0.08, 0.42 - 0.42 * peak)

    raw = np.append(scores, neutral_score)

    # Softmax keeps scores comparable and gives a clean visual distribution.
    temperature = 0.24
    z = raw / temperature
    z -= np.max(z)
    probs = np.exp(z)
    probs /= np.sum(probs)

    targets = {
        name: float(probs[i] * 100.0)
        for i, name in enumerate(EMOTIONS)
    }

    result = {}
    for name in EMOTIONS:
        # Faster response when an emotion is clearly dominant, otherwise
        # smooth more aggressively.
        target = targets[name]
        alpha = 0.24 if target > 40 else 0.16
        result[name] = lerp(
            float(old_probs.get(name, 0.0)),
            target,
            alpha,
        )

    total = sum(result.values())
    if total > 0:
        result = {
            name: value * 100.0 / total
            for name, value in result.items()
        }

    ranked = sorted(
        result.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    top_name, top_prob = ranked[0]
    second_prob = ranked[1][1]

    # A non-neutral emotion needs both reasonable probability and a
    # meaningful lead over its nearest competitor.
    if top_name != "Neutral":
        if top_prob < 28.0 or top_prob - second_prob < 4.0:
            top_name = "Neutral"

    confidence = float(np.clip(
        42.0
        + max(0.0, top_prob - second_prob) * 2.7
        + top_prob * 0.28,
        0.0,
        98.0,
    ))

    if top_name == "Neutral":
        confidence = float(np.clip(
            max(result["Neutral"], 50.0),
            0.0,
            98.0,
        ))

    return result, top_name, confidence


# ============================================================
# EYE STATE
# ============================================================

def eye_open_state(features, geometry_opening):
    blink = feature(
        features,
        "eyeBlinkLeft",
        "eyeBlinkRight",
    )

    # Blendshape + geometry. Either can support "closed".
    if blink > 0.58:
        return False

    if geometry_opening < 0.010:
        return False

    return True


# ============================================================
# DRAWING
# ============================================================

def put_text(img, s, pos, size=0.6, color=(235, 235, 235), thick=2):
    cv2.putText(
        img,
        str(s),
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        size,
        color,
        thick,
        cv2.LINE_AA,
    )


def draw_bar(img, x, y, width, height, value, label, color):
    value = float(np.clip(value, 0.0, 100.0))

    put_text(
        img,
        label,
        (x - 105, y + height - 4),
        0.45,
        (215, 215, 215),
        1,
    )

    cv2.rectangle(
        img,
        (x, y),
        (x + width, y + height),
        (80, 84, 92),
        1,
    )

    fill = int((width - 4) * value / 100.0)

    if fill > 0:
        cv2.rectangle(
            img,
            (x + 2, y + 2),
            (x + 2 + fill, y + height - 2),
            color,
            -1,
        )

    put_text(
        img,
        f"{value:.0f}%",
        (x + width + 10, y + height - 4),
        0.43,
        (220, 220, 220),
        1,
    )


def draw_points(frame, face):
    h, w = frame.shape[:2]

    # Dense enough to show tracking, not so dense that it hides the face.
    for i, p in enumerate(face):
        if i % 5 != 0:
            continue

        x = int(p.x * w)
        y = int(p.y * h)

        if 0 <= x < w and 0 <= y < h:
            cv2.circle(
                frame,
                (x, y),
                1,
                (50, 255, 80),
                -1,
            )


def build_dashboard(
    frame,
    attention,
    status,
    gaze,
    eyes_open,
    emotion,
    confidence,
    probabilities,
    faces,
    fps,
    show_points,
    calibrating,
):
    h, w = frame.shape[:2]

    # --------------------------------------------------------
    # Premium, presentation-ready layout
    # --------------------------------------------------------
    panel_w = 380
    canvas = np.zeros((h, w + panel_w, 3), dtype=np.uint8)
    canvas[:, :w] = frame
    canvas[:, w:] = (15, 18, 25)

    # Header
    cv2.rectangle(
        canvas,
        (0, 0),
        (w + panel_w, 58),
        (24, 28, 37),
        -1,
    )
    cv2.line(
        canvas,
        (0, 58),
        (w + panel_w, 58),
        (58, 67, 80),
        1,
    )

    put_text(
        canvas,
        "EMOTION MONITOR",
        (22, 38),
        0.76,
        (245, 247, 250),
        2,
    )

    put_text(
        canvas,
        "● LIVE",
        (w + panel_w - 82, 36),
        0.46,
        (70, 235, 140),
        2,
    )

    # Camera area border
    cv2.rectangle(
        canvas,
        (8, 66),
        (w - 8, h - 55),
        (65, 72, 82),
        1,
    )

    # Camera status card
    card_x1, card_y1 = 22, 80
    card_x2, card_y2 = w - 22, 206

    overlay = canvas.copy()
    cv2.rectangle(
        overlay,
        (card_x1, card_y1),
        (card_x2, card_y2),
        (9, 12, 17),
        -1,
    )
    cv2.addWeighted(
        overlay,
        0.62,
        canvas,
        0.38,
        0,
        canvas,
    )

    if calibrating:
        attention_color = (0, 215, 255)
        status_text = "CALIBRATING"
    elif attention >= 75:
        attention_color = (70, 232, 140)
        status_text = "FOCUSED"
    elif attention >= 45:
        attention_color = (0, 215, 255)
        status_text = "PARTIALLY FOCUSED"
    else:
        attention_color = (75, 95, 255)
        status_text = "DISTRACTED"

    put_text(
        canvas,
        "GAZE",
        (38, 108),
        0.35,
        (135, 147, 163),
        1,
    )
    put_text(
        canvas,
        gaze,
        (38, 136),
        0.58,
        (236, 239, 243),
        2,
    )

    put_text(
        canvas,
        "EYES",
        (250, 108),
        0.35,
        (135, 147, 163),
        1,
    )
    put_text(
        canvas,
        "OPEN" if eyes_open else "CLOSED",
        (250, 136),
        0.58,
        (70, 232, 140) if eyes_open else (75, 95, 255),
        2,
    )

    put_text(
        canvas,
        "ATTENTION",
        (38, 165),
        0.35,
        (135, 147, 163),
        1,
    )
    put_text(
        canvas,
        f"{attention:.0f}%",
        (38, 194),
        0.64,
        attention_color,
        2,
    )

    put_text(
        canvas,
        status_text,
        (250, 190),
        0.42,
        attention_color,
        2,
    )

    # --------------------------------------------------------
    # Right rail
    # --------------------------------------------------------
    px = w + 24

    put_text(
        canvas,
        "ATTENTION SCORE",
        (px, 91),
        0.40,
        (140, 153, 169),
        1,
    )

    put_text(
        canvas,
        f"{attention:.0f}%",
        (px, 137),
        1.02,
        attention_color,
        2,
    )

    put_text(
        canvas,
        status_text,
        (px, 160),
        0.43,
        attention_color,
        2,
    )

    cv2.rectangle(
        canvas,
        (px, 177),
        (w + panel_w - 22, 190),
        (52, 57, 67),
        -1,
    )

    fill = int(
        (panel_w - 46) *
        float(np.clip(attention, 0, 100)) /
        100.0
    )

    if fill > 0:
        cv2.rectangle(
            canvas,
            (px, 177),
            (px + fill, 190),
            attention_color,
            -1,
        )

    # Current emotion card
    put_text(
        canvas,
        "CURRENT EXPRESSION",
        (px, 226),
        0.40,
        (140, 153, 169),
        1,
    )

    emotion_colors = {
        "Happy": (75, 225, 80),
        "Sad": (175, 135, 255),
        "Angry": (70, 80, 255),
        "Surprised": (0, 220, 255),
        "Fear": (215, 95, 255),
        "Disgust": (80, 205, 135),
        "Neutral": (215, 220, 225),
        "No Face": (165, 170, 180),
    }

    ecolor = emotion_colors.get(
        emotion,
        (230, 230, 230),
    )

    put_text(
        canvas,
        emotion,
        (px, 264),
        0.78,
        ecolor,
        2,
    )

    put_text(
        canvas,
        f"Confidence  {confidence:.0f}%",
        (px, 286),
        0.38,
        (195, 200, 208),
        1,
    )

    # Expression profile
    put_text(
        canvas,
        "EXPRESSION PROFILE",
        (px, 325),
        0.40,
        (140, 153, 169),
        1,
    )

    colors = [
        (75, 220, 75),
        (170, 130, 255),
        (70, 80, 255),
        (0, 220, 255),
        (215, 95, 255),
        (80, 205, 135),
        (190, 195, 200),
    ]

    y = 349

    for name, color in zip(EMOTIONS, colors):
        value = float(
            probabilities.get(name, 0.0)
        )

        put_text(
            canvas,
            name,
            (px, y + 13),
            0.36,
            (210, 215, 222),
            1,
        )

        bx = px + 76
        bw = 174

        cv2.rectangle(
            canvas,
            (bx, y + 2),
            (bx + bw, y + 14),
            (49, 54, 63),
            -1,
        )

        fill = int(
            bw * np.clip(value, 0, 100) / 100.0
        )

        if fill > 0:
            cv2.rectangle(
                canvas,
                (bx, y + 2),
                (bx + fill, y + 14),
                color,
                -1,
            )

        put_text(
            canvas,
            f"{value:.0f}%",
            (bx + bw + 8, y + 13),
            0.33,
            (190, 195, 202),
            1,
        )

        y += 28

    # Footer
    footer_y = h - 42

    cv2.rectangle(
        canvas,
        (0, footer_y),
        (w + panel_w, h),
        (19, 22, 29),
        -1,
    )

    cv2.line(
        canvas,
        (0, footer_y),
        (w + panel_w, footer_y),
        (53, 59, 68),
        1,
    )

    put_text(
        canvas,
        f"Faces  {faces}",
        (20, h - 16),
        0.38,
        (178, 184, 192),
        1,
    )

    put_text(
        canvas,
        f"FPS  {fps:.1f}",
        (112, h - 16),
        0.38,
        (178, 184, 192),
        1,
    )

    put_text(
        canvas,
        "Q Exit   •   F Points   •   R Recalibrate",
        (w + 20, h - 16),
        0.35,
        (165, 170, 178),
        1,
    )

    return canvas


# ============================================================
# CAMERA
# ============================================================

cap = None
camera_index_used = None

for index in (0, 1, 2):
    candidate = cv2.VideoCapture(index, cv2.CAP_DSHOW)

    if candidate.isOpened():
        cap = candidate
        camera_index_used = index
        break

    candidate.release()

if cap is None:
    raise RuntimeError(
        "No working camera found. Close Camera, WhatsApp, Teams, "
        "Chrome camera tabs, Zoom, etc. and run again."
    )

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_H)
cap.set(cv2.CAP_PROP_FPS, 30)

print("=" * 60)
print("             EMOTION MONITOR")
print("=" * 60)
print(f"Camera: {camera_index_used}")
print("Look directly at the camera for the first ~1.5 seconds.")
print("The program will automatically calibrate your normal gaze.")
print("")
print("Q / ESC = Exit")
print("F       = Toggle face points")
print("R       = Recalibrate")
print("=" * 60)

# ============================================================
# STATE
# ============================================================

attention = AttentionTracker()

emotion_probs = {name: 0.0 for name in EMOTIONS}
emotion_probs["Neutral"] = 100.0

current_emotion = "Neutral"
confidence = 100.0
current_gaze = "No face"
eyes_open = False

show_points = False

timestamp_ms = 0

last_time = time.perf_counter()
fps = 0.0

# Short grace period when the detector temporarily loses a face.
last_face_time = None
NO_FACE_GRACE = 0.35

focused_frames = 0
partial_frames = 0
distracted_frames = 0
session_start = time.time()

# ============================================================
# MAIN LOOP
# ============================================================

try:
    while True:
        ok, frame = cap.read()

        if not ok:
            print("Camera frame unavailable.")
            break

        frame = cv2.flip(frame, 1)

        frame_h, frame_w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )

        timestamp_ms += 33

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms,
        )

        faces_count = (
            len(result.face_landmarks)
            if result.face_landmarks
            else 0
        )

        face_index = choose_face(result)

        now = time.perf_counter()

        if face_index is not None:
            face = result.face_landmarks[face_index]
            last_face_time = now

            features = get_blendshapes(
                result,
                face_index,
            )

            # ---------- Eye geometry ----------
            gaze_h, gaze_v, eye_opening = get_gaze_geometry(face)

            eyes_open = eye_open_state(
                features,
                eye_opening,
            )

            # ---------- Head ----------
            yaw, pitch = head_orientation(face)

            # ---------- Attention ----------
            attention_value = attention.calculate(
                gaze_h,
                gaze_v,
                yaw,
                pitch,
                eyes_open,
                now,
            )

            # ---------- Gaze label ----------
            current_gaze = attention.gaze_text(
                gaze_h,
                gaze_v,
                yaw,
                pitch,
            )

            # ---------- Emotion ----------
            (
                emotion_probs,
                current_emotion,
                confidence,
            ) = emotion_probabilities(
                features,
                emotion_probs,
            )

            if show_points:
                draw_points(frame, face)

        else:
            # Do not instantly collapse because MediaPipe missed
            # one frame. This prevents visible flickering.
            if (
                last_face_time is None
                or now - last_face_time > NO_FACE_GRACE
            ):
                attention.score = lerp(
                    attention.score,
                    0.0,
                    0.12,
                )
                attention.last_valid_score = attention.score

                current_emotion = "No Face"
                confidence = 0.0
                current_gaze = "No face"
                eyes_open = False

                emotion_probs = {
                    name: 0.0
                    for name in EMOTIONS
                }

        current_attention = float(
            np.clip(attention.score, 0.0, 100.0)
        )

        # ---------- Session statistics ----------
        if face_index is not None:
            if attention.calibrating:
                pass
            elif current_attention >= 75:
                focused_frames += 1
            elif current_attention >= 45:
                partial_frames += 1
            else:
                distracted_frames += 1

        # ---------- FPS ----------
        current_time = time.perf_counter()
        dt = current_time - last_time

        if dt > 0:
            instant = 1.0 / dt
            fps = lerp(
                fps,
                min(60.0, instant),
                0.18,
            )

        last_time = current_time

        status_for_ui = (
            "CALIBRATING"
            if attention.calibrating
            else (
                "FOCUSED"
                if current_attention >= 75
                else (
                    "PARTIALLY FOCUSED"
                    if current_attention >= 45
                    else "DISTRACTED"
                )
            )
        )

        dashboard = build_dashboard(
            frame=frame,
            attention=current_attention,
            status=status_for_ui,
            gaze=current_gaze,
            eyes_open=eyes_open,
            emotion=current_emotion,
            confidence=confidence,
            probabilities=emotion_probs,
            faces=faces_count,
            fps=fps,
            show_points=show_points,
            calibrating=attention.calibrating,
        )

        cv2.imshow(WINDOW, dashboard)

        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord("q"), ord("Q")):
            break

        if key in (ord("f"), ord("F")):
            show_points = not show_points

        if key in (ord("r"), ord("R")):
            attention.reset()

            emotion_probs = {
                name: 0.0
                for name in EMOTIONS
            }
            emotion_probs["Neutral"] = 100.0

            current_emotion = "Neutral"
            confidence = 100.0

            print("\nRecalibrating...")
            print("Look directly at the camera for ~1.5 seconds.")

finally:
    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()

    elapsed = max(0.001, time.time() - session_start)

    total = (
        focused_frames
        + partial_frames
        + distracted_frames
    )

    if total > 0:
        focused_pct = focused_frames / total * 100.0
        partial_pct = partial_frames / total * 100.0
        distracted_pct = distracted_frames / total * 100.0
    else:
        focused_pct = partial_pct = distracted_pct = 0.0

    print("\n" + "=" * 60)
    print("                 SESSION SUMMARY")
    print("=" * 60)
    print(f"Session time : {elapsed:.1f} sec")
    print(f"Focused      : {focused_frames} ({focused_pct:.1f}%)")
    print(f"Partial      : {partial_frames} ({partial_pct:.1f}%)")
    print(f"Distracted   : {distracted_frames} ({distracted_pct:.1f}%)")
    print("=" * 60)
    print("Emotion Monitor stopped.")
