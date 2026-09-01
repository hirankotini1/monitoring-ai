import cv2
import time
import numpy as np
from collections import deque

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "models/face_landmarker.task"
CAMERA_INDEX = 0

WIDTH = 960
HEIGHT = 720


# ============================================================
# MEDIAPIPE FACE LANDMARKER
# ============================================================

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,

    min_face_detection_confidence=0.55,
    min_face_presence_confidence=0.55,
    min_tracking_confidence=0.55,

    output_face_blendshapes=True,
    output_facial_transformation_matrixes=False
)

landmarker = vision.FaceLandmarker.create_from_options(options)


# ============================================================
# CAMERA
# ============================================================

camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not camera.isOpened():
    camera = cv2.VideoCapture(CAMERA_INDEX)

if not camera.isOpened():
    print("❌ Could not open camera.")
    raise SystemExit

camera.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)


# ============================================================
# HELPERS
# ============================================================

def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, float(value)))


def smooth(old, new, amount):
    return old + (new - old) * amount


def get_bs(blendshapes, name):
    return float(blendshapes.get(name, 0.0))


def average_bs(blendshapes, name1, name2):
    return (
        get_bs(blendshapes, name1) +
        get_bs(blendshapes, name2)
    ) / 2.0


# ============================================================
# GET BLENDSHAPES
# ============================================================

def get_blendshapes(result):

    if not result.face_blendshapes:
        return {}

    return {
        category.category_name: float(category.score)
        for category in result.face_blendshapes[0]
    }


# ============================================================
# EYE / IRIS TRACKING
# ============================================================

# Left eye
LEFT_OUTER = 33
LEFT_INNER = 133
LEFT_TOP = 159
LEFT_BOTTOM = 145

# Right eye
RIGHT_INNER = 362
RIGHT_OUTER = 263
RIGHT_TOP = 386
RIGHT_BOTTOM = 374

LEFT_IRIS = range(468, 473)
RIGHT_IRIS = range(473, 478)


def eye_position(points, outer, inner, top, bottom, iris_indices):

    outer_point = np.array([
        points[outer].x,
        points[outer].y
    ], dtype=np.float32)

    inner_point = np.array([
        points[inner].x,
        points[inner].y
    ], dtype=np.float32)

    top_point = np.array([
        points[top].x,
        points[top].y
    ], dtype=np.float32)

    bottom_point = np.array([
        points[bottom].x,
        points[bottom].y
    ], dtype=np.float32)

    iris_points = np.array([
        [points[i].x, points[i].y]
        for i in iris_indices
    ], dtype=np.float32)

    iris_center = np.mean(
        iris_points,
        axis=0
    )

    # --------------------------------------------------------
    # Horizontal position
    # --------------------------------------------------------

    horizontal = inner_point - outer_point

    horizontal_length = np.linalg.norm(horizontal)

    if horizontal_length < 0.00001:
        return 0.5, 0.5, 0.0

    horizontal_ratio = np.dot(
        iris_center - outer_point,
        horizontal
    ) / (
        horizontal_length ** 2
    )

    # --------------------------------------------------------
    # Vertical position
    # --------------------------------------------------------

    vertical = bottom_point - top_point

    vertical_length = np.linalg.norm(vertical)

    if vertical_length < 0.00001:
        return clamp(horizontal_ratio), 0.5, 0.0

    vertical_ratio = np.dot(
        iris_center - top_point,
        vertical
    ) / (
        vertical_length ** 2
    )

    # --------------------------------------------------------
    # Eye openness
    # --------------------------------------------------------

    eye_width = np.linalg.norm(horizontal)

    eye_height = np.linalg.norm(vertical)

    openness = eye_height / max(
        eye_width,
        0.00001
    )

    return (
        clamp(horizontal_ratio),
        clamp(vertical_ratio),
        openness
    )


def calculate_gaze(points):

    left_x, left_y, left_open = eye_position(
        points,
        LEFT_OUTER,
        LEFT_INNER,
        LEFT_TOP,
        LEFT_BOTTOM,
        LEFT_IRIS
    )

    right_x, right_y, right_open = eye_position(
        points,
        RIGHT_INNER,
        RIGHT_OUTER,
        RIGHT_TOP,
        RIGHT_BOTTOM,
        RIGHT_IRIS
    )

    # If one eye is unreliable, use the other.
    if left_open < 0.12 and right_open >= 0.12:

        return (
            right_x,
            right_y,
            right_open
        )

    if right_open < 0.12 and left_open >= 0.12:

        return (
            left_x,
            left_y,
            left_open
        )

    return (
        (left_x + right_x) / 2,
        (left_y + right_y) / 2,
        (left_open + right_open) / 2
    )


# ============================================================
# GAZE CLASSIFICATION
# ============================================================

def classify_gaze(x, y):

    # Vertical first
    if y < 0.34:
        return "Looking Up"

    if y > 0.66:
        return "Looking Down"

    # Horizontal
    if x < 0.35:
        return "Looking Left"

    if x > 0.65:
        return "Looking Right"

    return "Looking Center"


# ============================================================
# EMOTION DETECTION
# ============================================================

def calculate_emotions(bs):

    smile = average_bs(
        bs,
        "mouthSmileLeft",
        "mouthSmileRight"
    )

    frown = average_bs(
        bs,
        "mouthFrownLeft",
        "mouthFrownRight"
    )

    brow_down = average_bs(
        bs,
        "browDownLeft",
        "browDownRight"
    )

    brow_inner = get_bs(
        bs,
        "browInnerUp"
    )

    brow_outer = average_bs(
        bs,
        "browOuterUpLeft",
        "browOuterUpRight"
    )

    eye_wide = average_bs(
        bs,
        "eyeWideLeft",
        "eyeWideRight"
    )

    eye_squint = average_bs(
        bs,
        "eyeSquintLeft",
        "eyeSquintRight"
    )

    jaw_open = get_bs(
        bs,
        "jawOpen"
    )

    mouth_press = average_bs(
        bs,
        "mouthPressLeft",
        "mouthPressRight"
    )

    mouth_stretch = average_bs(
        bs,
        "mouthStretchLeft",
        "mouthStretchRight"
    )

    nose_sneer = average_bs(
        bs,
        "noseSneerLeft",
        "noseSneerRight"
    )

    upper_lip = average_bs(
        bs,
        "mouthUpperUpLeft",
        "mouthUpperUpRight"
    )

    lower_lip = average_bs(
        bs,
        "mouthLowerDownLeft",
        "mouthLowerDownRight"
    )

    cheek_squint = average_bs(
        bs,
        "cheekSquintLeft",
        "cheekSquintRight"
    )


    # ========================================================
    # HAPPY
    # ========================================================

    happy = (
        smile * 1.60
        + cheek_squint * 0.35
        + eye_squint * 0.20
        - frown * 0.45
        - brow_down * 0.20
    )


    # ========================================================
    # SAD
    # ========================================================

    sad = (
        frown * 1.40
        + brow_inner * 0.75
        + mouth_press * 0.25
        + lower_lip * 0.15
        - smile * 0.40
    )


    # ========================================================
    # ANGRY
    # ========================================================

    angry = (
        brow_down * 1.40
        + mouth_press * 0.50
        + eye_squint * 0.35
        + frown * 0.25
        - smile * 0.50
    )


    # ========================================================
    # SURPRISED
    # ========================================================

    surprised = (
        eye_wide * 1.35
        + jaw_open * 1.20
        + brow_inner * 0.85
        + brow_outer * 0.35
    )


    # ========================================================
    # FEAR
    # ========================================================

    fear = (
        eye_wide * 1.00
        + brow_inner * 0.90
        + jaw_open * 0.65
        + mouth_stretch * 0.40
        + mouth_press * 0.15
        - smile * 0.30
    )


    # ========================================================
    # DISGUST
    # ========================================================

    disgust = (
        nose_sneer * 1.50
        + upper_lip * 1.15
        + mouth_press * 0.25
        + cheek_squint * 0.20
        - smile * 0.30
    )


    # ========================================================
    # CONFLICT PENALTIES
    # ========================================================

    happy -= frown * 0.30
    happy -= brow_down * 0.20

    sad -= smile * 0.35

    angry -= smile * 0.40

    surprised -= smile * 0.15

    fear -= smile * 0.20

    disgust -= smile * 0.25


    scores = {

        "Happy": max(0, happy),

        "Sad": max(0, sad),

        "Angry": max(0, angry),

        "Surprised": max(0, surprised),

        "Fear": max(0, fear),

        "Disgust": max(0, disgust)
    }


    # ========================================================
    # NEUTRAL
    # ========================================================

    strongest_expression = max(
        scores.values()
    )

    neutral = max(
        0.0,
        0.55 - strongest_expression * 0.60
    )

    scores["Neutral"] = neutral

    return scores


# ============================================================
# CHOOSE EMOTION
# ============================================================

def choose_emotion(history):

    if not history:

        return (
            "Neutral",
            1.0
        )

    emotions = history[0].keys()

    averaged = {}

    for emotion in emotions:

        averaged[emotion] = sum(
            item.get(emotion, 0.0)
            for item in history
        ) / len(history)


    total = sum(
        averaged.values()
    )

    if total <= 0:

        return (
            "Neutral",
            1.0
        )


    probabilities = {
        emotion: value / total
        for emotion, value in averaged.items()
    }


    best_emotion = max(
        probabilities,
        key=probabilities.get
    )

    confidence = probabilities[
        best_emotion
    ]


    # Don't randomly call weak expressions emotions.
    if (
        best_emotion != "Neutral"
        and confidence < 0.32
    ):

        return (
            "Neutral",
            probabilities.get(
                "Neutral",
                0.5
            )
        )


    return (
        best_emotion,
        confidence
    )


# ============================================================
# ATTENTION
# ============================================================

def calculate_attention(
    gaze_x,
    gaze_y,
    eyes_open
):

    horizontal_distance = abs(
        gaze_x - 0.50
    ) / 0.50

    vertical_distance = abs(
        gaze_y - 0.50
    ) / 0.50


    # Vertical gaze is slightly more important.
    gaze_distance = (
        horizontal_distance * 0.40
        + vertical_distance * 0.60
    )


    gaze_score = 1.0 - clamp(
        gaze_distance
    )


    if eyes_open:

        eye_score = 1.0

    else:

        eye_score = 0.05


    attention = (
        gaze_score * 0.85
        + eye_score * 0.15
    )


    return clamp(
        attention
    ) * 100


# ============================================================
# DRAW TEXT
# ============================================================

def text(
    frame,
    message,
    x,
    y,
    color,
    size=0.65,
    thickness=2
):

    cv2.putText(
        frame,
        message,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        size,
        color,
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# VARIABLES
# ============================================================

smooth_gaze_x = 0.50
smooth_gaze_y = 0.50

smooth_attention = 50.0

gaze_history = deque(
    maxlen=5
)

emotion_history = deque(
    maxlen=12
)

current_gaze = "Looking Center"

current_emotion = "Neutral"

emotion_confidence = 0.0

show_points = True

timestamp = 0

previous_time = time.perf_counter()


# ============================================================
# START
# ============================================================

print()
print("=" * 45)
print("          EMOTION MONITOR")
print("=" * 45)
print("ESC = Exit")
print("E   = Toggle face points")
print("=" * 45)
print()


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    success, frame = camera.read()

    if not success:

        print("Could not read camera frame.")
        continue


    # Mirror webcam
    frame = cv2.flip(
        frame,
        1
    )


    # ========================================================
    # FPS
    # ========================================================

    now = time.perf_counter()

    fps = 1.0 / max(
        now - previous_time,
        0.001
    )

    previous_time = now


    # ========================================================
    # MEDIAPIPE
    # ========================================================

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    timestamp += 33


    try:

        result = landmarker.detect_for_video(
            mp_image,
            timestamp
        )

    except Exception as error:

        print(
            "Detection error:",
            error
        )

        continue


    # ========================================================
    # NO FACE
    # ========================================================

    if not result.face_landmarks:

        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (0, 0),
            (500, 180),
            (0, 0, 0),
            -1
        )

        frame = cv2.addWeighted(
            overlay,
            0.65,
            frame,
            0.35,
            0
        )

        text(
            frame,
            "NO FACE DETECTED",
            25,
            55,
            (0, 0, 255),
            0.85,
            3
        )

        text(
            frame,
            "Move into camera view",
            25,
            95,
            (230, 230, 230),
            0.55,
            1
        )

        text(
            frame,
            f"FPS: {fps:.1f}",
            25,
            130,
            (200, 200, 200),
            0.50,
            1
        )

        cv2.imshow(
            "Emotion Monitor",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

        if key == ord("e"):
            show_points = not show_points

        continue


    # ========================================================
    # FACE LANDMARKS
    # ========================================================

    points = result.face_landmarks[0]


    # ========================================================
    # GAZE
    # ========================================================

    gaze_x, gaze_y, eye_open_ratio = calculate_gaze(
        points
    )


    smooth_gaze_x = smooth(
        smooth_gaze_x,
        gaze_x,
        0.25
    )

    smooth_gaze_y = smooth(
        smooth_gaze_y,
        gaze_y,
        0.25
    )


    gaze_history.append(
        (
            smooth_gaze_x,
            smooth_gaze_y
        )
    )


    average_gaze_x = sum(
        value[0]
        for value in gaze_history
    ) / len(gaze_history)


    average_gaze_y = sum(
        value[1]
        for value in gaze_history
    ) / len(gaze_history)


    candidate_gaze = classify_gaze(
        average_gaze_x,
        average_gaze_y
    )


    # Small hysteresis prevents flickering.
    if candidate_gaze == "Looking Center":

        if (
            abs(average_gaze_x - 0.50) < 0.12
            and
            abs(average_gaze_y - 0.50) < 0.12
        ):

            current_gaze = "Looking Center"

    else:

        current_gaze = candidate_gaze


    # ========================================================
    # EYES
    # ========================================================

    blendshapes = get_blendshapes(
        result
    )


    blink_left = get_bs(
        blendshapes,
        "eyeBlinkLeft"
    )

    blink_right = get_bs(
        blendshapes,
        "eyeBlinkRight"
    )


    blink_average = (
        blink_left +
        blink_right
    ) / 2


    eyes_open = blink_average < 0.45


    # ========================================================
    # ATTENTION
    # ========================================================

    raw_attention = calculate_attention(
        average_gaze_x,
        average_gaze_y,
        eyes_open
    )


    smooth_attention = smooth(
        smooth_attention,
        raw_attention,
        0.12
    )


    # ========================================================
    # EMOTIONS
    # ========================================================

    emotion_scores = calculate_emotions(
        blendshapes
    )


    emotion_history.append(
        emotion_scores
    )


    emotion, confidence = choose_emotion(
        emotion_history
    )


    # Don't rapidly switch between emotions.
    if emotion == current_emotion:

        emotion_confidence = confidence

    else:

        if (
            confidence >= 0.36
            or
            current_emotion == "Neutral"
        ):

            current_emotion = emotion
            emotion_confidence = confidence


    # ========================================================
    # ATTENTION STATUS
    # ========================================================

    if smooth_attention >= 75:

        status = "FOCUSED"
        status_color = (
            0,
            255,
            0
        )

    elif smooth_attention >= 50:

        status = "PARTIALLY FOCUSED"
        status_color = (
            0,
            220,
            255
        )

    else:

        status = "DISTRACTED"
        status_color = (
            0,
            100,
            255
        )


    # ========================================================
    # FACE POINTS
    # ========================================================

    if show_points:

        important_points = [
            33,
            133,
            362,
            263,
            468,
            473,
            1,
            61,
            291
        ]


        for index in important_points:

            if index >= len(points):
                continue


            px = int(
                points[index].x *
                frame.shape[1]
            )

            py = int(
                points[index].y *
                frame.shape[0]
            )


            cv2.circle(
                frame,
                (px, py),
                3,
                (0, 255, 0),
                -1
            )


    # ========================================================
    # UI PANEL
    # ========================================================

    overlay = frame.copy()


    cv2.rectangle(
        overlay,
        (0, 0),
        (500, 285),
        (0, 0, 0),
        -1
    )


    frame = cv2.addWeighted(
        overlay,
        0.60,
        frame,
        0.40,
        0
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    text(
        frame,
        f"Gaze: {current_gaze}",
        28,
        45,
        (0, 255, 0),
        0.70,
        2
    )


    text(
        frame,
        "Eyes: Open" if eyes_open else "Eyes: Closed",
        28,
        82,
        (0, 255, 0)
        if eyes_open
        else
        (0, 100, 255),
        0.70,
        2
    )


    text(
        frame,
        f"Attention: {smooth_attention:.0f}%",
        28,
        120,
        (0, 255, 255),
        0.70,
        2
    )


    text(
        frame,
        f"Status: {status}",
        28,
        158,
        status_color,
        0.65,
        2
    )


    text(
        frame,
        f"Expression: {current_emotion}",
        28,
        198,
        (255, 0, 255),
        0.70,
        2
    )


    text(
        frame,
        f"Confidence: {emotion_confidence * 100:.0f}%",
        28,
        235,
        (220, 220, 220),
        0.55,
        1
    )


    text(
        frame,
        f"FPS: {fps:.1f}",
        28,
        265,
        (190, 190, 190),
        0.50,
        1
    )


    text(
        frame,
        "ESC: Exit   E: Toggle points",
        15,
        frame.shape[0] - 15,
        (220, 220, 220),
        0.45,
        1
    )


    # ========================================================
    # SHOW
    # ========================================================

    cv2.imshow(
        "Emotion Monitor",
        frame
    )


    key = cv2.waitKey(1) & 0xFF


    if key == 27:
        break


    if key == ord("e"):
        show_points = not show_points


# ============================================================
# CLEANUP
# ============================================================

camera.release()

landmarker.close()

cv2.destroyAllWindows()

print()
print("========================================")
print("Emotion Monitor stopped.")
print("========================================")