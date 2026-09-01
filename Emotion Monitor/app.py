import os
import sys
import time
import json
import threading
from collections import deque
import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, Response, render_template, jsonify, request

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "face_landmarker.task")

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"Missing model: {MODEL_PATH}")

# Flask App
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"), static_folder=os.path.join(BASE_DIR, "static"))

# ---------------- MediaPipe Setup ----------------
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

# ---------------- Landmarks & Constants ----------------
CAMERA_W = 960
CAMERA_H = 540

EMOTIONS = [
    "Happy",
    "Sad",
    "Angry",
    "Surprised",
    "Fear",
    "Neutral",
]

LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

LEFT_EYE_IN = 133
LEFT_EYE_OUT = 33
LEFT_EYE_UP = 159
LEFT_EYE_DOWN = 145

RIGHT_EYE_IN = 362
RIGHT_EYE_OUT = 263
RIGHT_EYE_UP = 386
RIGHT_EYE_DOWN = 374

NOSE = 1
CHIN = 152
LEFT_FACE_SIDE = 234
RIGHT_FACE_SIDE = 454

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
    if not result.face_blendshapes or face_index >= len(result.face_blendshapes):
        return data
    for c in result.face_blendshapes[face_index]:
        name = getattr(c, "category_name", None)
        score = getattr(c, "score", None)
        if name is not None and score is not None:
            data[normalize_name(name)] = float(score)
    return data

def face_box(face):
    xs = [p.x for p in face]
    ys = [p.y for p in face]
    return min(xs), min(ys), max(xs), max(ys)

def choose_face(result):
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
        score = area * 2.0 - center_distance * 0.25
        if score > best_score:
            best_score = score
            best = i
    return best

def eye_ratio(face, iris_indices, inner_idx, outer_idx, upper_idx, lower_idx):
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
    opening = float(np.linalg.norm(vert_vec))
    return horizontal, vertical, opening

def get_gaze_geometry(face):
    lh, lv, lopen = eye_ratio(face, LEFT_IRIS, LEFT_EYE_IN, LEFT_EYE_OUT, LEFT_EYE_UP, LEFT_EYE_DOWN)
    rh, rv, ropen = eye_ratio(face, RIGHT_IRIS, RIGHT_EYE_IN, RIGHT_EYE_OUT, RIGHT_EYE_UP, RIGHT_EYE_DOWN)
    return safe_mean([lh, rh]), safe_mean([lv, rv]), safe_mean([lopen, ropen])

def head_orientation(face):
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

class AttentionTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.calibrating = False
        self.samples = []
        self.center_h = 0.5
        self.center_v = 0.5
        self.center_yaw = 0.0
        self.center_pitch = 0.0
        self.raw_history = deque(maxlen=8)
        self.gaze_history = deque(maxlen=5)
        self.score = 100.0
        self.last_valid_score = 100.0
        self.closed_since = None

    def add_calibration_sample(self, h, v, yaw, pitch):
        self.samples.append((h, v, yaw, pitch))
        if len(self.samples) >= 12:
            arr = np.array(self.samples, dtype=np.float32)
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

        # Eyes closed handling (rapid drop past a micro-blink)
        if not eyes_open:
            if self.closed_since is None:
                self.closed_since = now
            closed_duration = now - self.closed_since
            if closed_duration < 0.25:
                raw = self.last_valid_score
            else:
                # Drops to 0% when eyes stay closed
                raw = max(0.0, 100.0 * max(0.0, 1.0 - (closed_duration - 0.25) * 3.5))
        else:
            self.closed_since = None

            # Dynamic auto-centering when focused on screen
            if self.score > 80.0:
                self.center_h = lerp(self.center_h, h, 0.005)
                self.center_v = lerp(self.center_v, v, 0.005)
                self.center_yaw = lerp(self.center_yaw, yaw, 0.005)
                self.center_pitch = lerp(self.center_pitch, pitch, 0.005)

            # Balanced 4-way screen deviations
            dh = abs(h - self.center_h)
            dv = abs(v - self.center_v)
            dyaw = abs(yaw - self.center_yaw)
            dpitch = abs(pitch - self.center_pitch)

            dh_eff = max(0.0, dh - 0.05) / 0.16
            dv_eff = max(0.0, dv - 0.05) / 0.14
            dyaw_eff = max(0.0, dyaw - 0.08) / 0.24
            dpitch_eff = max(0.0, dpitch - 0.07) / 0.20

            eye_dist = np.sqrt(dh_eff**2 + dv_eff**2)
            head_dist = np.sqrt(dyaw_eff**2 + dpitch_eff**2)

            eye_attn = 100.0 * np.exp(-1.4 * (eye_dist ** 1.6))
            head_attn = 100.0 * np.exp(-1.4 * (head_dist ** 1.6))

            raw = 0.50 * eye_attn + 0.50 * head_attn

        raw = float(np.clip(raw, 0.0, 100.0))
        self.raw_history.append(raw)
        median_raw = float(np.median(self.raw_history))
        self.score = lerp(self.score, median_raw, 0.40)
        self.score = float(np.clip(self.score, 0.0, 100.0))

        if eyes_open:
            self.last_valid_score = self.score
        return self.score

    def gaze_text(self, h, v, yaw, pitch):
        if self.calibrating:
            return "Calibrating..."
        dh = h - self.center_h
        dv = v - self.center_v
        dyaw = yaw - self.center_yaw
        dpitch = pitch - self.center_pitch

        # Balanced 4-way gaze direction
        if abs(dh) < 0.08 and abs(dv) < 0.06 and abs(dyaw) < 0.12 and abs(dpitch) < 0.10:
            label = "Looking Center"
        elif abs(dyaw) >= abs(dpitch) and abs(dyaw) > 0.12:
            label = "Looking Right" if dyaw < 0 else "Looking Left"
        elif dpitch > 0.10 or dv > 0.06:
            label = "Looking Down"
        elif dpitch < -0.10 or dv < -0.06:
            label = "Looking Up"
        elif abs(dh) > 0.08:
            label = "Looking Right" if dh < 0 else "Looking Left"
        else:
            label = "Looking Center"

        self.gaze_history.append(label)
        if len(self.gaze_history) >= 3:
            counts = {}
            for item in self.gaze_history:
                counts[item] = counts.get(item, 0) + 1
            label = max(counts, key=counts.get)
        return label

def emotion_probabilities(features, old_probs):
    smile = feature(features, "mouthSmileLeft", "mouthSmileRight")
    dimple = feature(features, "mouthDimpleLeft", "mouthDimpleRight")
    cheek_squint = feature(features, "cheekSquintLeft", "cheekSquintRight")
    eye_squint = feature(features, "eyeSquintLeft", "eyeSquintRight")

    brow_down = feature(features, "browDownLeft", "browDownRight")
    brow_inner_up = feature(features, "browInnerUp")
    brow_outer_up = feature(features, "browOuterUpLeft", "browOuterUpRight")

    frown = feature(features, "mouthFrownLeft", "mouthFrownRight")
    press = feature(features, "mouthPressLeft", "mouthPressRight")
    stretch = feature(features, "mouthStretchLeft", "mouthStretchRight")
    funnel = feature(features, "mouthFunnel")
    pucker = feature(features, "mouthPucker")
    jaw_open = feature(features, "jawOpen")
    eye_wide = feature(features, "eyeWideLeft", "eyeWideRight")
    sneer = feature(features, "noseSneerLeft", "noseSneerRight")
    upper_lip = feature(features, "mouthUpperLipRaiseLeft", "mouthUpperLipRaiseRight")

    # 1. HAPPY: Smile driven
    happy_score = 0.0
    if smile > 0.08 and brow_down < 0.28:
        happy_score = (smile - 0.06) * 4.5 + (cheek_squint * 1.2 if smile > 0.12 else 0.0) + (dimple * 0.8)
        happy_score -= (brow_down * 2.0) + (frown * 1.5)

    # 2. ANGRY: Requires genuine furrowing (> 0.28) or moderate furrowing (> 0.20) + mouth grimace/frown/sneer
    angry_score = 0.0
    if (brow_down > 0.28) or (brow_down > 0.20 and (frown > 0.12 or press > 0.14 or sneer > 0.10)):
        angry_score = (brow_down - 0.18) * 4.2 + (frown * 2.0) + (press * 1.5) + (sneer * 1.4)
        angry_score -= (smile * 3.5) + (brow_inner_up * 0.8)

    # 3. SAD: Raised inner eyebrows or mouth frown + inner brow
    sad_score = 0.0
    if (brow_inner_up > 0.16 and eye_wide < 0.20) or (frown > 0.12 and (brow_inner_up > 0.08 or press > 0.10)):
        sad_score = (brow_inner_up - 0.10) * 3.4 + (frown * 3.2) + (press * 1.6)
        sad_score -= (smile * 3.5) + (eye_wide * 1.5) + (brow_down * 1.2)

    # 4. SURPRISED: Direct open mouth trigger (jawOpen / mouthFunnel) or wide eyes + raised brows
    surprised_score = 0.0
    if (jaw_open > 0.12) or (funnel > 0.15) or (eye_wide > 0.10 and (brow_outer_up > 0.08 or brow_inner_up > 0.10)):
        surprised_score = (jaw_open * 4.2) + (funnel * 2.5) + (eye_wide * 3.0) + (brow_outer_up * 2.5) + (brow_inner_up * 1.5)
        surprised_score -= (frown * 2.0) + (brow_down * 2.5)

    # 5. FEAR: Wide eyes + raised inner brows + mouth stretch
    fear_score = 0.0
    if (eye_wide > 0.10 and (brow_inner_up > 0.10 or stretch > 0.08)) or (stretch > 0.12 and (brow_inner_up > 0.08 or eye_wide > 0.08)):
        fear_score = (eye_wide - 0.05) * 3.6 + (brow_inner_up - 0.06) * 3.2 + (stretch - 0.05) * 4.2 + (funnel * 2.0)
        fear_score -= (smile * 3.0) + (brow_down * 1.8)

    raw_scores = {
        "Happy": max(0.0, happy_score),
        "Angry": max(0.0, angry_score),
        "Sad": max(0.0, sad_score),
        "Surprised": max(0.0, surprised_score),
        "Fear": max(0.0, fear_score),
    }

    max_expression = max(raw_scores.values())
    total_expression = sum(raw_scores.values())

    if max_expression < 0.12:
        target_probs = {k: 0.0 for k in EMOTIONS[:-1]}
        target_probs["Neutral"] = 100.0
    else:
        neutral_rem = max(0.0, 0.35 - max_expression * 0.4)
        total = total_expression + neutral_rem
        target_probs = {k: (raw_scores[k] / total * 100.0) for k in EMOTIONS[:-1]}
        target_probs["Neutral"] = (neutral_rem / total * 100.0)

    # Smooth transition
    result = {}
    for name in EMOTIONS:
        alpha = 0.42 if target_probs[name] > 20 else 0.25
        result[name] = lerp(float(old_probs.get(name, 0.0)), target_probs[name], alpha)

    total_pct = sum(result.values())
    result = {k: round(v / total_pct * 100.0, 1) for k, v in result.items()}

    ranked = sorted(result.items(), key=lambda x: x[1], reverse=True)
    top_name, top_val = ranked[0]
    confidence = min(98.0, max(55.0, top_val * 1.05))

    return result, top_name, round(confidence, 1)

def eye_open_state(features, geometry_opening):
    blink = feature(features, "eyeBlinkLeft", "eyeBlinkRight")
    if blink > 0.45:
        return False
    return True

# ---------------- Global Monitor State ----------------
class MonitorEngine:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = True
        self.show_points = True
        self.attention_tracker = AttentionTracker()
        
        self.session_start = time.time()
        self.total_frames = 0
        self.focused_frames = 0
        self.partial_frames = 0
        self.distracted_frames = 0
        self.blink_count = 0
        self.was_eyes_open = True
        
        self.current_frame_jpeg = None
        self.fps = 0.0
        
        self.emotion_probs = {e: (100.0 if e == "Neutral" else 0.0) for e in EMOTIONS}
        self.current_emotion = "Neutral"
        self.emotion_confidence = 100.0
        self.current_attention = 100.0
        self.current_gaze = "Looking Center"
        self.eyes_open = True
        self.faces_count = 0
        self.status = "FOCUSED"
        self.gaze_h = 0.5
        self.gaze_v = 0.5
        self.head_yaw = 0.0
        self.head_pitch = 0.0
        self.calibrating = False
        self.distraction_duration = 0.0
        self.last_distracted_time = None

    def reset_session(self):
        with self.lock:
            self.session_start = time.time()
            self.total_frames = 0
            self.focused_frames = 0
            self.partial_frames = 0
            self.distracted_frames = 0
            self.blink_count = 0
            self.attention_tracker.reset()
            self.emotion_probs = {e: (100.0 if e == "Neutral" else 0.0) for e in EMOTIONS}
            self.current_emotion = "Neutral"
            self.emotion_confidence = 100.0

    def recalibrate(self):
        with self.lock:
            self.attention_tracker.reset()
            self.emotion_probs = {e: (100.0 if e == "Neutral" else 0.0) for e in EMOTIONS}
            self.current_emotion = "Neutral"
            self.emotion_confidence = 100.0

    def toggle_mesh(self):
        with self.lock:
            self.show_points = not self.show_points
            return self.show_points

    def get_metrics(self):
        with self.lock:
            elapsed = max(0.1, time.time() - self.session_start)
            total = max(1, self.total_frames)
            focused_pct = round(self.focused_frames / total * 100.0, 1)
            partial_pct = round(self.partial_frames / total * 100.0, 1)
            distracted_pct = round(self.distracted_frames / total * 100.0, 1)

            return {
                "attention_score": round(self.current_attention, 1),
                "status": self.status,
                "calibrating": self.calibrating,
                "emotion": self.current_emotion,
                "emotion_confidence": round(self.emotion_confidence, 1),
                "emotion_probabilities": {k: round(v, 1) for k, v in self.emotion_probs.items()},
                "gaze_direction": self.current_gaze,
                "gaze_coordinates": {
                    "h": round(self.gaze_h, 3),
                    "v": round(self.gaze_v, 3),
                    "center_h": round(self.attention_tracker.center_h, 3),
                    "center_v": round(self.attention_tracker.center_v, 3),
                },
                "head_pose": {
                    "yaw": round(self.head_yaw, 3),
                    "pitch": round(self.head_pitch, 3),
                },
                "eyes_open": self.eyes_open,
                "faces_detected": self.faces_count,
                "fps": round(self.fps, 1),
                "mesh_visible": self.show_points,
                "distraction_duration": round(self.distraction_duration, 1),
                "session": {
                    "duration_seconds": int(elapsed),
                    "total_frames": self.total_frames,
                    "focused_frames": self.focused_frames,
                    "partial_frames": self.partial_frames,
                    "distracted_frames": self.distracted_frames,
                    "focused_pct": focused_pct,
                    "partial_pct": partial_pct,
                    "distracted_pct": distracted_pct,
                    "blink_count": self.blink_count,
                    "blinks_per_min": round(self.blink_count / (elapsed / 60.0), 1) if elapsed >= 5 else 0.0,
                }
            }

monitor = MonitorEngine()

# Initialize camera cleanly once at startup
cap = None
for idx in (0, 1, 2):
    candidate = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if candidate.isOpened():
        cap = candidate
        print(f"[OK] Successfully opened camera index {idx} with DirectShow")
        break
    candidate.release()

if cap is None:
    print("[INFO] DirectShow probe failed, falling back to default backend...")
    cap = cv2.VideoCapture(0)

def video_worker():
    global cap
    timestamp_ms = 0
    fps_time = time.perf_counter()
    fps_count = 0
    calculated_fps = 0.0

    while monitor.running:
        if cap is None or not cap.isOpened():
            time.sleep(0.1)
            continue

        ok, raw_frame = cap.read()
        now = time.time()

        if not ok or raw_frame is None:
            time.sleep(0.01)
            continue

        fps_count += 1
        perf_now = time.perf_counter()
        if perf_now - fps_time >= 1.0:
            calculated_fps = fps_count / (perf_now - fps_time)
            fps_count = 0
            fps_time = perf_now

        # Resize and flip
        frame = cv2.resize(raw_frame, (CAMERA_W, CAMERA_H))
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        timestamp_ms += 33
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        face_idx = choose_face(result)
        faces_count = len(result.face_landmarks) if result.face_landmarks else 0

        with monitor.lock:
            monitor.faces_count = faces_count
            monitor.fps = calculated_fps
            monitor.total_frames += 1

            if face_idx is not None and result.face_landmarks:
                face = result.face_landmarks[face_idx]
                features = get_blendshapes(result, face_idx)

                h, v, opening = get_gaze_geometry(face)
                yaw, pitch = head_orientation(face)
                eyes_open = eye_open_state(features, opening)

                if monitor.was_eyes_open and not eyes_open:
                    monitor.blink_count += 1
                monitor.was_eyes_open = eyes_open

                monitor.gaze_h = h
                monitor.gaze_v = v
                monitor.head_yaw = yaw
                monitor.head_pitch = pitch
                monitor.eyes_open = eyes_open

                attention_score = monitor.attention_tracker.calculate(h, v, yaw, pitch, eyes_open, now)
                gaze_label = monitor.attention_tracker.gaze_text(h, v, yaw, pitch)
                calibrating = monitor.attention_tracker.calibrating

                probs, emotion_label, confidence = emotion_probabilities(features, monitor.emotion_probs)
                monitor.emotion_probs = probs
                monitor.current_emotion = emotion_label
                monitor.emotion_confidence = confidence

                monitor.current_attention = attention_score
                monitor.current_gaze = gaze_label
                monitor.calibrating = calibrating

                if calibrating:
                    status = "CALIBRATING"
                elif attention_score >= 75:
                    status = "FOCUSED"
                    monitor.focused_frames += 1
                    monitor.last_distracted_time = None
                    monitor.distraction_duration = 0.0
                elif attention_score >= 45:
                    status = "PARTIAL"
                    monitor.partial_frames += 1
                    monitor.last_distracted_time = None
                    monitor.distraction_duration = 0.0
                else:
                    status = "DISTRACTED"
                    monitor.distracted_frames += 1
                    if monitor.last_distracted_time is None:
                        monitor.last_distracted_time = now
                    monitor.distraction_duration = now - monitor.last_distracted_time

                monitor.status = status

                if monitor.show_points:
                    fh, fw = frame.shape[:2]
                    for i, p in enumerate(face):
                        if i % 6 == 0:
                            px = int(p.x * fw)
                            py = int(p.y * fh)
                            if 0 <= px < fw and 0 <= py < fh:
                                cv2.circle(frame, (px, py), 1, (0, 255, 128), -1)
                    
                    for i in LEFT_IRIS + RIGHT_IRIS:
                        p = face[i]
                        px = int(p.x * fw)
                        py = int(p.y * fh)
                        cv2.circle(frame, (px, py), 2, (0, 215, 255), -1)

            else:
                monitor.status = "NO FACE"
                monitor.calibrating = False
                monitor.current_attention = max(0.0, monitor.current_attention - 3.0)
                monitor.current_gaze = "No Face Detected"
                monitor.distracted_frames += 1
                if monitor.last_distracted_time is None:
                    monitor.last_distracted_time = now
                monitor.distraction_duration = now - monitor.last_distracted_time

            ret_enc, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if ret_enc:
                monitor.current_frame_jpeg = jpeg.tobytes()

        time.sleep(0.005)

video_thread = threading.Thread(target=video_worker, daemon=True)
video_thread.start()

@app.route("/")
def index():
    return render_template("index.html")

def gen_frames():
    while True:
        with monitor.lock:
            frame_bytes = monitor.current_frame_jpeg
        if frame_bytes is not None:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
        time.sleep(0.03)

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/metrics")
def api_metrics():
    return jsonify(monitor.get_metrics())

@app.route("/api/recalibrate", methods=["POST"])
def api_recalibrate():
    monitor.recalibrate()
    return jsonify({"status": "success", "message": "Attention tracker recalibrated."})

@app.route("/api/toggle_mesh", methods=["POST"])
def api_toggle_mesh():
    visible = monitor.toggle_mesh()
    return jsonify({"status": "success", "mesh_visible": visible})

@app.route("/api/reset_session", methods=["POST"])
def api_reset_session():
    monitor.reset_session()
    return jsonify({"status": "success", "message": "Session statistics reset."})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 60)
    print("      EMOTION & ATTENTION MONITOR - LOCAL HOST")
    print("=" * 60)
    print(f"Server starting on http://localhost:{port}")
    print(f"Open your browser and navigate to: http://localhost:{port}")
    print("=" * 60)
    try:
        app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
    finally:
        monitor.running = False
        if cap is not None:
            cap.release()
        landmarker.close()
