"""
Extended API Blueprint Module
Exposes REST endpoints for Sessions, Analytics, Reports, and ML without modifying the Video Analyzer.
"""

from flask import Blueprint, jsonify, request, Response
from analyzer_adapter import VideoAnalyzerAdapter
from session_manager import SessionManager
from analytics import compute_session_analytics, compute_overall_trends
from reports import generate_text_summary, generate_html_report, export_csv_timeline
from ml_engine import ml_engine
import db

extended_bp = Blueprint("extended_bp", __name__)

adapter = VideoAnalyzerAdapter()
session_manager = SessionManager(adapter)


def init_adapter(monitor_instance):
    """Binds the active Video Analyzer monitor instance to the adapter."""
    adapter.set_monitor(monitor_instance)


# ============================================================
# Session Endpoints
# ============================================================

@extended_bp.route("/api/session/start", methods=["POST"])
def api_session_start():
    data = request.get_json(silent=True) or {}
    title = data.get("title", "Focus Session")
    res = session_manager.start_session(title=title)
    return jsonify(res)


@extended_bp.route("/api/session/pause", methods=["POST"])
def api_session_pause():
    return jsonify(session_manager.pause_session())


@extended_bp.route("/api/session/resume", methods=["POST"])
def api_session_resume():
    return jsonify(session_manager.resume_session())


@extended_bp.route("/api/session/stop", methods=["POST"])
def api_session_stop():
    return jsonify(session_manager.stop_session())


@extended_bp.route("/api/session/status", methods=["GET"])
def api_session_status():
    return jsonify(session_manager.get_current_status())


@extended_bp.route("/api/sessions", methods=["GET"])
def api_list_sessions():
    limit = int(request.args.get("limit", 50))
    offset = int(request.args.get("offset", 0))
    return jsonify(db.list_sessions(limit=limit, offset=offset))


@extended_bp.route("/api/sessions/<session_uuid>", methods=["GET"])
def api_get_session(session_uuid):
    session = db.get_session_by_uuid(session_uuid)
    if not session:
        return jsonify({"status": "error", "message": "Session not found"}), 404
    return jsonify(session)


@extended_bp.route("/api/sessions/<session_uuid>/delete", methods=["POST"])
def api_delete_session(session_uuid):
    success = db.delete_session(session_uuid)
    return jsonify({"status": "success" if success else "error"})


# ============================================================
# Analytics & Reports Endpoints
# ============================================================

@extended_bp.route("/api/analytics/<session_uuid>", methods=["GET"])
def api_session_analytics(session_uuid):
    analytics = compute_session_analytics(session_uuid)
    if not analytics:
        return jsonify({"status": "error", "message": "Session not found"}), 404
    return jsonify(analytics)


@extended_bp.route("/api/analytics/trends", methods=["GET"])
def api_analytics_trends():
    return jsonify(compute_overall_trends())


@extended_bp.route("/api/reports/<session_uuid>/text", methods=["GET"])
def api_report_text(session_uuid):
    text = generate_text_summary(session_uuid)
    return Response(text, mimetype="text/plain")


@extended_bp.route("/api/reports/<session_uuid>/html", methods=["GET"])
def api_report_html(session_uuid):
    html = generate_html_report(session_uuid)
    return Response(html, mimetype="text/html")


@extended_bp.route("/api/reports/<session_uuid>/csv", methods=["GET"])
def api_report_csv(session_uuid):
    csv_data = export_csv_timeline(session_uuid)
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment;filename=session_{session_uuid}.csv"}
    )


# ============================================================
# ML Endpoints
# ============================================================

@extended_bp.route("/api/ml/info", methods=["GET"])
def api_ml_info():
    return jsonify(ml_engine.get_info())


@extended_bp.route("/api/ml/train", methods=["POST"])
def api_ml_train():
    return jsonify(ml_engine.train_model())


@extended_bp.route("/api/ml/predict", methods=["POST"])
def api_ml_predict():
    data = request.get_json(silent=True) or {}
    if not data:
        # Predict using current latest analyzer sample
        latest = adapter.get_latest_result()
        if latest:
            data = {
                "gaze_h": latest.gaze_h,
                "gaze_v": latest.gaze_v,
                "head_yaw": latest.head_yaw,
                "head_pitch": latest.head_pitch,
                "eyes_open": latest.eyes_open,
                "blink_rate": latest.blinks_per_min,
                "emotion_confidence": latest.expression_confidence,
                "attention_score": latest.attention_score
            }
    return jsonify(ml_engine.predict(data))


# ============================================================
# Cloud Browser Frame Processing Endpoint
# ============================================================

@extended_bp.route("/api/process_frame", methods=["POST"])
def api_process_frame():
    try:
        import base64
        import numpy as np
        import cv2
        import mediapipe as mp
        from app import (
            monitor, landmarker, choose_face, get_blendshapes,
            get_gaze_geometry, head_orientation, eye_open_state,
            emotion_probabilities, CAMERA_W, CAMERA_H
        )

        data = request.get_json(silent=True) or {}
        img_b64 = data.get("image", "")
        if not img_b64:
            return jsonify(monitor.get_metrics())
        
        if "," in img_b64:
            img_b64 = img_b64.split(",", 1)[1]
        
        img_bytes = base64.b64decode(img_b64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        raw_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if raw_frame is None:
            return jsonify(monitor.get_metrics())
        
        frame = cv2.resize(raw_frame, (CAMERA_W, CAMERA_H))
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        now_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, now_ms)
        
        face_idx = choose_face(result)
        faces_count = len(result.face_landmarks) if result.face_landmarks else 0
        now = time.time()

        with monitor.lock:
            monitor.faces_count = faces_count
            monitor.total_frames += 1
            monitor.fps = 15.0

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

                monitor.current_status = status
            else:
                monitor.current_status = "NO FACE"
                monitor.current_attention = 0.0
                monitor.distracted_frames += 1
        
        return jsonify(monitor.get_metrics())
    except Exception as e:
        return jsonify(monitor.get_metrics())
