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
