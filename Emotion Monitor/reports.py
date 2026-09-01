"""
Reports Engine Module
Generates human-readable, JSON, CSV, and printable HTML session reports
from the authoritative Video Analyzer analytics.
"""

import io
import csv
import json
import time
from typing import Dict, Any, Optional
from analytics import compute_session_analytics
import db


def generate_text_summary(session_uuid: str) -> str:
    """Generates an ASCII / Text summary report of a session."""
    data = compute_session_analytics(session_uuid)
    if not data:
        return "Session report not found."

    s = data["session"]
    dur_min = round(data["duration_seconds"] / 60.0, 1)
    states = data["state_distribution"]
    exprs = data["expression_distribution"]
    gazes = data["gaze_distribution"]
    events = data["events_summary"]

    lines = [
        "=" * 55,
        f"       SESSION REPORT: {s.get('title', 'Focus Session').upper()}",
        "=" * 55,
        f"Session ID        : {session_uuid}",
        f"Duration          : {dur_min} minutes ({int(data['duration_seconds'])} seconds)",
        f"Average Attention : {data['average_attention']}%",
        f"Status            : {s.get('status', 'COMPLETED')}",
        "-" * 55,
        "ATTENTION BREAKDOWN:",
        f"  • Focused   : {states.get('Focused', 0)}%",
        f"  • Partial   : {states.get('Partial', 0)}%",
        f"  • Distracted: {states.get('Distracted', 0)}%",
        f"  • No Face   : {states.get('No Face', 0)}%",
        "-" * 55,
        "FACIAL EXPRESSION ESTIMATES:",
    ]

    for name, pct in sorted(exprs.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"  • {name:12}: {pct}%")

    lines.extend([
        "-" * 55,
        "GAZE DIRECTION DISTRIBUTION:",
    ])

    for dir_name, pct in sorted(gazes.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"  • {dir_name:12}: {pct}%")

    lines.extend([
        "-" * 55,
        "ENGAGEMENT & STABILITY:",
        f"  • Longest Focus Streak : {data['longest_focus_streak_seconds']}s",
        f"  • Distraction Events   : {events['distraction_events_count']} (Total {events['total_distracted_seconds']}s)",
        f"  • Prolonged Closures   : {events['prolonged_closures_count']}",
        f"  • Head Pose Stability  : {data['head_pose_stats']['stability_index']}/100",
        "=" * 55,
    ])

    return "\n".join(lines)


def export_csv_timeline(session_uuid: str) -> str:
    """Exports the entire time-series metric timeline of a session as CSV."""
    metrics = db.get_session_metrics(session_uuid)
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Header
    writer.writerow([
        "RelativeTimeSec", "AttentionScore", "AttentionState",
        "DetectedExpression", "ExpressionConfidence", "GazeDirection",
        "GazeH", "GazeV", "HeadYaw", "HeadPitch", "EyesOpen", "FacePresent"
    ])

    for m in metrics:
        writer.writerow([
            m["relative_time_sec"],
            m["attention_score"],
            m["attention_state"],
            m["detected_expression"],
            m["expression_confidence"],
            m["gaze_direction"],
            m["gaze_h"],
            m["gaze_v"],
            m["head_yaw"],
            m["head_pitch"],
            m["eyes_open"],
            m["face_present"]
        ])

    return output.getvalue()


def generate_html_report(session_uuid: str) -> str:
    """Generates a standalone, beautiful printable HTML report."""
    data = compute_session_analytics(session_uuid)
    if not data:
        return "<h3>Session not found</h3>"

    s = data["session"]
    dur_min = round(data["duration_seconds"] / 60.0, 1)
    states = data["state_distribution"]
    exprs = data["expression_distribution"]
    gazes = data["gaze_distribution"]
    events = data["events_summary"]
    stats = data["head_pose_stats"]

    expr_rows = "".join([
        f"<tr><td>{name}</td><td><b>{pct}%</b></td></tr>"
        for name, pct in sorted(exprs.items(), key=lambda x: x[1], reverse=True)
    ])

    gaze_rows = "".join([
        f"<tr><td>{dir_name}</td><td><b>{pct}%</b></td></tr>"
        for dir_name, pct in sorted(gazes.items(), key=lambda x: x[1], reverse=True)
    ])

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Session Report - {s.get('title', 'Focus Session')}</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #0f172a; color: #f8fafc; padding: 30px; }}
        .report-card {{ background: #1e293b; max-width: 800px; margin: 0 auto; border-radius: 12px; padding: 32px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }}
        h1 {{ margin-top: 0; color: #38bdf8; font-size: 24px; border-bottom: 2px solid #334155; padding-bottom: 12px; }}
        .grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin: 20px 0; }}
        .metric-box {{ background: #0f172a; padding: 16px; border-radius: 8px; text-align: center; }}
        .metric-box .label {{ font-size: 12px; color: #94a3b8; text-transform: uppercase; }}
        .metric-box .val {{ font-size: 24px; font-weight: bold; color: #10b981; margin-top: 4px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #334155; }}
        th {{ background: #0f172a; color: #94a3b8; }}
        .sec-title {{ font-size: 16px; color: #cbd5e1; margin-top: 24px; margin-bottom: 8px; font-weight: 600; }}
        .footer {{ font-size: 12px; color: #64748b; margin-top: 30px; text-align: center; border-top: 1px solid #334155; padding-top: 12px; }}
    </style>
</head>
<body>
    <div class="report-card">
        <h1>📊 Session Summary: {s.get('title', 'Focus Session')}</h1>
        <p style="color: #94a3b8; font-size: 13px;">UUID: {session_uuid} | Date: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(s.get('start_time', time.time())))}</p>
        
        <div class="grid">
            <div class="metric-box">
                <div class="label">Duration</div>
                <div class="val" style="color: #38bdf8;">{dur_min} min</div>
            </div>
            <div class="metric-box">
                <div class="label">Avg Attention</div>
                <div class="val">{data['average_attention']}%</div>
            </div>
            <div class="metric-box">
                <div class="label">Stability Index</div>
                <div class="val" style="color: #a855f7;">{stats['stability_index']}/100</div>
            </div>
        </div>

        <div class="sec-title">🎯 Attention State Distribution</div>
        <table>
            <tr><th>State</th><th>Percentage</th></tr>
            <tr><td>Focused (≥75%)</td><td><b>{states.get('Focused', 0)}%</b></td></tr>
            <tr><td>Partial Focus (45-74%)</td><td><b>{states.get('Partial', 0)}%</b></td></tr>
            <tr><td>Distracted (&lt;45%)</td><td><b>{states.get('Distracted', 0)}%</b></td></tr>
            <tr><td>No Face Detected</td><td><b>{states.get('No Face', 0)}%</b></td></tr>
        </table>

        <div class="grid" style="grid-template-columns: 1fr 1fr; margin-top: 20px;">
            <div>
                <div class="sec-title">🎭 Detected Expressions</div>
                <table>
                    <tr><th>Expression</th><th>Share</th></tr>
                    {expr_rows}
                </table>
            </div>
            <div>
                <div class="sec-title">👁️ Gaze Directions</div>
                <table>
                    <tr><th>Direction</th><th>Share</th></tr>
                    {gaze_rows}
                </table>
            </div>
        </div>

        <div class="sec-title">⚡ Key Session Metrics</div>
        <table>
            <tr><td>Longest Continuous Focus Streak</td><td><b>{data['longest_focus_streak_seconds']} seconds</b></td></tr>
            <tr><td>Distraction Events Count</td><td><b>{events['distraction_events_count']} times ({events['total_distracted_seconds']}s total)</b></td></tr>
            <tr><td>Prolonged Eye Closures</td><td><b>{events['prolonged_closures_count']} times</b></td></tr>
            <tr><td>Head Yaw / Pitch Std Dev</td><td><b>{stats['yaw_std']} / {stats['pitch_std']}</b></td></tr>
        </table>

        <div class="footer">
            Generated by AI Emotion & Attention Neural Monitoring System — Authoritative Vision Pipeline
        </div>
    </div>
</body>
</html>"""
    return html
