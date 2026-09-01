"""
Analytics Engine Module
Computes statistical aggregations, distributions, streak lengths, and historical trends
directly from authoritative Video Analyzer output data.
"""

import math
from typing import Dict, List, Any, Optional
import db


def compute_session_analytics(session_uuid: str) -> Optional[Dict[str, Any]]:
    """Calculates comprehensive analytics for a single recorded session."""
    session = db.get_session_by_uuid(session_uuid)
    if not session:
        return None
    
    metrics = db.get_session_metrics(session_uuid)
    events = db.get_session_events(session_uuid)

    if not metrics:
        return {
            "session": session,
            "total_samples": 0,
            "duration_seconds": session.get("duration_seconds", 0),
            "average_attention": session.get("avg_attention", 100.0),
            "state_distribution": {"Focused": 0, "Partial": 0, "Distracted": 0, "No Face": 0},
            "expression_distribution": {},
            "gaze_distribution": {},
            "head_pose_stats": {"yaw_mean": 0.0, "pitch_mean": 0.0, "yaw_std": 0.0, "pitch_std": 0.0, "stability_index": 100.0},
            "events_summary": {"distractions": len(events), "prolonged_closures": 0, "total_distracted_seconds": 0.0},
            "longest_focus_streak_seconds": 0,
            "timeline": []
        }

    total_samples = len(metrics)
    
    # 1. Attention State Breakdown
    state_counts = {"Focused": 0, "Partial": 0, "Distracted": 0, "No Face": 0}
    scores = []
    
    # 2. Expression & Gaze Breakdown
    expression_counts: Dict[str, int] = {}
    gaze_counts: Dict[str, int] = {}
    
    # 3. Head Pose tracking
    yaws = []
    pitches = []
    
    # 4. Streak Calculation
    current_focus_streak = 0
    longest_focus_streak = 0
    
    timeline = []

    for m in metrics:
        score = float(m["attention_score"])
        scores.append(score)
        state = m["attention_state"]
        expr = m["detected_expression"]
        gaze = m["gaze_direction"]
        
        yaws.append(float(m.get("head_yaw") or 0.0))
        pitches.append(float(m.get("head_pitch") or 0.0))

        if state == "FOCUSED":
            state_counts["Focused"] += 1
            current_focus_streak += 1
            if current_focus_streak > longest_focus_streak:
                longest_focus_streak = current_focus_streak
        elif state == "PARTIAL":
            state_counts["Partial"] += 1
            current_focus_streak = 0
        elif state == "DISTRACTED":
            state_counts["Distracted"] += 1
            current_focus_streak = 0
        elif state == "NO FACE":
            state_counts["No Face"] += 1
            current_focus_streak = 0

        expression_counts[expr] = expression_counts.get(expr, 0) + 1
        
        # Normalize gaze direction names for distribution
        gaze_clean = gaze.replace("Looking ", "").strip()
        gaze_counts[gaze_clean] = gaze_counts.get(gaze_clean, 0) + 1

        timeline.append({
            "t": round(m["relative_time_sec"], 1),
            "score": round(score, 1),
            "state": state,
            "expression": expr,
            "gaze": gaze
        })

    # Percentages
    state_distribution_pct = {
        k: round((v / total_samples) * 100.0, 1) for k, v in state_counts.items()
    }
    
    expression_distribution_pct = {
        k: round((v / total_samples) * 100.0, 1) for k, v in expression_counts.items()
    }
    
    gaze_distribution_pct = {
        k: round((v / total_samples) * 100.0, 1) for k, v in gaze_counts.items()
    }

    # Head Pose Stability (Standard deviation & Index)
    yaw_mean = sum(yaws) / len(yaws) if yaws else 0.0
    pitch_mean = sum(pitches) / len(pitches) if pitches else 0.0
    yaw_var = sum((y - yaw_mean) ** 2 for y in yaws) / len(yaws) if yaws else 0.0
    pitch_var = sum((p - pitch_mean) ** 2 for p in pitches) / len(pitches) if pitches else 0.0
    yaw_std = math.sqrt(yaw_var)
    pitch_std = math.sqrt(pitch_var)
    
    # Stability: lower jitter = higher stability (0-100)
    jitter = (yaw_std + pitch_std) * 0.5
    stability_index = max(0.0, min(100.0, round(100.0 - (jitter * 120.0), 1)))

    # Events Summary
    distraction_events = [e for e in events if e["event_type"] == "DISTRACTION_EVENT"]
    closure_events = [e for e in events if e["event_type"] == "PROLONGED_EYE_CLOSURE"]
    total_distracted_time = sum(float(e["duration_seconds"]) for e in distraction_events)

    return {
        "session": session,
        "total_samples": total_samples,
        "duration_seconds": session.get("duration_seconds", total_samples),
        "average_attention": round(sum(scores) / len(scores), 1) if scores else 100.0,
        "state_distribution": state_distribution_pct,
        "expression_distribution": expression_distribution_pct,
        "gaze_distribution": gaze_distribution_pct,
        "head_pose_stats": {
            "yaw_mean": round(yaw_mean, 3),
            "pitch_mean": round(pitch_mean, 3),
            "yaw_std": round(yaw_std, 3),
            "pitch_std": round(pitch_std, 3),
            "stability_index": stability_index
        },
        "events_summary": {
            "distraction_events_count": len(distraction_events),
            "prolonged_closures_count": len(closure_events),
            "total_distracted_seconds": round(total_distracted_time, 1),
            "avg_distraction_duration": round(total_distracted_time / max(1, len(distraction_events)), 1)
        },
        "longest_focus_streak_seconds": longest_focus_streak,
        "timeline": timeline
    }


def compute_overall_trends() -> Dict[str, Any]:
    """Computes global historical aggregate across all recorded sessions."""
    sessions = db.list_sessions(limit=100)
    global_stats = db.get_global_aggregate_stats()

    if not sessions:
        return {
            "total_sessions": 0,
            "total_duration_minutes": 0,
            "overall_avg_attention": 100.0,
            "peak_focus_score": 100.0,
            "total_distractions_logged": 0,
            "longest_focus_streak_ever": 0,
            "state_distribution": {"Focused": 0, "Partial": 0, "Distracted": 0, "No Face": 0},
            "expression_distribution": {"Happy": 0, "Neutral": 100, "Sad": 0, "Angry": 0, "Surprised": 0, "Fear": 0},
            "gaze_distribution": {"Center": 100, "Right": 0, "Left": 0, "Up": 0, "Down": 0},
            "sessions_history": []
        }

    total_dur = sum(s.get("duration_seconds", 0) for s in sessions)
    avg_scores = [s.get("avg_attention", 100.0) for s in sessions if s.get("avg_attention") is not None]
    overall_avg = sum(avg_scores) / len(avg_scores) if avg_scores else 100.0
    peak_score = max(avg_scores) if avg_scores else 100.0

    # Total Samples across DB
    states = global_stats.get("states", {})
    total_samples = max(1, sum(states.values()))
    
    focused_cnt = states.get("FOCUSED", 0)
    partial_cnt = states.get("PARTIAL", 0)
    distracted_cnt = states.get("DISTRACTED", 0)
    noface_cnt = states.get("NO FACE", 0)

    state_distribution = {
        "Focused": round((focused_cnt / total_samples) * 100.0, 1),
        "Partial": round((partial_cnt / total_samples) * 100.0, 1),
        "Distracted": round((distracted_cnt / total_samples) * 100.0, 1),
        "No Face": round((noface_cnt / total_samples) * 100.0, 1)
    }

    # Expressions
    exprs = global_stats.get("expressions", {})
    total_expr_samples = max(1, sum(exprs.values()))
    expression_distribution = {
        k: round((v / total_expr_samples) * 100.0, 1) for k, v in exprs.items()
    } if exprs else {"Neutral": 100.0}

    # Gazes
    gazes = global_stats.get("gazes", {})
    total_gaze_samples = max(1, sum(gazes.values()))
    gaze_clean_counts = {}
    for g, cnt in gazes.items():
        name = g.replace("Looking ", "").strip()
        gaze_clean_counts[name] = gaze_clean_counts.get(name, 0) + cnt
    
    gaze_distribution = {
        k: round((v / total_gaze_samples) * 100.0, 1) for k, v in gaze_clean_counts.items()
    } if gaze_clean_counts else {"Center": 100.0}

    # Events
    events = global_stats.get("events", {})
    distraction_info = events.get("DISTRACTION_EVENT", {"count": 0, "total_duration": 0.0})
    closure_info = events.get("PROLONGED_EYE_CLOSURE", {"count": 0, "total_duration": 0.0})

    history_points = []
    for s in reversed(sessions):
        history_points.append({
            "session_uuid": s["session_uuid"],
            "title": s.get("title") or "Focus Session",
            "start_time": s["start_time"],
            "avg_attention": s.get("avg_attention", 100.0),
            "duration_seconds": s.get("duration_seconds", 0),
            "dominant_expression": s.get("dominant_expression", "Neutral"),
            "focused_pct": s.get("focused_pct", 0.0),
            "distracted_pct": s.get("distracted_pct", 0.0)
        })

    return {
        "total_sessions": len(sessions),
        "total_duration_minutes": round(total_dur / 60.0, 1),
        "overall_avg_attention": round(overall_avg, 1),
        "peak_focus_score": round(peak_score, 1),
        "total_distractions_logged": distraction_info.get("count", 0),
        "total_distracted_time_seconds": round(distraction_info.get("total_duration", 0.0), 1),
        "total_prolonged_closures": closure_info.get("count", 0),
        "state_distribution": state_distribution,
        "expression_distribution": expression_distribution,
        "gaze_distribution": gaze_distribution,
        "sessions_history": history_points
    }
