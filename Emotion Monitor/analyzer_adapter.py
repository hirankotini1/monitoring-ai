"""
Video Analyzer Adapter Module
Provides a clean, locked interface to consume outputs from the existing Video Analyzer.
Adheres strictly to the protection boundary: does not recalculate or alter any vision outputs.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class AnalyzerResult:
    """Authoritative snapshot produced by the locked Video Analyzer."""
    timestamp: float
    attention_score: float
    attention_state: str
    calibrating: bool
    detected_expression: str
    expression_confidence: float
    expression_probabilities: Dict[str, float] = field(default_factory=dict)
    gaze_direction: str = "Looking Center"
    gaze_h: float = 0.5
    gaze_v: float = 0.5
    head_yaw: float = 0.0
    head_pitch: float = 0.0
    eyes_open: bool = True
    faces_detected: int = 1
    fps: float = 0.0
    distraction_duration: float = 0.0
    total_frames: int = 0
    blink_count: int = 0
    blinks_per_min: float = 0.0
    raw_payload: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], timestamp: Optional[float] = None) -> "AnalyzerResult":
        """Converts raw Video Analyzer dictionary (e.g. from /api/metrics or monitor.get_metrics()) into AnalyzerResult."""
        now = timestamp or time.time()
        gaze_coords = data.get("gaze_coordinates", {})
        head_pose = data.get("head_pose", {})
        session_info = data.get("session", {})

        return cls(
            timestamp=now,
            attention_score=float(data.get("attention_score", 100.0)),
            attention_state=str(data.get("status", "FOCUSED")),
            calibrating=bool(data.get("calibrating", False)),
            detected_expression=str(data.get("emotion", "Neutral")),
            expression_confidence=float(data.get("emotion_confidence", 100.0)),
            expression_probabilities=data.get("emotion_probabilities", {}),
            gaze_direction=str(data.get("gaze_direction", "Looking Center")),
            gaze_h=float(gaze_coords.get("h", 0.5)),
            gaze_v=float(gaze_coords.get("v", 0.5)),
            head_yaw=float(head_pose.get("yaw", 0.0)),
            head_pitch=float(head_pose.get("pitch", 0.0)),
            eyes_open=bool(data.get("eyes_open", True)),
            faces_detected=int(data.get("faces_detected", 1)),
            fps=float(data.get("fps", 0.0)),
            distraction_duration=float(data.get("distraction_duration", 0.0)),
            total_frames=int(session_info.get("total_frames", 0)),
            blink_count=int(session_info.get("blink_count", 0)),
            blinks_per_min=float(session_info.get("blinks_per_min", 0.0)),
            raw_payload=data
        )


class VideoAnalyzerAdapter:
    """Adapter to read outputs from the Video Analyzer without touching its internals."""
    
    def __init__(self, monitor_engine=None):
        self._monitor = monitor_engine

    def set_monitor(self, monitor_engine):
        self._monitor = monitor_engine

    def get_latest_result(self) -> Optional[AnalyzerResult]:
        if self._monitor is None:
            return None
        try:
            metrics_dict = self._monitor.get_metrics()
            return AnalyzerResult.from_dict(metrics_dict)
        except Exception as e:
            return None
