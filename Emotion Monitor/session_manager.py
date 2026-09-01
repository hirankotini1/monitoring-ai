"""
Session Manager Module
Orchestrates monitoring sessions, background time-series sampling, event logging, and database persistence.
Consumes results strictly from the Video Analyzer Adapter.
"""

import time
import uuid
import threading
from typing import Optional, Dict, Any, List
from analyzer_adapter import VideoAnalyzerAdapter, AnalyzerResult
import db


class SessionManager:
    """Manages recording sessions and background sampling from Video Analyzer."""
    
    def __init__(self, adapter: VideoAnalyzerAdapter):
        self.adapter = adapter
        self.lock = threading.Lock()
        
        self.active_session_uuid: Optional[str] = None
        self.session_title: str = "Focus Session"
        self.session_start_time: Optional[float] = None
        self.is_paused: bool = False
        self.pause_start_time: Optional[float] = None
        self.total_paused_duration: float = 0.0
        
        # In-memory accumulators for current active session
        self.sampled_scores: List[float] = []
        self.focused_count: int = 0
        self.partial_count: int = 0
        self.distracted_count: int = 0
        self.noface_count: int = 0
        self.expression_counts: Dict[str, int] = {}
        
        # Event Tracking
        self.in_distraction_event: bool = False
        self.distraction_event_start: Optional[float] = None
        self.distraction_event_count: int = 0
        
        self.in_prolonged_closure: bool = False
        self.closure_start: Optional[float] = None
        self.closure_event_count: int = 0
        
        # Background worker for 1Hz sampling
        self.sampling_thread: Optional[threading.Thread] = None
        self.running: bool = True
        self._start_sampler()

    def _start_sampler(self):
        self.sampling_thread = threading.Thread(target=self._sampling_loop, daemon=True)
        self.sampling_thread.start()

    def _sampling_loop(self):
        while self.running:
            try:
                time.sleep(1.0)
                with self.lock:
                    if not self.active_session_uuid or self.is_paused:
                        continue
                    
                    result = self.adapter.get_latest_result()
                    if result is None:
                        continue
                    
                    self._record_sample(result)
            except Exception as e:
                # Keep loop resilient
                pass

    def _record_sample(self, result: AnalyzerResult):
        now = time.time()
        rel_time = now - self.session_start_time - self.total_paused_duration
        if rel_time < 0:
            rel_time = 0.0

        # Accumulate metrics
        score = result.attention_score
        state = result.attention_state
        expr = result.detected_expression
        
        self.sampled_scores.append(score)
        self.expression_counts[expr] = self.expression_counts.get(expr, 0) + 1
        
        if state == "FOCUSED":
            self.focused_count += 1
        elif state == "PARTIAL":
            self.partial_count += 1
        elif state == "DISTRACTED":
            self.distracted_count += 1
        elif state == "NO FACE":
            self.noface_count += 1

        # 1. Distraction Event Lifecycle
        if state in ("DISTRACTED", "NO FACE"):
            if not self.in_distraction_event:
                self.in_distraction_event = True
                self.distraction_event_start = now
                self.distraction_event_count += 1
        else:
            if self.in_distraction_event:
                self.in_distraction_event = False
                dur = now - (self.distraction_event_start or now)
                if dur >= 2.0:
                    db.insert_attention_event(
                        session_uuid=self.active_session_uuid,
                        event_type="DISTRACTION_EVENT",
                        start_time=self.distraction_event_start or now,
                        end_time=now,
                        duration_seconds=round(dur, 1),
                        severity="HIGH" if dur >= 5.0 else "MEDIUM",
                        metadata={"trigger_state": state, "score": score}
                    )
                self.distraction_event_start = None

        # 2. Prolonged Eye Closure Lifecycle
        if not result.eyes_open:
            if not self.in_prolonged_closure:
                self.in_prolonged_closure = True
                self.closure_start = now
        else:
            if self.in_prolonged_closure:
                self.in_prolonged_closure = False
                cdur = now - (self.closure_start or now)
                if cdur >= 1.5:
                    self.closure_event_count += 1
                    db.insert_attention_event(
                        session_uuid=self.active_session_uuid,
                        event_type="PROLONGED_EYE_CLOSURE",
                        start_time=self.closure_start or now,
                        end_time=now,
                        duration_seconds=round(cdur, 1),
                        severity="HIGH" if cdur >= 3.0 else "MEDIUM"
                    )
                self.closure_start = None

        # Persist Sample to DB
        metric_dict = {
            "session_uuid": self.active_session_uuid,
            "timestamp": now,
            "relative_time_sec": round(rel_time, 1),
            "attention_score": score,
            "attention_state": state,
            "detected_expression": expr,
            "expression_confidence": result.expression_confidence,
            "gaze_direction": result.gaze_direction,
            "gaze_h": result.gaze_h,
            "gaze_v": result.gaze_v,
            "head_yaw": result.head_yaw,
            "head_pitch": result.head_pitch,
            "eyes_open": result.eyes_open,
            "face_present": (result.faces_detected > 0),
            "raw_json": {
                "probs": result.expression_probabilities,
                "fps": result.fps,
                "blinks": result.blink_count
            }
        }
        db.insert_metric_sample(metric_dict)

        # Record into ML dataset
        engagement_label = "High" if score >= 75 else ("Medium" if score >= 45 else "Low")
        ml_entry = {
            "session_uuid": self.active_session_uuid,
            "timestamp": now,
            "gaze_h": result.gaze_h,
            "gaze_v": result.gaze_v,
            "head_yaw": result.head_yaw,
            "head_pitch": result.head_pitch,
            "eyes_open": result.eyes_open,
            "blink_rate": result.blinks_per_min,
            "emotion_confidence": result.expression_confidence,
            "attention_score": score,
            "target_engagement_label": engagement_label
        }
        db.insert_ml_record(ml_entry)

    def start_session(self, title: str = "Focus Session") -> Dict[str, Any]:
        with self.lock:
            # Stop existing if any
            if self.active_session_uuid:
                self._finalize_active_session()

            new_uuid = str(uuid.uuid4())
            now = time.time()
            self.active_session_uuid = new_uuid
            self.session_title = title or "Focus Session"
            self.session_start_time = now
            self.is_paused = False
            self.pause_start_time = None
            self.total_paused_duration = 0.0
            
            self.sampled_scores.clear()
            self.focused_count = 0
            self.partial_count = 0
            self.distracted_count = 0
            self.noface_count = 0
            self.expression_counts.clear()
            self.in_distraction_event = False
            self.distraction_event_count = 0
            self.in_prolonged_closure = False
            self.closure_event_count = 0

            db.create_session(new_uuid, title=self.session_title, start_time=now)
            return {
                "session_uuid": new_uuid,
                "title": self.session_title,
                "start_time": now,
                "status": "ACTIVE"
            }

    def pause_session(self) -> Dict[str, Any]:
        with self.lock:
            if not self.active_session_uuid or self.is_paused:
                return {"status": "error", "message": "No active running session to pause"}
            
            self.is_paused = True
            self.pause_start_time = time.time()
            db.update_session(self.active_session_uuid, status="PAUSED")
            return {"status": "success", "message": "Session paused"}

    def resume_session(self) -> Dict[str, Any]:
        with self.lock:
            if not self.active_session_uuid or not self.is_paused:
                return {"status": "error", "message": "Session is not paused"}
            
            now = time.time()
            if self.pause_start_time:
                self.total_paused_duration += (now - self.pause_start_time)
            self.is_paused = False
            self.pause_start_time = None
            db.update_session(self.active_session_uuid, status="ACTIVE")
            return {"status": "success", "message": "Session resumed"}

    def stop_session(self) -> Dict[str, Any]:
        with self.lock:
            if not self.active_session_uuid:
                return {"status": "error", "message": "No active session"}
            
            session_info = self._finalize_active_session()
            return {"status": "success", "session": session_info}

    def _finalize_active_session(self) -> Dict[str, Any]:
        now = time.time()
        duration = max(1.0, now - self.session_start_time - self.total_paused_duration)
        avg_attn = sum(self.sampled_scores) / len(self.sampled_scores) if self.sampled_scores else 100.0
        
        total_samples = max(1, len(self.sampled_scores))
        f_pct = round(self.focused_count / total_samples * 100.0, 1)
        p_pct = round(self.partial_count / total_samples * 100.0, 1)
        d_pct = round((self.distracted_count + self.noface_count) / total_samples * 100.0, 1)
        
        dominant_expr = max(self.expression_counts, key=self.expression_counts.get) if self.expression_counts else "Neutral"
        
        # Read latest blink count from adapter
        latest_res = self.adapter.get_latest_result()
        blinks = latest_res.blink_count if latest_res else 0

        summary = {
            "end_time": now,
            "duration_seconds": round(duration, 1),
            "status": "COMPLETED",
            "avg_attention": round(avg_attn, 1),
            "focused_pct": f_pct,
            "partial_pct": p_pct,
            "distracted_pct": d_pct,
            "dominant_expression": dominant_expr,
            "total_blinks": blinks,
            "distraction_count": self.distraction_event_count
        }

        db.update_session(self.active_session_uuid, **summary)
        summary["session_uuid"] = self.active_session_uuid
        summary["title"] = self.session_title
        summary["start_time"] = self.session_start_time

        self.active_session_uuid = None
        self.is_paused = False
        return summary

    def get_current_status(self) -> Dict[str, Any]:
        with self.lock:
            if not self.active_session_uuid:
                return {
                    "has_active_session": False,
                    "session_uuid": None,
                    "status": "IDLE",
                    "duration_seconds": 0
                }
            
            now = time.time()
            if self.is_paused:
                dur = (self.pause_start_time or now) - self.session_start_time - self.total_paused_duration
            else:
                dur = now - self.session_start_time - self.total_paused_duration

            total_s = max(1, len(self.sampled_scores))
            return {
                "has_active_session": True,
                "session_uuid": self.active_session_uuid,
                "title": self.session_title,
                "status": "PAUSED" if self.is_paused else "ACTIVE",
                "duration_seconds": int(max(0, dur)),
                "samples_count": len(self.sampled_scores),
                "current_avg_attention": round(sum(self.sampled_scores) / total_s, 1) if self.sampled_scores else 100.0,
                "distraction_events": self.distraction_event_count,
                "prolonged_closures": self.closure_event_count
            }
