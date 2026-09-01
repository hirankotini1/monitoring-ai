"""
Database Module for Emotion & Attention Monitor
Manages SQLite storage for sessions, time-series metrics, attention events, and ML feature records.
"""

import os
import sqlite3
import json
import time
from typing import Dict, List, Optional, Any

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "monitor.db")

os.makedirs(DATA_DIR, exist_ok=True)


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def init_db():
    """Initializes all database tables if they do not already exist."""
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # 1. Sessions Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_uuid TEXT UNIQUE NOT NULL,
                title TEXT DEFAULT 'Focus Session',
                start_time REAL NOT NULL,
                end_time REAL,
                duration_seconds REAL DEFAULT 0.0,
                status TEXT DEFAULT 'ACTIVE',
                avg_attention REAL DEFAULT 100.0,
                focused_pct REAL DEFAULT 0.0,
                partial_pct REAL DEFAULT 0.0,
                distracted_pct REAL DEFAULT 0.0,
                dominant_expression TEXT DEFAULT 'Neutral',
                total_blinks INTEGER DEFAULT 0,
                distraction_count INTEGER DEFAULT 0,
                notes TEXT DEFAULT ''
            );
        """)

        # 2. Session Metrics Samples (1s interval snapshots)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_uuid TEXT NOT NULL,
                timestamp REAL NOT NULL,
                relative_time_sec REAL NOT NULL,
                attention_score REAL NOT NULL,
                attention_state TEXT NOT NULL,
                detected_expression TEXT NOT NULL,
                expression_confidence REAL NOT NULL,
                gaze_direction TEXT NOT NULL,
                gaze_h REAL,
                gaze_v REAL,
                head_yaw REAL,
                head_pitch REAL,
                eyes_open INTEGER NOT NULL,
                face_present INTEGER NOT NULL,
                raw_json TEXT,
                FOREIGN KEY(session_uuid) REFERENCES sessions(session_uuid) ON DELETE CASCADE
            );
        """)

        # 3. Attention Events Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attention_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_uuid TEXT NOT NULL,
                event_type TEXT NOT NULL,
                start_time REAL NOT NULL,
                end_time REAL,
                duration_seconds REAL DEFAULT 0.0,
                severity TEXT DEFAULT 'MEDIUM',
                metadata TEXT DEFAULT '{}',
                FOREIGN KEY(session_uuid) REFERENCES sessions(session_uuid) ON DELETE CASCADE
            );
        """)

        # 4. ML Dataset Records
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ml_dataset (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_uuid TEXT NOT NULL,
                timestamp REAL NOT NULL,
                gaze_h REAL,
                gaze_v REAL,
                head_yaw REAL,
                head_pitch REAL,
                eyes_open INTEGER,
                blink_rate REAL,
                emotion_confidence REAL,
                attention_score REAL,
                target_engagement_label TEXT,
                FOREIGN KEY(session_uuid) REFERENCES sessions(session_uuid) ON DELETE CASCADE
            );
        """)

        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_session ON session_metrics(session_uuid);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_session ON attention_events(session_uuid);")
        conn.commit()


# ============================================================
# Session Operations
# ============================================================

def create_session(session_uuid: str, title: str = "Focus Session", start_time: Optional[float] = None) -> int:
    if start_time is None:
        start_time = time.time()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO sessions (session_uuid, title, start_time, status)
            VALUES (?, ?, ?, 'ACTIVE')
            """,
            (session_uuid, title, start_time)
        )
        conn.commit()
        return cursor.lastrowid


def update_session(session_uuid: str, **kwargs) -> bool:
    if not kwargs:
        return False
    columns = ", ".join([f"{k} = ?" for k in kwargs.keys()])
    values = list(kwargs.values()) + [session_uuid]
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(f"UPDATE sessions SET {columns} WHERE session_uuid = ?", values)
        conn.commit()
        return cursor.rowcount > 0


def get_session_by_uuid(session_uuid: str) -> Optional[Dict[str, Any]]:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sessions WHERE session_uuid = ?", (session_uuid,))
        row = cursor.fetchone()
        return dict(row) if row else None


def list_sessions(limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT * FROM sessions 
            ORDER BY start_time DESC 
            LIMIT ? OFFSET ?
            """,
            (limit, offset)
        )
        return [dict(row) for row in cursor.fetchall()]


def delete_session(session_uuid: str) -> bool:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM session_metrics WHERE session_uuid = ?", (session_uuid,))
        cursor.execute("DELETE FROM attention_events WHERE session_uuid = ?", (session_uuid,))
        cursor.execute("DELETE FROM ml_dataset WHERE session_uuid = ?", (session_uuid,))
        cursor.execute("DELETE FROM sessions WHERE session_uuid = ?", (session_uuid,))
        conn.commit()
        return cursor.rowcount > 0


# ============================================================
# Metrics & Events Operations
# ============================================================

def insert_metric_sample(metric: Dict[str, Any]):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO session_metrics (
                session_uuid, timestamp, relative_time_sec,
                attention_score, attention_state, detected_expression,
                expression_confidence, gaze_direction, gaze_h, gaze_v,
                head_yaw, head_pitch, eyes_open, face_present, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                metric.get("session_uuid"),
                metric.get("timestamp", time.time()),
                metric.get("relative_time_sec", 0.0),
                metric.get("attention_score", 100.0),
                metric.get("attention_state", "FOCUSED"),
                metric.get("detected_expression", "Neutral"),
                metric.get("expression_confidence", 100.0),
                metric.get("gaze_direction", "Looking Center"),
                metric.get("gaze_h", 0.5),
                metric.get("gaze_v", 0.5),
                metric.get("head_yaw", 0.0),
                metric.get("head_pitch", 0.0),
                1 if metric.get("eyes_open", True) else 0,
                1 if metric.get("face_present", True) else 0,
                json.dumps(metric.get("raw_json", {}))
            )
        )
        conn.commit()


def get_session_metrics(session_uuid: str) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT * FROM session_metrics 
            WHERE session_uuid = ? 
            ORDER BY timestamp ASC
            """,
            (session_uuid,)
        )
        return [dict(row) for row in cursor.fetchall()]


def insert_attention_event(
    session_uuid: str,
    event_type: str,
    start_time: float,
    end_time: Optional[float] = None,
    duration_seconds: float = 0.0,
    severity: str = "MEDIUM",
    metadata: Optional[Dict[str, Any]] = None
) -> int:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO attention_events (
                session_uuid, event_type, start_time, end_time, duration_seconds, severity, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_uuid,
                event_type,
                start_time,
                end_time,
                duration_seconds,
                severity,
                json.dumps(metadata or {})
            )
        )
        conn.commit()
        return cursor.lastrowid


def get_session_events(session_uuid: str) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT * FROM attention_events 
            WHERE session_uuid = ? 
            ORDER BY start_time ASC
            """,
            (session_uuid,)
        )
        return [dict(row) for row in cursor.fetchall()]


def insert_ml_record(record: Dict[str, Any]):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO ml_dataset (
                session_uuid, timestamp, gaze_h, gaze_v,
                head_yaw, head_pitch, eyes_open, blink_rate,
                emotion_confidence, attention_score, target_engagement_label
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.get("session_uuid"),
                record.get("timestamp", time.time()),
                record.get("gaze_h", 0.5),
                record.get("gaze_v", 0.5),
                record.get("head_yaw", 0.0),
                record.get("head_pitch", 0.0),
                1 if record.get("eyes_open", True) else 0,
                record.get("blink_rate", 0.0),
                record.get("emotion_confidence", 100.0),
                record.get("attention_score", 100.0),
                record.get("target_engagement_label", "High")
            )
        )
        conn.commit()


def get_all_ml_records() -> List[Dict[str, Any]]:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM ml_dataset ORDER BY timestamp ASC")
        return [dict(row) for row in cursor.fetchall()]


def get_global_aggregate_stats() -> Dict[str, Any]:
    """Computes high-performance aggregate breakdown across all metrics in database."""
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # State counts
        cursor.execute("SELECT attention_state, COUNT(*) as cnt FROM session_metrics GROUP BY attention_state")
        state_rows = cursor.fetchall()
        states = {row["attention_state"]: row["cnt"] for row in state_rows}
        
        # Expression counts
        cursor.execute("SELECT detected_expression, COUNT(*) as cnt FROM session_metrics GROUP BY detected_expression")
        expr_rows = cursor.fetchall()
        exprs = {row["detected_expression"]: row["cnt"] for row in expr_rows}
        
        # Gaze counts
        cursor.execute("SELECT gaze_direction, COUNT(*) as cnt FROM session_metrics GROUP BY gaze_direction")
        gaze_rows = cursor.fetchall()
        gazes = {row["gaze_direction"]: row["cnt"] for row in gaze_rows}
        
        # Total events
        cursor.execute("SELECT event_type, COUNT(*) as cnt, SUM(duration_seconds) as total_dur FROM attention_events GROUP BY event_type")
        event_rows = cursor.fetchall()
        events = {row["event_type"]: {"count": row["cnt"], "total_duration": row["total_dur"] or 0.0} for row in event_rows}
        
        return {
            "states": states,
            "expressions": exprs,
            "gazes": gazes,
            "events": events
        }


# Initialize on import
init_db()
