"""
Machine Learning Engine Module
Consumes structured features produced by the Video Analyzer to build datasets,
train predictive engagement models, and compute feature importance.
Does not perform vision processing or replace the Video Analyzer.
"""

import os
import time
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import db

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(DATASETS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_FILE = os.path.join(MODELS_DIR, "engagement_classifier.joblib")
CSV_DATASET_FILE = os.path.join(DATASETS_DIR, "engagement_training_data.csv")

FEATURE_NAMES = [
    "gaze_h", "gaze_v", "head_yaw", "head_pitch",
    "eyes_open", "blink_rate", "emotion_confidence", "attention_score"
]


class MLEngine:
    """Manages ML training and inference on top of Video Analyzer metrics."""

    def __init__(self):
        self.model: Optional[RandomForestClassifier] = None
        self.model_metadata: Dict[str, Any] = {}
        self.load_model_if_exists()

    def load_model_if_exists(self):
        if os.path.isfile(MODEL_FILE):
            try:
                saved = joblib.load(MODEL_FILE)
                self.model = saved.get("model")
                self.model_metadata = saved.get("metadata", {})
            except Exception as e:
                self.model = None

    def export_dataset_csv(self) -> str:
        """Exports all recorded ML records from database to CSV."""
        records = db.get_all_ml_records()
        if len(records) < 15:
            # Seed with baseline synthetic distribution if not enough session data yet
            self._seed_initial_dataset()
            records = db.get_all_ml_records()

        df = pd.DataFrame(records)
        df.to_csv(CSV_DATASET_FILE, index=False)
        return CSV_DATASET_FILE

    def _seed_initial_dataset(self):
        """Generates initial baseline dataset samples if database is empty."""
        np.random.seed(42)
        samples = []
        for _ in range(120):
            # Focused samples
            samples.append({
                "session_uuid": "baseline-seed",
                "timestamp": time.time(),
                "gaze_h": float(np.random.normal(0.50, 0.04)),
                "gaze_v": float(np.random.normal(0.50, 0.03)),
                "head_yaw": float(np.random.normal(0.0, 0.05)),
                "head_pitch": float(np.random.normal(0.0, 0.04)),
                "eyes_open": 1,
                "blink_rate": float(np.random.uniform(12, 20)),
                "emotion_confidence": float(np.random.uniform(85, 99)),
                "attention_score": float(np.random.uniform(80, 100)),
                "target_engagement_label": "High"
            })
            # Moderate samples
            samples.append({
                "session_uuid": "baseline-seed",
                "timestamp": time.time(),
                "gaze_h": float(np.random.normal(0.58, 0.08)),
                "gaze_v": float(np.random.normal(0.55, 0.06)),
                "head_yaw": float(np.random.normal(0.12, 0.08)),
                "head_pitch": float(np.random.normal(0.10, 0.06)),
                "eyes_open": 1,
                "blink_rate": float(np.random.uniform(22, 35)),
                "emotion_confidence": float(np.random.uniform(70, 90)),
                "attention_score": float(np.random.uniform(50, 74)),
                "target_engagement_label": "Medium"
            })
            # Distracted samples
            samples.append({
                "session_uuid": "baseline-seed",
                "timestamp": time.time(),
                "gaze_h": float(np.random.choice([0.15, 0.85])),
                "gaze_v": float(np.random.choice([0.20, 0.80])),
                "head_yaw": float(np.random.choice([-0.45, 0.45])),
                "head_pitch": float(np.random.choice([-0.35, 0.35])),
                "eyes_open": int(np.random.choice([0, 1], p=[0.4, 0.6])),
                "blink_rate": float(np.random.uniform(5, 45)),
                "emotion_confidence": float(np.random.uniform(60, 85)),
                "attention_score": float(np.random.uniform(0, 44)),
                "target_engagement_label": "Low"
            })

        for s in samples:
            db.insert_ml_record(s)

    def train_model(self) -> Dict[str, Any]:
        """Trains a Random Forest classifier using features collected from Video Analyzer."""
        self.export_dataset_csv()
        df = pd.read_csv(CSV_DATASET_FILE)
        
        if len(df) < 15:
            return {"status": "error", "message": "Need at least 15 recorded samples to train model."}

        X = df[FEATURE_NAMES]
        y = df["target_engagement_label"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

        clf = RandomForestClassifier(n_estimators=60, max_depth=6, random_state=42)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc = round(accuracy_score(y_test, y_pred) * 100.0, 1)

        # Feature importances
        importances = {
            name: round(float(imp) * 100.0, 1)
            for name, imp in zip(FEATURE_NAMES, clf.feature_importances_)
        }

        metadata = {
            "trained_at": time.time(),
            "accuracy": acc,
            "total_samples": len(df),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "feature_importances": importances,
            "classes": list(clf.classes_)
        }

        # Save model
        joblib.dump({"model": clf, "metadata": metadata}, MODEL_FILE)
        self.model = clf
        self.model_metadata = metadata

        return {
            "status": "success",
            "message": f"Engagement Model trained successfully with {acc}% accuracy.",
            "metrics": metadata
        }

    def predict(self, feature_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Predicts engagement category given a feature vector from the analyzer."""
        if self.model is None:
            # Fallback heuristic if model not yet trained
            attn = float(feature_dict.get("attention_score", 100.0))
            label = "High" if attn >= 75 else ("Medium" if attn >= 45 else "Low")
            return {"predicted_label": label, "confidence": 95.0, "is_ml_model": False}

        vec = np.array([[
            float(feature_dict.get("gaze_h", 0.5)),
            float(feature_dict.get("gaze_v", 0.5)),
            float(feature_dict.get("head_yaw", 0.0)),
            float(feature_dict.get("head_pitch", 0.0)),
            1.0 if feature_dict.get("eyes_open", True) else 0.0,
            float(feature_dict.get("blink_rate", 0.0)),
            float(feature_dict.get("emotion_confidence", 100.0)),
            float(feature_dict.get("attention_score", 100.0))
        ]])

        probs = self.model.predict_proba(vec)[0]
        classes = self.model.classes_
        top_idx = int(np.argmax(probs))
        
        return {
            "predicted_label": classes[top_idx],
            "confidence": round(float(probs[top_idx]) * 100.0, 1),
            "probabilities": {cls_name: round(float(p) * 100.0, 1) for cls_name, p in zip(classes, probs)},
            "is_ml_model": True
        }

    def get_info(self) -> Dict[str, Any]:
        return {
            "has_trained_model": (self.model is not None),
            "metadata": self.model_metadata,
            "features": FEATURE_NAMES
        }


ml_engine = MLEngine()
