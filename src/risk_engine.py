"""Intelligent risk engine: hybrid rule-based + ML risk scoring.

Every request is described by the feature vector defined in
``config.FEATURE_COLUMNS`` and receives a risk score in [0, 100]:

    risk = min(100, heuristic_points + ML_RISK_WEIGHT * P(anomaly))

The heuristic part encodes domain knowledge (failed logins, invalid tokens,
suspicious payloads, endpoint sensitivity, traffic volume). The ML part is
the anomaly probability of the trained RandomForest classifier and can only
*raise* the score - the machine-learned signal amplifies the rules instead
of diluting them, and the gateway degrades gracefully to pure rule-based
scoring when no trained model is available.

Decision bands (see config): allow / flag / throttle / block.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd

from src import config


@dataclass
class RiskAssessment:
    score: float                 # 0..100
    decision: str                # allow | flag | throttle | block
    ml_probability: Optional[float]  # None when no model is loaded
    heuristic_points: float


def decide(score: float) -> str:
    """Map a 0-100 risk score to a gateway decision."""
    if score <= config.RISK_ALLOW_MAX:
        return "allow"
    if score <= config.RISK_FLAG_MAX:
        return "flag"
    if score <= config.RISK_THROTTLE_MAX:
        return "throttle"
    return "block"


class RiskEngine:
    def __init__(self, model_path: Path = config.MODEL_PATH, use_model: bool = True):
        self._model_path = model_path
        self._use_model = use_model
        self._model = None
        self._model_loaded = False

    # ------------------------------------------------------------------
    # Model handling
    # ------------------------------------------------------------------
    def _load_model(self):
        """Lazily load the trained classifier; tolerate a missing file."""
        if self._model_loaded:
            return self._model
        self._model_loaded = True
        if self._use_model and self._model_path.exists():
            artifact = joblib.load(self._model_path)
            self._model = artifact["model"]
        return self._model

    @property
    def model_available(self) -> bool:
        return self._load_model() is not None

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    @staticmethod
    def _heuristic_points(features: Dict[str, float]) -> float:
        """Rule-based risk points derived from security domain knowledge."""
        pts = 0.0
        if not features["token_valid"]:
            pts += 30.0
        if features["payload_suspicious"]:
            pts += 45.0
        # Repeated authentication failures (brute force / credential stuffing)
        pts += min(features["failed_auth_count"] * 10.0, 55.0)
        # Traffic volume relative to the hard rate limit
        ratio = min(features["requests_per_minute"] / config.RATE_LIMIT_MAX_REQUESTS, 1.0)
        pts += ratio * 20.0
        # Business criticality of the target endpoint
        pts += features["endpoint_sensitivity"] * 20.0
        # Abnormal request size / response time
        if features["request_size_bytes"] > config.LARGE_REQUEST_BYTES:
            pts += 10.0
        if features["response_time_ms"] > config.SLOW_RESPONSE_MS:
            pts += 10.0
        return min(pts, 100.0)

    def ml_probability(self, features: Dict[str, float]) -> Optional[float]:
        """P(anomaly) from the trained classifier, or None if unavailable."""
        model = self._load_model()
        if model is None:
            return None
        frame = pd.DataFrame([features])[config.FEATURE_COLUMNS]
        return float(model.predict_proba(frame)[0][1])

    def score(self, features: Dict[str, float]) -> RiskAssessment:
        """Score one request context and return the gateway decision."""
        heuristic = self._heuristic_points(features)
        prob = self.ml_probability(features)
        total = heuristic if prob is None else min(
            100.0, heuristic + config.ML_RISK_WEIGHT * prob
        )
        return RiskAssessment(
            score=round(total, 1),
            decision=decide(total),
            ml_probability=prob,
            heuristic_points=heuristic,
        )

    # ------------------------------------------------------------------
    # Batch scoring for offline evaluation
    # ------------------------------------------------------------------
    def score_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        """Vectorised risk scores for a dataset (used by evaluate_model)."""
        pts = np.zeros(len(df), dtype=float)
        pts += np.where(df["token_valid"] == 0, 30.0, 0.0)
        pts += np.where(df["payload_suspicious"] == 1, 45.0, 0.0)
        pts += np.minimum(df["failed_auth_count"] * 10.0, 55.0)
        pts += np.minimum(df["requests_per_minute"] / config.RATE_LIMIT_MAX_REQUESTS, 1.0) * 20.0
        pts += df["endpoint_sensitivity"] * 20.0
        pts += np.where(df["request_size_bytes"] > config.LARGE_REQUEST_BYTES, 10.0, 0.0)
        pts += np.where(df["response_time_ms"] > config.SLOW_RESPONSE_MS, 10.0, 0.0)
        pts = np.minimum(pts, 100.0)

        model = self._load_model()
        if model is not None:
            probs = model.predict_proba(df[config.FEATURE_COLUMNS])[:, 1]
            pts = np.minimum(100.0, pts + config.ML_RISK_WEIGHT * probs)
        return pts
