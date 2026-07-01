"""Central configuration for the Intelligent Secure API Gateway prototype.

All tunable parameters of the model (tokens, endpoint policies, rate limits,
risk thresholds, ML feature set, file paths) live here so the rest of the
code base stays free of magic numbers.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

DATASET_PATH = DATA_DIR / "api_traffic_dataset.csv"
MODEL_PATH = MODELS_DIR / "anomaly_model.joblib"
ISOLATION_MODEL_PATH = MODELS_DIR / "isolation_forest.joblib"

# Global seed so dataset generation, train/test split and model training
# are reproducible for the thesis evaluation.
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Authentication / authorization (simulated OAuth2 bearer tokens / JWT)
# ---------------------------------------------------------------------------
# In a production system these would be verified JWT signatures + claims.
# For the prototype we simulate the validation step with a fixed token store.
API_TOKENS = {
    "valid_user_token": {"client_id": "client_user_1", "role": "user"},
    "valid_admin_token": {"client_id": "client_admin_1", "role": "admin"},
}

# Minimum role required per endpoint ("public" endpoints need no token).
ENDPOINT_ACCESS = {
    "/public": "public",
    "/users": "user",
    "/orders": "user",
    "/payments": "user",
    "/admin": "admin",
}

# How business-critical each endpoint is (0.0 = harmless, 1.0 = critical).
# Used both as an ML feature and as a heuristic risk factor.
ENDPOINT_SENSITIVITY = {
    "/public": 0.1,
    "/users": 0.5,
    "/orders": 0.5,
    "/payments": 0.9,
    "/admin": 1.0,
}
DEFAULT_SENSITIVITY = 0.5

# Paths that bypass the gateway pipeline (interactive API documentation).
BYPASS_PATHS = {"/docs", "/redoc", "/openapi.json", "/favicon.ico"}

# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
RATE_LIMIT_MAX_REQUESTS = 15   # hard limit: requests per client per window
RATE_LIMIT_WINDOW_SECONDS = 60

# ---------------------------------------------------------------------------
# Traffic telemetry (sliding-window statistics kept per client)
# ---------------------------------------------------------------------------
FAILED_AUTH_WINDOW_SECONDS = 300   # window for counting failed authentications
DEFAULT_RESPONSE_TIME_MS = 120.0   # assumed response time for unseen clients

# ---------------------------------------------------------------------------
# Risk engine
# ---------------------------------------------------------------------------
# Decision bands over the 0-100 risk score (inclusive upper bounds):
#   0-30 allow | 31-60 allow + flag | 61-80 throttle | 81-100 block
RISK_ALLOW_MAX = 30
RISK_FLAG_MAX = 60
RISK_THROTTLE_MAX = 80

ML_RISK_WEIGHT = 35.0        # max points the ML anomaly probability can add
LARGE_REQUEST_BYTES = 10_000  # payloads above this size are a risk factor
SLOW_RESPONSE_MS = 1_000      # response times above this are a risk factor

# Features used by the ML model. All of them are observable by the gateway
# at decision time (response_time_ms is the client's recent average).
FEATURE_COLUMNS = [
    "requests_per_minute",
    "failed_auth_count",
    "token_valid",
    "endpoint_sensitivity",
    "payload_suspicious",
    "request_size_bytes",
    "response_time_ms",
]
LABEL_COLUMN = "anomaly_label"
