"""Tests for dataset generation, model training and report generation.

Small dataset sizes are used so the whole suite stays fast; the real
artifacts are produced with the defaults via the module entry points.
"""
import pandas as pd
import pytest

from src import config
from src.evaluate_model import evaluate
from src.generate_dataset import generate_dataset
from src.train_model import train


@pytest.fixture(scope="module")
def small_dataset_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "traffic.csv"
    generate_dataset(n_records=800, seed=7).to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------
def test_dataset_has_required_columns_and_labels(small_dataset_path):
    df = pd.read_csv(small_dataset_path)
    required = {
        "timestamp", "client_id", "ip_address", "endpoint", "http_method",
        "status_code", "response_time_ms", "request_size_bytes",
        "requests_per_minute", "failed_auth_count", "token_valid",
        "user_role", "endpoint_sensitivity", "payload_suspicious",
        "anomaly_label",
    }
    assert required.issubset(df.columns)
    assert set(df["anomaly_label"].unique()) == {0, 1}
    # Both classes present, anomalies in the configured minority ratio.
    assert 0.10 < df["anomaly_label"].mean() < 0.20


def test_dataset_generation_is_deterministic():
    a = generate_dataset(n_records=300, seed=11)
    b = generate_dataset(n_records=300, seed=11)
    pd.testing.assert_frame_equal(a, b)


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------
def test_model_training_runs_successfully(small_dataset_path, tmp_path):
    model_path = tmp_path / "model.joblib"
    iso_path = tmp_path / "iso.joblib"
    results = train(small_dataset_path, model_path, iso_path)

    assert model_path.exists()
    assert iso_path.exists()
    metrics = results["random_forest"]
    assert metrics["accuracy"] > 0.9
    assert metrics["f1_score"] > 0.8
    assert metrics["confusion_matrix"].shape == (2, 2)


# ---------------------------------------------------------------------------
# Evaluation pipeline
# ---------------------------------------------------------------------------
def test_evaluation_report_is_generated(small_dataset_path, tmp_path):
    results_dir = tmp_path / "results"
    evaluate(
        dataset_path=small_dataset_path,
        model_path=tmp_path / "model.joblib",      # missing -> trains first
        isolation_path=tmp_path / "iso.joblib",
        results_dir=results_dir,
    )
    report = results_dir / "evaluation_report.md"
    assert report.exists()
    text = report.read_text(encoding="utf-8")
    for keyword in ("Accuracy", "Precision", "Recall", "F1-score",
                    "Confusion matrix", "Interpretation"):
        assert keyword in text

    assert (results_dir / "confusion_matrix.png").exists()
    assert (results_dir / "risk_score_distribution.png").exists()
    assert (results_dir / "feature_importance.png").exists()
    assert (results_dir / "classification_report.md").exists()
    assert (results_dir / "model_metrics.csv").exists()

    perf = pd.read_csv(results_dir / "model_metrics.csv")
    assert "accuracy" in perf.columns
    assert len(perf) >= 2  # RandomForest + IsolationForest rows
