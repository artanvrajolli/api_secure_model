"""Error analysis of the RandomForest anomaly detector.

Extracts the misclassified test-set requests so they can be inspected
qualitatively for the thesis:

    False positives - normal requests predicted anomalous. In the gateway
                      these cause friction for legitimate users (flag /
                      throttle), so their feature patterns matter for
                      availability.
    False negatives - anomalous requests predicted normal. These are the
                      attacks the ML component misses (the static controls
                      remain the backstop).

Outputs:
    results/false_positives.csv
    results/false_negatives.csv
    results/error_analysis_summary.csv   (per-anomaly-type miss breakdown)

Run:  python -m src.error_analysis
"""
import sys
from pathlib import Path

# Allow running both as a module and as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd

from src import config
from src.train_model import load_dataset, split_dataset

# Columns kept in the exported error tables (features + context + probability).
EXPORT_COLUMNS = [
    "client_id", "ip_address", "endpoint", "http_method", "status_code",
    "response_time_ms", "request_size_bytes", "requests_per_minute",
    "failed_auth_count", "token_valid", "user_role", "endpoint_sensitivity",
    "payload_suspicious", "anomaly_label", "anomaly_type",
    "predicted_label", "anomaly_probability",
]


def run_error_analysis(
    dataset_path: Path = config.DATASET_PATH,
    model_path: Path = config.MODEL_PATH,
    results_dir: Path = config.RESULTS_DIR,
    seed: int = config.RANDOM_SEED,
) -> dict:
    df = load_dataset(dataset_path)
    X_train, X_test, y_train, y_test = split_dataset(df, seed=seed)

    rf = joblib.load(model_path)["model"]
    pred = rf.predict(X_test)
    proba = rf.predict_proba(X_test)[:, 1]

    test = df.loc[X_test.index].copy()
    test["predicted_label"] = pred
    test["anomaly_probability"] = proba.round(4)

    fp = test[(test["anomaly_label"] == 0) & (test["predicted_label"] == 1)]
    fn = test[(test["anomaly_label"] == 1) & (test["predicted_label"] == 0)]

    results_dir.mkdir(parents=True, exist_ok=True)
    fp[EXPORT_COLUMNS].to_csv(results_dir / "false_positives.csv", index=False)
    fn[EXPORT_COLUMNS].to_csv(results_dir / "false_negatives.csv", index=False)

    # Per-anomaly-type breakdown: how many of each type were caught vs missed.
    anomalies = test[test["anomaly_label"] == 1]
    summary = (
        anomalies.groupby("anomaly_type")
        .apply(lambda g: pd.Series({
            "total": len(g),
            "missed_false_negatives": int((g["predicted_label"] == 0).sum()),
            "detected": int((g["predicted_label"] == 1).sum()),
            "recall": round((g["predicted_label"] == 1).mean(), 4),
        }), include_groups=False)
        .reset_index()
    )
    summary.to_csv(results_dir / "error_analysis_summary.csv", index=False)

    return {
        "n_test": len(test),
        "n_fp": len(fp),
        "n_fn": len(fn),
        "fp_rate_of_normal": round(len(fp) / int((test["anomaly_label"] == 0).sum()), 4),
        "fn_rate_of_anomaly": round(len(fn) / int((test["anomaly_label"] == 1).sum()), 4),
        "summary": summary,
    }


def main() -> None:
    r = run_error_analysis()
    print(f"Test samples: {r['n_test']}")
    print(f"False positives: {r['n_fp']} "
          f"({r['fp_rate_of_normal']:.2%} of normal traffic)")
    print(f"False negatives: {r['n_fn']} "
          f"({r['fn_rate_of_anomaly']:.2%} of anomalies)")
    print("\nPer-anomaly-type detection breakdown:")
    print(r["summary"].to_string(index=False))
    print(f"\nSaved: {config.RESULTS_DIR / 'false_positives.csv'}")
    print(f"Saved: {config.RESULTS_DIR / 'false_negatives.csv'}")
    print(f"Saved: {config.RESULTS_DIR / 'error_analysis_summary.csv'}")


if __name__ == "__main__":
    main()
