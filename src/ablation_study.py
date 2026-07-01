"""Ablation study: rule-based vs ML-based vs hybrid anomaly detection.

Quantifies the contribution of each half of the intelligent risk engine by
evaluating three detectors on the same held-out test set:

    1. rules_only   heuristic risk score only (no ML term), thresholded
    2. ml_only      RandomForest classifier prediction only
    3. hybrid       full risk engine (heuristics + ML), thresholded

For the score-based detectors (rules_only, hybrid) a request counts as a
"detected anomaly" when its risk score exceeds a decision threshold. Because
that choice strongly affects the precision/recall balance, both operational
thresholds are reported:

    flag boundary  (> RISK_ALLOW_MAX = 30) - request treated as suspicious
                   (flag / throttle / block); recall-oriented.
    action boundary(> RISK_FLAG_MAX  = 60) - request actively restricted
                   (throttle / block); precision-oriented.

ML-based only always uses the classifier's binary prediction (threshold
independent). The chart uses the action boundary, the most meaningful point
for comparing systems that take restrictive action.

Outputs:
    results/ablation_study.csv        metrics per approach x threshold
    results/ablation_study_chart.png  grouped bar chart (action boundary)

Run:  python -m src.ablation_study
"""
import sys
from pathlib import Path
from typing import Dict

# Allow running both as a module and as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from src import config
from src.risk_engine import RiskEngine
from src.train_model import load_dataset, split_dataset


# Score-based decision boundaries (see module docstring).
FLAG_THRESHOLD = config.RISK_ALLOW_MAX    # > 30 : suspicious (flag+)
ACTION_THRESHOLD = config.RISK_FLAG_MAX    # > 60 : restricted (throttle+)


def _metrics(name: str, threshold_label: str, y_true, y_pred) -> Dict:
    return {
        "approach": name,
        "decision_boundary": threshold_label,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4),
    }


def run_ablation(
    dataset_path: Path = config.DATASET_PATH,
    model_path: Path = config.MODEL_PATH,
    results_dir: Path = config.RESULTS_DIR,
    seed: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    df = load_dataset(dataset_path)
    _, X_test, _, y_test = split_dataset(df, seed=seed)
    test_rows = df.loc[X_test.index]
    y_true = y_test.to_numpy()

    # 1) Rules only - risk engine with the ML term disabled.
    rules_scores = RiskEngine(model_path=model_path, use_model=False).score_dataframe(test_rows)
    # 3) Hybrid - full risk engine (heuristics + ML).
    hybrid_scores = RiskEngine(model_path=model_path, use_model=True).score_dataframe(test_rows)
    # 2) ML only - raw RandomForest classifier prediction (threshold-free).
    ml_pred = joblib.load(model_path)["model"].predict(X_test)

    rows = []
    for label, thr in (("flag (>30)", FLAG_THRESHOLD), ("action (>60)", ACTION_THRESHOLD)):
        rows.append(_metrics("Rule-based only", label, y_true,
                             (rules_scores > thr).astype(int)))
        rows.append(_metrics("ML-based only", label, y_true, ml_pred))
        rows.append(_metrics("Hybrid (rules + ML)", label, y_true,
                             (hybrid_scores > thr).astype(int)))
    table = pd.DataFrame(rows)

    results_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(results_dir / "ablation_study.csv", index=False)
    # Chart uses the action boundary (the "take restrictive action" point).
    _plot(table[table["decision_boundary"] == "action (>60)"],
          results_dir / "ablation_study_chart.png")
    return table


def _plot(table: pd.DataFrame, path: Path) -> None:
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    labels = ["Accuracy", "Precision", "Recall", "F1-score"]
    colors = ["#7fb3d5", "#f0a868", "#82c082"]
    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (_, row) in enumerate(table.reset_index(drop=True).iterrows()):
        values = [row[m] for m in metrics]
        bars = ax.bar(x + (i - 1) * width, values, width,
                      label=row["approach"], color=colors[i % len(colors)])
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Ablation Study - Detection Approaches at the Action "
                 "Boundary (score > 60, test set)")
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    table = run_ablation()
    print("Ablation study (held-out test set):\n")
    print(table.to_string(index=False))
    print()
    print(f"\nSaved: {config.RESULTS_DIR / 'ablation_study.csv'}")
    print(f"Saved: {config.RESULTS_DIR / 'ablation_study_chart.png'}")


if __name__ == "__main__":
    main()
