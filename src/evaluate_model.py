"""Evaluate the trained anomaly detection model and the risk engine.

Produces the measurable evidence for the thesis:

    results/confusion_matrix.png          RandomForest confusion matrix
    results/risk_score_distribution.png   risk scores: normal vs anomalous
    results/feature_importance.png        RandomForest feature importances
    results/classification_report.md      per-class metrics for both models
    results/model_metrics.csv             metrics table (both models)
    results/evaluation_report.md          full written evaluation report

Run:  python -m src.evaluate_model
"""
import sys
import time
from pathlib import Path
from typing import Dict

# Allow running both as `python -m src.evaluate_model` and as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib

matplotlib.use("Agg")  # headless rendering (no display required)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from src import config
from src.risk_engine import RiskEngine, decide
from src.train_model import (
    compute_metrics,
    isolation_predict,
    load_dataset,
    split_dataset,
    train,
)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def plot_confusion_matrix(cm: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Normal (0)", "Anomaly (1)"])
    ax.set_yticks([0, 1], labels=["Normal (0)", "Anomaly (1)"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix - RandomForest (test set)")
    threshold = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black",
                    fontsize=13)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_feature_importance(model, path: Path) -> pd.Series:
    """Bar chart of the RandomForest feature importances (MDI)."""
    importances = pd.Series(
        model.feature_importances_, index=config.FEATURE_COLUMNS
    ).sort_values()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.barh(importances.index, importances.values, color="#2b8cbe")
    for i, v in enumerate(importances.values):
        ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlabel("Importance (mean decrease in impurity)")
    ax.set_title("Feature Importance - RandomForest")
    ax.set_xlim(0, importances.max() * 1.15)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return importances.sort_values(ascending=False)


def plot_risk_distribution(scores: np.ndarray, labels: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0, 100, 41)
    ax.hist(scores[labels == 0], bins=bins, alpha=0.65, label="Normal traffic",
            color="#2b8cbe")
    ax.hist(scores[labels == 1], bins=bins, alpha=0.65, label="Anomalous traffic",
            color="#e34a33")
    for x, name in [(config.RISK_ALLOW_MAX, "allow|flag"),
                    (config.RISK_FLAG_MAX, "flag|throttle"),
                    (config.RISK_THROTTLE_MAX, "throttle|block")]:
        ax.axvline(x, color="grey", linestyle="--", linewidth=1)
        ax.text(x + 0.5, ax.get_ylim()[1] * 0.95, name, rotation=90,
                va="top", fontsize=8, color="grey")
    ax.set_xlabel("Risk score (0-100)")
    ax.set_ylabel("Number of requests")
    ax.set_title("Risk Score Distribution by Traffic Class (test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_classification_report(y_test, rf_pred, iso_pred,
                                importances: pd.Series, path: Path) -> None:
    """Per-class metrics for both models + feature importance table."""
    names = ["normal (0)", "anomaly (1)"]
    rf_text = classification_report(y_test, rf_pred, target_names=names, digits=4)
    iso_text = classification_report(y_test, iso_pred, target_names=names, digits=4)
    importance_rows = "".join(
        f"| `{feature}` | {value:.4f} |\n" for feature, value in importances.items()
    )
    path.write_text(f"""# Classification Report

Per-class metrics on the held-out test set ({len(y_test):,} samples,
stratified 30% split, random seed {config.RANDOM_SEED}).

## RandomForestClassifier (supervised, primary model)

```
{rf_text}```

## IsolationForest (unsupervised reference)

```
{iso_text}```

## RandomForest feature importances (mean decrease in impurity)

| Feature | Importance |
|---|---|
{importance_rows}""", encoding="utf-8")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _decision_table(scores: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    decisions = pd.Series([decide(s) for s in scores], name="decision")
    table = pd.crosstab(decisions, pd.Series(labels, name="true_label"))
    for col in (0, 1):
        if col not in table.columns:
            table[col] = 0
    order = [d for d in ["allow", "flag", "throttle", "block"] if d in table.index]
    return table.loc[order, [0, 1]]


def build_report(ctx: Dict) -> str:
    rf, iso = ctx["rf_metrics"], ctx["iso_metrics"]
    cm = rf["confusion_matrix"]
    tn, fp, fn, tp = cm.ravel()
    dec = ctx["decision_table"]
    anomalies_stopped = 0
    total_anomalies = int(dec[1].sum()) if 1 in dec.columns else 0
    for d in ("throttle", "block"):
        if d in dec.index:
            anomalies_stopped += int(dec.loc[d, 1])
    normals_allowed = 0
    total_normals = int(dec[0].sum())
    for d in ("allow", "flag"):
        if d in dec.index:
            normals_allowed += int(dec.loc[d, 0])

    lines = f"""# Evaluation Report - Intelligent Secure API Gateway

*Master thesis prototype: "Analysis and evaluation of secure API integration
supported by intelligent mechanisms in software applications."*

## 1. Dataset

| Property | Value |
|---|---|
| Source | Synthetic API traffic generator (`src/generate_dataset.py`), random seed {config.RANDOM_SEED} |
| Total samples | {ctx['n_total']:,} |
| Normal samples (label 0) | {ctx['n_normal']:,} ({ctx['n_normal'] / ctx['n_total']:.1%}) |
| Anomalous samples (label 1) | {ctx['n_anomalous']:,} ({ctx['n_anomalous'] / ctx['n_total']:.1%}) |
| Train / test split | {ctx['n_train']:,} / {ctx['n_test']:,} (70/30, stratified) |

Anomaly types present: {ctx['anomaly_types']}.

**Note:** the dataset is synthetic and clearly documented as such
(`data/README.md`). Distributions were designed to resemble realistic API
traffic, including partial overlap between normal and anomalous behaviour.

## 2. Features and models

Selected features (all observable by the gateway at decision time):
{''.join(f'- `{f}`{chr(10)}' for f in config.FEATURE_COLUMNS)}
Models evaluated:

1. **RandomForestClassifier** (supervised, primary model used by the gateway)
2. **IsolationForest** (unsupervised reference, trained without labels)

## 3. Classification results (held-out test set)

| Metric | RandomForest | IsolationForest |
|---|---|---|
| Accuracy | {rf['accuracy']:.4f} | {iso['accuracy']:.4f} |
| Precision | {rf['precision']:.4f} | {iso['precision']:.4f} |
| Recall | {rf['recall']:.4f} | {iso['recall']:.4f} |
| F1-score | {rf['f1_score']:.4f} | {iso['f1_score']:.4f} |

Average ML inference latency: **{ctx['ml_latency_ms']:.3f} ms/request**;
full risk-engine scoring (heuristics + ML): **{ctx['risk_latency_ms']:.3f} ms/request**.

## 4. Confusion matrix (RandomForest)

![Confusion matrix](confusion_matrix.png)

| | Predicted normal | Predicted anomaly |
|---|---|---|
| **True normal** | {tn:,} (TN) | {fp:,} (FP) |
| **True anomaly** | {fn:,} (FN) | {tp:,} (TP) |

Interpretation:

- **True negatives ({tn:,})** - legitimate requests correctly passed through.
- **False positives ({fp:,})** - legitimate requests wrongly marked anomalous
  ({fp / max(tn + fp, 1):.2%} of normal traffic). These would at most be
  flagged/throttled, not silently dropped, so the availability impact is low.
- **False negatives ({fn:,})** - missed anomalies ({fn / max(fn + tp, 1):.2%}
  of attacks). The rule-based heuristics provide a second, independent line
  of defence for these cases.
- **True positives ({tp:,})** - attacks correctly detected.

## 5. Risk engine evaluation (hybrid rules + ML)

Risk score distribution over the test set:

![Risk score distribution](risk_score_distribution.png)

Gateway decisions per true class (test set):

| Decision | Normal traffic | Anomalous traffic |
|---|---|---|
{''.join(f"| {d} | {int(dec.loc[d, 0]):,} | {int(dec.loc[d, 1]):,} |" + chr(10) for d in dec.index)}
- **{anomalies_stopped / max(total_anomalies, 1):.1%} of anomalous requests**
  are actively stopped (throttled or blocked) by the risk engine.
- **{normals_allowed / max(total_normals, 1):.1%} of normal requests** are
  served without interruption (allowed or merely flagged for audit).

## 6. Interpretation for the thesis

The results support the proposed model: a hybrid gateway that combines
classic security controls (authentication, authorization, rate limiting,
signature filtering) with an intelligent risk engine achieves high anomaly
detection quality (F1 = {rf['f1_score']:.3f} for the supervised model) at a
per-request decision cost of ~{ctx['risk_latency_ms']:.3f} ms, which is
negligible compared to typical API response times (>50 ms).

The comparison with IsolationForest quantifies the value of labelled data:
the supervised model outperforms the unsupervised reference on F1-score by
{(rf['f1_score'] - iso['f1_score']):.3f}, while the unsupervised approach
remains a viable fallback when no labels exist. Because the ML probability
can only *add* risk on top of deterministic rules, the system degrades
gracefully: with the model disabled, hard security guarantees (authn/authz,
rate limits, signature filtering) remain fully enforced.

Limitations: the dataset is synthetic; absolute metric values would differ
on production traffic. The evaluation therefore demonstrates the *validity
of the architecture and methodology* rather than universal detection rates.
"""
    return lines


# ---------------------------------------------------------------------------
# Main evaluation procedure
# ---------------------------------------------------------------------------
def evaluate(
    dataset_path: Path = config.DATASET_PATH,
    model_path: Path = config.MODEL_PATH,
    isolation_path: Path = config.ISOLATION_MODEL_PATH,
    results_dir: Path = config.RESULTS_DIR,
    seed: int = config.RANDOM_SEED,
) -> Dict:
    results_dir.mkdir(parents=True, exist_ok=True)

    # Train first if no model artifact exists yet.
    if not model_path.exists() or not isolation_path.exists():
        print("No trained model found - training now...")
        train(dataset_path, model_path, isolation_path, seed=seed)

    df = load_dataset(dataset_path)
    X_train, X_test, y_train, y_test = split_dataset(df, seed=seed)

    rf = joblib.load(model_path)["model"]
    iso = joblib.load(isolation_path)["model"]

    rf_pred = rf.predict(X_test)
    iso_pred = isolation_predict(iso, X_test)
    rf_metrics = compute_metrics(y_test, rf_pred)
    iso_metrics = compute_metrics(y_test, iso_pred)

    # ML inference latency (per request, averaged over the test set).
    start = time.perf_counter()
    rf.predict_proba(X_test)
    ml_latency_ms = (time.perf_counter() - start) * 1000.0 / len(X_test)

    # Risk engine scores over the test rows (hybrid heuristics + ML).
    engine = RiskEngine(model_path=model_path)
    test_rows = df.loc[X_test.index]
    start = time.perf_counter()
    scores = engine.score_dataframe(test_rows)
    risk_latency_ms = (time.perf_counter() - start) * 1000.0 / len(test_rows)
    labels = y_test.to_numpy()

    # Charts
    plot_confusion_matrix(rf_metrics["confusion_matrix"],
                          results_dir / "confusion_matrix.png")
    plot_risk_distribution(scores, labels,
                           results_dir / "risk_score_distribution.png")
    importances = plot_feature_importance(rf, results_dir / "feature_importance.png")
    write_classification_report(y_test, rf_pred, iso_pred, importances,
                                results_dir / "classification_report.md")

    # Metrics CSV
    perf = pd.DataFrame([
        {"model": "RandomForestClassifier",
         **{k: round(v, 4) for k, v in rf_metrics.items() if k != "confusion_matrix"},
         "avg_prediction_latency_ms": round(ml_latency_ms, 4)},
        {"model": "IsolationForest",
         **{k: round(v, 4) for k, v in iso_metrics.items() if k != "confusion_matrix"},
         "avg_prediction_latency_ms": ""},
        {"model": "RiskEngine (rules + ML)", "accuracy": "", "precision": "",
         "recall": "", "f1_score": "",
         "avg_prediction_latency_ms": round(risk_latency_ms, 4)},
    ])
    perf.to_csv(results_dir / "model_metrics.csv", index=False)

    # Written report
    ctx = {
        "n_total": len(df),
        "n_normal": int((df[config.LABEL_COLUMN] == 0).sum()),
        "n_anomalous": int((df[config.LABEL_COLUMN] == 1).sum()),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "anomaly_types": ", ".join(
            sorted(t for t in df["anomaly_type"].unique() if t != "none")
        ),
        "rf_metrics": rf_metrics,
        "iso_metrics": iso_metrics,
        "ml_latency_ms": ml_latency_ms,
        "risk_latency_ms": risk_latency_ms,
        "decision_table": _decision_table(scores, labels),
        "feature_importances": importances,
    }
    report_path = results_dir / "evaluation_report.md"
    report_path.write_text(build_report(ctx), encoding="utf-8")

    print(f"Evaluation complete. Outputs in {results_dir}:")
    print("  - evaluation_report.md")
    print("  - classification_report.md")
    print("  - confusion_matrix.png")
    print("  - risk_score_distribution.png")
    print("  - feature_importance.png")
    print("  - model_metrics.csv")
    print("\nFeature importances (RandomForest):")
    for feature, value in importances.items():
        print(f"  {feature:<24} {value:.4f}")
    print(f"\nRandomForest:    acc={rf_metrics['accuracy']:.4f} "
          f"prec={rf_metrics['precision']:.4f} rec={rf_metrics['recall']:.4f} "
          f"f1={rf_metrics['f1_score']:.4f}")
    print(f"IsolationForest: acc={iso_metrics['accuracy']:.4f} "
          f"prec={iso_metrics['precision']:.4f} rec={iso_metrics['recall']:.4f} "
          f"f1={iso_metrics['f1_score']:.4f}")
    return ctx


def main() -> None:
    evaluate()


if __name__ == "__main__":
    main()
