"""Train the anomaly detection models on the synthetic traffic dataset.

Primary model:   RandomForestClassifier (supervised - labels available)
Reference model: IsolationForest (unsupervised - trained without labels,
                 included to show what is achievable when no labelled data
                 exists in practice)

The trained RandomForest is stored as models/anomaly_model.joblib and is
loaded by the gateway's risk engine at runtime.

Run:  python -m src.train_model
"""
import sys
from pathlib import Path
from typing import Dict, Tuple

# Allow running both as `python -m src.train_model` and as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from src import config


def load_dataset(path: Path = config.DATASET_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run: python -m src.generate_dataset"
        )
    return pd.read_csv(path)


def split_dataset(
    df: pd.DataFrame, test_size: float = 0.30, seed: int = config.RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split on the ML feature columns."""
    X = df[config.FEATURE_COLUMNS]
    y = df[config.LABEL_COLUMN]
    return train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )


def compute_metrics(y_true, y_pred) -> Dict:
    """Standard binary classification metrics for the thesis evaluation."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def train_random_forest(X_train, y_train, seed: int = config.RANDOM_SEED):
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=None,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_isolation_forest(X_train, contamination: float,
                           seed: int = config.RANDOM_SEED):
    model = IsolationForest(
        n_estimators=150,
        contamination=contamination,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train)
    return model


def isolation_predict(model: IsolationForest, X) -> np.ndarray:
    """Map IsolationForest output (-1 anomaly / 1 normal) to labels 1/0."""
    return (model.predict(X) == -1).astype(int)


def train(
    dataset_path: Path = config.DATASET_PATH,
    model_path: Path = config.MODEL_PATH,
    isolation_path: Path = config.ISOLATION_MODEL_PATH,
    seed: int = config.RANDOM_SEED,
) -> Dict:
    """Full training procedure; returns metrics for both models."""
    df = load_dataset(dataset_path)
    X_train, X_test, y_train, y_test = split_dataset(df, seed=seed)

    # --- Supervised model (primary) ---------------------------------
    rf = train_random_forest(X_train, y_train, seed=seed)
    rf_metrics = compute_metrics(y_test, rf.predict(X_test))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": rf, "features": config.FEATURE_COLUMNS}, model_path)

    # --- Unsupervised reference model --------------------------------
    contamination = float(y_train.mean())
    iso = train_isolation_forest(X_train, contamination, seed=seed)
    iso_metrics = compute_metrics(y_test, isolation_predict(iso, X_test))
    joblib.dump({"model": iso, "features": config.FEATURE_COLUMNS}, isolation_path)

    return {
        "random_forest": rf_metrics,
        "isolation_forest": iso_metrics,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "model_path": model_path,
    }


def _print_metrics(name: str, m: Dict) -> None:
    print(f"\n--- {name} ---")
    print(f"Accuracy : {m['accuracy']:.4f}")
    print(f"Precision: {m['precision']:.4f}")
    print(f"Recall   : {m['recall']:.4f}")
    print(f"F1-score : {m['f1_score']:.4f}")
    print(f"Confusion matrix (rows=true, cols=predicted):\n{m['confusion_matrix']}")


def main() -> None:
    results = train()
    print(f"Training samples: {results['n_train']}")
    print(f"Test samples:     {results['n_test']}")
    _print_metrics("RandomForestClassifier (supervised)", results["random_forest"])
    _print_metrics("IsolationForest (unsupervised reference)", results["isolation_forest"])
    print(f"\nModel saved to {results['model_path']}")


if __name__ == "__main__":
    main()
