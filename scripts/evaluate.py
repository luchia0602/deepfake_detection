"""
Train a logistic-regression classifier on pre-extracted features and report
test-set metrics.

Usage:
Single representation: python scripts/evaluate.py --representation prosodic
Multiple representations (trains one classifier per rep): python scripts/evaluate.py --representation prosodic whisper xlsr
Ensemble of several representations (concatenated features): python scripts/evaluate.py --representation ensemble --ensemble-of prosodic mfcc lfcc cqcc xlsr whisper
Force overwrite of saved predictions: python scripts/evaluate.py --representation prosodic --force
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

DEFAULT_FEATURES_DIR = Path("artifacts/features")


def load_split(features_dir: Path, split: str, feat_file: str):
    X = np.load(features_dir / split / feat_file).astype(np.float32)
    y = np.load(features_dir / split / "y.npy")
    np.nan_to_num(X, copy=False)
    return X, y


def load_split_concat(features_dir: Path, split: str, reps):
    """Load several representations for one split and concatenate them column-wise.

    All representations share the same row order (guaranteed by the extraction
    pipeline), so horizontal stacking keeps each row aligned to the same utterance.
    """
    parts = []
    for rep in reps:
        X = np.load(features_dir / split / f"X_{rep}.npy").astype(np.float32)
        np.nan_to_num(X, copy=False)
        parts.append(X)
    X = np.hstack(parts)
    y = np.load(features_dir / split / "y.npy")
    return X, y


def compute_eer(y_true, scores):
    fpr, tpr, _ = roc_curve(y_true, scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2)


def evaluate_one(rep: str, features_dir: Path, force: bool, ensemble_of=None):
    out_dir = features_dir / "predictions" / rep
    out_dir.mkdir(parents=True, exist_ok=True)

    probs_path = out_dir / "probs_test.npy"
    preds_path = out_dir / "preds_test.npy"
    metrics_path = out_dir / "metrics.json"

    if probs_path.exists() and not force:
        print(f"[{rep}] predictions already exist — skipping (use --force to overwrite)")
        metrics = json.loads(metrics_path.read_text())
        print(f"[{rep}] acc={metrics['accuracy']:.4f}  auc={metrics['auc']:.4f}  eer={metrics['eer']:.4f}")
        return

    if ensemble_of:
        print(f"[{rep}] concatenating: {', '.join(ensemble_of)}")
        X_tr, y_tr = load_split_concat(features_dir, "train", ensemble_of)
        X_te, y_te = load_split_concat(features_dir, "test",  ensemble_of)
    else:
        feat_file = f"X_{rep}.npy"
        X_tr, y_tr = load_split(features_dir, "train", feat_file)
        X_te, y_te = load_split(features_dir, "test",  feat_file)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    clf = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0, verbose=1)
    clf.fit(X_tr, y_tr)

    probs = clf.predict_proba(X_te)[:, 1]
    preds = clf.predict(X_te)

    # default-threshold (0.5) metrics
    metrics = {
        "representation": rep,
        "accuracy":  float(accuracy_score(y_te, preds)),
        "auc":       float(roc_auc_score(y_te, probs)),
        "eer":       float(compute_eer(y_te, probs)),
        "precision": float(precision_score(y_te, preds)),
        "recall":    float(recall_score(y_te, preds)),
        "f1":        float(f1_score(y_te, preds)),
    }

    if ensemble_of:
        metrics["ensemble_of"] = list(ensemble_of)

    print(
        f"[{rep}] "
        f"acc={metrics['accuracy']:.4f}  "
        f"auc={metrics['auc']:.4f}  "
        f"eer={metrics['eer']:.4f}  "
        f"f1={metrics['f1']:.4f}"
    )

    np.save(probs_path, probs)
    np.save(preds_path, preds)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"[{rep}] saved predictions → {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train LR classifier and evaluate on test set.")
    parser.add_argument(
        "--representation",
        nargs="+",
        required=True,
        metavar="REP",
        help="One or more feature representations, e.g. prosodic whisper xlsr. "
             "Use the name 'ensemble' together with --ensemble-of to train on "
             "concatenated features.",
    )
    parser.add_argument(
        "--ensemble-of",
        nargs="+",
        default=None,
        metavar="REP",
        help="Representations to concatenate when --representation includes 'ensemble'.",
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=DEFAULT_FEATURES_DIR,
        help=f"Root directory of extracted features (default: {DEFAULT_FEATURES_DIR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing predictions",
    )
    args = parser.parse_args()

    for rep in args.representation:
        if rep == "ensemble":
            if not args.ensemble_of:
                parser.error("--representation ensemble requires --ensemble-of REP [REP ...]")
            evaluate_one(rep, args.features_dir, args.force, ensemble_of=args.ensemble_of)
        else:
            evaluate_one(rep, args.features_dir, args.force)

    if len(args.representation) > 1:
        rows = []
        for rep in args.representation:
            p = args.features_dir / "predictions" / rep / "metrics.json"
            if p.exists():
                rows.append(json.loads(p.read_text()))
        if rows:
            df = pd.DataFrame(rows).set_index("representation")
            cols = ["accuracy", "auc", "eer", "precision", "recall", "f1"]
            print("\nSummary (threshold = 0.5):")
            print(df[cols].round(4).to_string())

if __name__ == "__main__":
    main()
