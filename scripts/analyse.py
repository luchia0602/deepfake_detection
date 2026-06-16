"""
Load saved predictions (produced by evaluate.py) and generate analysis plots: ROC curves, Confusion matrices, Per-model accuracy bar chart, Error analysis summary, PCA scatter plots.

Usage:
Single representation: python scripts/analyse.py --representation prosodic
Multiple representations: python scripts/analyse.py --representation prosodic whisper xlsr
Save plots to disk: python scripts/analyse.py --representation prosodic whisper --save-plots
Custom output directory for plots: python scripts/analyse.py --representation prosodic --save-plots --plots-dir results/plots
"""

import argparse
from pathlib import Path
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

DEFAULT_FEATURES_DIR = Path("artifacts/features")
DEFAULT_META_DIR = Path("artifacts")
DEFAULT_PLOTS_DIR = Path("results/plots")


def compute_eer(y_true, scores):
    fpr, tpr, _ = roc_curve(y_true, scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2)


def load_predictions(features_dir: Path, rep: str):
    pred_dir = features_dir / "predictions" / rep
    probs = np.load(pred_dir / "probs_test.npy")
    preds = np.load(pred_dir / "preds_test.npy")
    return probs, preds


def load_test_features(features_dir: Path, rep: str):
    X = np.load(features_dir / "test" / f"X_{rep}.npy").astype(np.float32)
    np.nan_to_num(X, copy=False)
    return X


def savefig(fig, plots_dir: Path, name: str, save: bool):
    if save:
        plots_dir.mkdir(parents=True, exist_ok=True)
        path = plots_dir / name
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  saved → {path}")
    else:
        plt.show()
    plt.close(fig)

def plot_roc(reps, y_test, probs_dict, plots_dir, save):
    fig, ax = plt.subplots(figsize=(6, 5))
    for rep in reps:
        fpr, tpr, _ = roc_curve(y_test, probs_dict[rep])
        auc = roc_auc_score(y_test, probs_dict[rep])
        ax.plot(fpr, tpr, label=f"{rep} (AUC={auc:.3f})", alpha=0.8)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves")
    ax.legend()
    plt.tight_layout()
    savefig(fig, plots_dir, "roc_curves.png", save)


def plot_confusion_matrices(reps, y_test, preds_dict, plots_dir, save):
    n = len(reps)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    axes = axes.ravel()
    for ax, rep in zip(axes, reps):
        cm = confusion_matrix(y_test, preds_dict[rep])
        disp = ConfusionMatrixDisplay(cm, display_labels=["bonafide", "spoof"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        total = cm.sum()
        for text_obj, (i, j) in zip(
            disp.text_.ravel(),
            [(i, j) for i in range(2) for j in range(2)],
        ):
            count = cm[i, j]
            text_obj.set_text(f"{count}\n({100 * count / total:.1f}%)")
        ax.set_title(
            f"{rep}\n"
            f"Acc={accuracy_score(y_test, preds_dict[rep]):.3f}  "
            f"F1={f1_score(y_test, preds_dict[rep]):.3f}"
        )
    plt.suptitle("Confusion matrices — test set", y=1.02)
    plt.tight_layout()
    savefig(fig, plots_dir, "confusion_matrices.png", save)


def plot_per_model_accuracy(reps, y_test, preds_dict, models, plots_dir, save):
    model_names = sorted(set(models))
    rows = []
    for model_name in model_names:
        mask = models == model_name
        y_m = y_test[mask]
        for rep in reps:
            pred_m = preds_dict[rep][mask]
            rows.append({
                "Model":    model_name,
                "Feature":  rep,
                "N":        int(mask.sum()),
                "Accuracy": accuracy_score(y_m, pred_m),
                "F1":       f1_score(y_m, pred_m, zero_division=0),
            })
    df_model = pd.DataFrame(rows)
    df_pivot = df_model.pivot(index="Model", columns="Feature", values="Accuracy")
    ax = df_pivot.plot(kind="bar", figsize=(9, 4), width=0.7)
    ax.axhline(0.5, color="red", linestyle="--", linewidth=0.8, label="chance")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-model accuracy by feature representation")
    ax.set_xlabel("")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    savefig(ax.get_figure(), plots_dir, "per_model_accuracy.png", save)


def plot_error_heatmap(reps, y_test, preds_dict, models, plots_dir, save):
    model_names = sorted(set(models))
    error_matrix = np.zeros((len(model_names), len(reps)))
    for i, model_name in enumerate(model_names):
        mask = models == model_name
        for j, rep in enumerate(reps):
            error_matrix[i, j] = 1 - (preds_dict[rep][mask] == y_test[mask]).mean()
    fig, ax = plt.subplots(figsize=(max(6, len(reps) * 2), max(4, len(model_names) * 0.5 + 1)))
    im = ax.imshow(error_matrix, cmap="Reds", vmin=0, vmax=0.5)
    ax.set_xticks(range(len(reps)))
    ax.set_xticklabels(reps)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names)
    plt.colorbar(im, ax=ax, label="Error rate")
    ax.set_title("Error rate by model and feature")
    for i in range(len(model_names)):
        for j in range(len(reps)):
            ax.text(j, i, f"{error_matrix[i, j]:.2f}", ha="center", va="center", fontsize=9)
    plt.tight_layout()
    savefig(fig, plots_dir, "error_heatmap.png", save)


def plot_pca_by_label(reps, y_test, X_test_dict, plots_dir, save, n_plot=2000):
    rng = np.random.default_rng(42)
    idx = rng.choice(len(y_test), size=min(n_plot, len(y_test)), replace=False)
    n = len(reps)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    axes = axes.ravel()
    for ax, rep in zip(axes, reps):
        X_sub = X_test_dict[rep][idx]
        y_sub = y_test[idx]
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X_sub)
        var = pca.explained_variance_ratio_
        for label, name, color in [(0, "bonafide", "steelblue"), (1, "spoof", "tomato")]:
            mask = y_sub == label
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=color, label=name, alpha=0.3, s=8, linewidths=0)
        ax.set_title(f"{rep}\n(PC1={var[0]:.1%}, PC2={var[1]:.1%})")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(markerscale=2)
    plt.suptitle("PCA of test set features — bonafide vs spoof", y=1.02)
    plt.tight_layout()
    savefig(fig, plots_dir, "pca_by_label.png", save)


def plot_pca_by_model(reps, y_test, X_test_dict, models, plots_dir, save, n_plot=2000):
    rng = np.random.default_rng(42)
    idx = rng.choice(len(y_test), size=min(n_plot, len(y_test)), replace=False)
    model_list = sorted(set(models))
    colors = cm.tab10(np.linspace(0, 1, len(model_list)))
    n = len(reps)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    axes = axes.ravel()
    for ax, rep in zip(axes, reps):
        X_sub = X_test_dict[rep][idx]
        m_sub = models[idx]
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X_sub)
        for model_name, color in zip(model_list, colors):
            mask = m_sub == model_name
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[color], label=model_name, alpha=0.4, s=8, linewidths=0)
        ax.set_title(rep)
        ax.set_xlabel("PC1")
        ax.legend(markerscale=2, fontsize=7)
    plt.suptitle("PCA of test set features — colored by model", y=1.02)
    plt.tight_layout()
    savefig(fig, plots_dir, "pca_by_model.png", save)


def print_error_summary(reps, y_test, preds_dict, models):
    n_reps = len(reps)
    df_errors = pd.DataFrame({"y_true": y_test, "model": models})
    for rep in reps:
        df_errors[f"correct_{rep}"] = (preds_dict[rep] == y_test).astype(int)
    correct_cols = [f"correct_{r}" for r in reps]
    df_errors["n_correct"] = df_errors[correct_cols].sum(axis=1)
    all_wrong = df_errors[df_errors["n_correct"] == 0]
    all_right = df_errors[df_errors["n_correct"] == n_reps]
    disagreed = df_errors[(df_errors["n_correct"] > 0) & (df_errors["n_correct"] < n_reps)]

    n = len(df_errors)
    print("Error analysis:")
    print(f"All correct   : {len(all_right):>6} ({100*len(all_right)/n:.1f}%)")
    print(f"All wrong     : {len(all_wrong):>6} ({100*len(all_wrong)/n:.1f}%)")
    print(f"Disagreement  : {len(disagreed):>6} ({100*len(disagreed)/n:.1f}%)")
    if len(all_wrong):
        print("All-wrong breakdown by model:")
        print(all_wrong.groupby("model").size().sort_values(ascending=False).to_string())
        print("All-wrong breakdown by true label:")
        print(all_wrong["y_true"].value_counts().rename({0: "bonafide", 1: "spoof"}).to_string())


def main():
    parser = argparse.ArgumentParser(description="Analyse saved predictions and generate plots.")
    parser.add_argument(
        "--representation",
        nargs="+",
        required=True,
        metavar="REP",
        help="One or more feature representations, e.g. prosodic whisper xlsr",
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=DEFAULT_FEATURES_DIR,
        help=f"Root features directory (default: {DEFAULT_FEATURES_DIR})",
    )
    parser.add_argument(
        "--meta-dir",
        type=Path,
        default=DEFAULT_META_DIR,
        help=f"Directory containing test.csv metadata (default: {DEFAULT_META_DIR})",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save plots to disk instead of displaying interactively",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=DEFAULT_PLOTS_DIR,
        help=f"Where to save plots (default: {DEFAULT_PLOTS_DIR})",
    )
    parser.add_argument(
        "--skip-pca",
        action="store_true",
        help="Skip PCA plots (slow for large/high-dimensional features)",
    )
    args = parser.parse_args()

    reps = args.representation

    probs_dict = {}
    preds_dict = {}
    for rep in reps:
        pred_dir = args.features_dir / "predictions" / rep
        if not (pred_dir / "probs_test.npy").exists():
            raise FileNotFoundError(
                f"No predictions found for '{rep}' at {pred_dir}. "
                "Run evaluate.py first."
            )
        probs_dict[rep], preds_dict[rep] = load_predictions(args.features_dir, rep)

    y_test = np.load(args.features_dir / "test" / "y.npy")
    test_meta = pd.read_csv(args.meta_dir / "test.csv")
    models = test_meta["model"].values
    rows = []
    for rep in reps:
        rows.append({
            "Feature":   rep,
            "Accuracy":  accuracy_score(y_test, preds_dict[rep]),
            "AUC":       roc_auc_score(y_test, probs_dict[rep]),
            "EER":       compute_eer(y_test, probs_dict[rep]),
            "Precision": float(precision_score(y_test, preds_dict[rep])),
            "Recall":    float(recall_score(y_test, preds_dict[rep])),
            "F1":        f1_score(y_test, preds_dict[rep]),
        })
    df_results = pd.DataFrame(rows).set_index("Feature")
    print("Metrics:")
    print(df_results.round(4).to_string())
    print_error_summary(reps, y_test, preds_dict, models)
    print("Generating plots")
    plot_roc(reps, y_test, probs_dict, args.plots_dir, args.save_plots)
    plot_confusion_matrices(reps, y_test, preds_dict, args.plots_dir, args.save_plots)
    plot_per_model_accuracy(reps, y_test, preds_dict, models, args.plots_dir, args.save_plots)
    plot_error_heatmap(reps, y_test, preds_dict, models, args.plots_dir, args.save_plots)

    if not args.skip_pca:
        X_test_dict = {rep: load_test_features(args.features_dir, rep) for rep in reps}
        plot_pca_by_label(reps, y_test, X_test_dict, args.plots_dir, args.save_plots)
        plot_pca_by_model(reps, y_test, X_test_dict, models, args.plots_dir, args.save_plots)


if __name__ == "__main__":
    main()