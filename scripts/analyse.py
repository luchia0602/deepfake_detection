"""
Load saved predictions (produced by evaluate.py) and generate analysis plots:
ROC curves, confusion matrices, per-model error heatmap, PCA scatter plots, and
(optionally) an acoustic error analysis.

Usage:
Single representation: python scripts/analyse.py --representation prosodic
Multiple representations: python scripts/analyse.py --representation prosodic whisper xlsr
Save plots to disk: python scripts/analyse.py --representation prosodic whisper --save-plots
Custom output directory for plots: python scripts/analyse.py --representation prosodic --save-plots --plots-dir results/plots
Run acoustic error analysis too: python scripts/analyse.py --representation mfcc lfcc cqcc prosodic whisper --save-plots --error-analysis
"""

import argparse
import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import mannwhitneyu
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

TARGET_SR = 16_000
DATASET_ID = "AKCIT-Deepfake/BRSpeech-DF"
ACOUSTIC_FEAT_COLS = ["duration_s", "silence_ratio", "spectral_flat",
                      "voiced_ratio", "f0_mean"]


def grid_shape(n, max_cols=3):
    """Rows x cols for n panels, capped at max_cols columns (multi-row layout)."""
    cols = min(max_cols, n)
    rows = math.ceil(n / cols)
    return rows, cols


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
    """Save a figure. `name` includes its extension (.pdf or .png)."""
    if save:
        plots_dir.mkdir(parents=True, exist_ok=True)
        path = plots_dir / name
        if name.lower().endswith(".pdf"):
            fig.savefig(path, bbox_inches="tight")
        else:
            fig.savefig(path, dpi=200, bbox_inches="tight")
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
    savefig(fig, plots_dir, "roc_curves.pdf", save)


def plot_confusion_matrices(reps, y_test, preds_dict, plots_dir, save):
    n = len(reps)
    rows, cols = grid_shape(n, max_cols=3)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    axes_flat = axes.ravel()
    for ax, rep in zip(axes_flat, reps):
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
    # hide any unused panels
    for ax in axes_flat[n:]:
        ax.axis("off")
    plt.suptitle("Confusion matrices — test set", y=1.0)
    plt.tight_layout()
    savefig(fig, plots_dir, "confusion_matrices.pdf", save)


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
    savefig(fig, plots_dir, "error_heatmap.pdf", save)


def plot_pca_by_label(reps, y_test, X_test_dict, plots_dir, save, n_plot=2000):
    rng = np.random.default_rng(42)
    idx = rng.choice(len(y_test), size=min(n_plot, len(y_test)), replace=False)
    n = len(reps)
    rows, cols = grid_shape(n, max_cols=3)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    axes_flat = axes.ravel()
    for ax, rep in zip(axes_flat, reps):
        X_sub = X_test_dict[rep][idx]
        y_sub = y_test[idx]
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X_sub)
        var = pca.explained_variance_ratio_
        for label, name, color in [(0, "bonafide", "steelblue"), (1, "spoof", "tomato")]:
            mask = y_sub == label
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=color, label=name,
                       alpha=0.3, s=8, linewidths=0)
        ax.set_title(f"{rep}\n(PC1={var[0]:.1%}, PC2={var[1]:.1%})")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(markerscale=2)
    for ax in axes_flat[n:]:
        ax.axis("off")
    plt.suptitle("PCA of test set features — bonafide vs spoof", y=1.0)
    plt.tight_layout()
    # PNG: this plot has many thousands of scatter points; a vector PDF would be heavy
    savefig(fig, plots_dir, "pca_by_label.png", save)


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


# ---------------------------------------------------------------------------
# Error analysis: acoustic profiling of misclassified samples
# ---------------------------------------------------------------------------
def extract_acoustic_profile(audio, sr):
    """Per-utterance acoustic descriptors used to characterise hard samples."""
    import librosa
    from amfm_decompy import pYAAPT, basic_tools

    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR
    if len(audio) < 400:
        audio = np.pad(audio, (0, 400 - len(audio)))

    duration_s = len(audio) / sr

    rms = librosa.feature.rms(y=audio, frame_length=400, hop_length=160)[0]
    silence_thresh = 0.01 * rms.max() if rms.max() > 0 else 0.0
    silence_ratio = float((rms < silence_thresh).mean())

    flat = librosa.feature.spectral_flatness(y=audio, hop_length=160)[0]
    spectral_flat = float(flat.mean())

    try:
        signal = basic_tools.SignalObj(audio, sr)
        pitch = pYAAPT.yaapt(signal, frame_length=25.0, frame_space=10.0)
        f0_frames = pitch.samp_values
        voiced_mask = f0_frames > 0
        voiced_ratio = float(voiced_mask.mean()) if len(f0_frames) > 0 else 0.0
        f0_mean = float(f0_frames[voiced_mask].mean()) if voiced_mask.any() else 0.0
    except (IndexError, ValueError):
        voiced_ratio, f0_mean = 0.0, 0.0

    return dict(duration_s=duration_s, silence_ratio=silence_ratio,
                spectral_flat=spectral_flat, voiced_ratio=voiced_ratio,
                f0_mean=f0_mean)


def load_acoustic_profiles(features_dir, meta_dir, y_test):
    """Load cached acoustic profiles, or stream the test audio once to build them."""
    cache = features_dir / "test" / "acoustic_profiles.npy"
    if cache.exists():
        print("Loading cached acoustic profiles ...")
        profiles = list(np.load(cache, allow_pickle=True))
    else:
        from tqdm import tqdm
        from loader import SplitLoader

        loader = SplitLoader(dataset_id=DATASET_ID, splits_dir=str(meta_dir))
        profiles = []
        n_test = loader.stats("test")["total"]
        for audio, sr, label, meta in tqdm(loader.stream("test"),
                                           total=n_test, desc="Acoustic profiling"):
            profiles.append(extract_acoustic_profile(audio, sr))
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, profiles)
        print(f"Saved {len(profiles)} profiles -> {cache}")

    acoustic_df = pd.DataFrame(profiles)
    acoustic_df["y_true"] = y_test
    return acoustic_df


def compute_correctness(reps, y_test, preds_dict):
    """Per-rep correctness masks + the 'universally hard' mask (wrong by ALL reps)."""
    correct = {rep: (preds_dict[rep] == y_test) for rep in reps}
    universally_hard = np.ones(len(y_test), dtype=bool)
    for rep in reps:
        universally_hard &= ~correct[rep]

    print(f"Test set size      : {len(y_test):,}")
    for rep in reps:
        n_wrong = int((~correct[rep]).sum())
        print(f"  {rep:<12}  errors = {n_wrong:,}  ({100 * n_wrong / len(y_test):.1f}%)")
    print(f"  {'universally hard':<12}  = {int(universally_hard.sum()):,}  "
          f"({100 * universally_hard.mean():.1f}% -- wrong by ALL reps)")
    return correct, universally_hard


def run_mwu(acoustic_df, mask_error, mask_correct, label, feat_cols):
    """Mann-Whitney U comparing correct vs error groups for one error definition."""
    rows = []
    for feat in feat_cols:
        vals_c = acoustic_df.loc[mask_correct, feat].values
        vals_e = acoustic_df.loc[mask_error, feat].values
        if len(vals_e) < 5:
            continue
        _, p = mannwhitneyu(vals_c, vals_e, alternative="two-sided")
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        rows.append(dict(group=label, feature=feat,
                         median_correct=float(np.median(vals_c)),
                         median_error=float(np.median(vals_e)),
                         p_value=float(p), sig=sig))
        print(f"  {feat:<16}  correct={np.median(vals_c):.4f}  "
              f"error={np.median(vals_e):.4f}  p={p:.3e}  {sig}")
    return rows


def statistical_comparison(reps, acoustic_df, correct, universally_hard, feat_cols):
    """Run MWU for (A) per-representation errors and (B) universally hard samples."""
    all_results = []
    print("-- A) Per-representation errors ----------------------------------------")
    for rep in reps:
        print(f"\n  [{rep}]")
        all_results.extend(
            run_mwu(acoustic_df, ~correct[rep], correct[rep], rep, feat_cols)
        )
    print("\n-- B) Universally hard (wrong by ALL reps) vs. rest --------------------")
    all_results.extend(
        run_mwu(acoustic_df, universally_hard, ~universally_hard,
                "universally_hard", feat_cols)
    )
    return pd.DataFrame(all_results)


def _plot_distributions(acoustic_df, mask_error, mask_correct, title,
                        ax_row, results_sub, feat_cols):
    for i, feat in enumerate(feat_cols):
        ax = ax_row[i]
        vals_c = acoustic_df.loc[mask_correct, feat].values
        vals_e = acoustic_df.loc[mask_error, feat].values
        lo = np.percentile(np.concatenate([vals_c, vals_e]), 1)
        hi = np.percentile(np.concatenate([vals_c, vals_e]), 99)
        vals_c = np.clip(vals_c, lo, hi)
        vals_e = np.clip(vals_e, lo, hi)
        ax.hist(vals_c, bins=35, alpha=0.55, color="steelblue", density=True,
                label=f"correct (n={len(vals_c):,})")
        ax.hist(vals_e, bins=35, alpha=0.55, color="tomato", density=True,
                label=f"error (n={len(vals_e):,})")
        row = results_sub[results_sub.feature == feat]
        p_str = f"p={row.p_value.values[0]:.2e} {row.sig.values[0]}" if len(row) else ""
        ax.set_title(f"{title}\n{feat}\n{p_str}", fontsize=8)
        if i == 0:
            ax.legend(fontsize=7)


def plot_error_distributions(reps, acoustic_df, correct, universally_hard,
                             results_df, feat_cols, plots_dir, save):
    n_reps = len(reps)
    n_feats = len(feat_cols)

    # Part A: per-representation grid (feature rows x representation columns)
    # PNG: this grid has many thousands of histogram bars; PDF would be heavy.
    fig_a, axes_a = plt.subplots(n_feats, n_reps,
                                 figsize=(3.8 * n_reps, 3.2 * n_feats),
                                 squeeze=False)
    for j, rep in enumerate(reps):
        rsub = results_df[results_df.group == rep]
        _plot_distributions(acoustic_df, ~correct[rep], correct[rep],
                            rep, axes_a[:, j], rsub, feat_cols)
    fig_a.suptitle("A) Per-representation: correct vs. misclassified",
                   y=1.01, fontsize=11)
    plt.tight_layout()
    savefig(fig_a, plots_dir, "error_dist_per_rep.png", save)

    # Part B: universally hard vs. rest (one column per feature)
    fig_b, axes_b = plt.subplots(1, n_feats, figsize=(3.8 * n_feats, 3.5),
                                 squeeze=False)
    rsub_b = results_df[results_df.group == "universally_hard"]
    _plot_distributions(acoustic_df, universally_hard, ~universally_hard,
                        "universally hard", axes_b[0], rsub_b, feat_cols)
    fig_b.suptitle("B) Universally hard (wrong by ALL reps) vs. rest", fontsize=11)
    plt.tight_layout()
    savefig(fig_b, plots_dir, "error_dist_universal.pdf", save)


def fit_misclassification_lr(X_acoustic, y_hard, label, feat_cols):
    """5-fold CV logistic regression predicting misclassification from acoustics."""
    if y_hard.sum() < 10:
        print(f"  {label:<20}  too few errors ({int(y_hard.sum())}) -- skipping")
        return None
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    sc = StandardScaler()
    for tr, va in cv.split(X_acoustic, y_hard):
        clf = LogisticRegression(max_iter=500, C=1.0)
        clf.fit(sc.fit_transform(X_acoustic[tr]), y_hard[tr])
        aucs.append(roc_auc_score(
            y_hard[va], clf.predict_proba(sc.transform(X_acoustic[va]))[:, 1]))
    clf_full = LogisticRegression(max_iter=500, C=1.0)
    clf_full.fit(StandardScaler().fit_transform(X_acoustic), y_hard)
    coefs = dict(zip(feat_cols, clf_full.coef_[0]))
    top = max(coefs, key=lambda k: abs(coefs[k]))
    direction = "harder" if coefs[top] > 0 else "easier"
    print(f"  {label:<22}  AUC={np.mean(aucs):.3f}+/-{np.std(aucs):.3f}"
          f"  top={top} ({direction}, {coefs[top]:+.3f})")
    return {"group": label, **coefs,
            "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs))}


def plot_lr_coefficients(reps, acoustic_df, correct, universally_hard,
                         feat_cols, plots_dir, save):
    X_acoustic = acoustic_df[feat_cols].values.astype(np.float32)
    np.nan_to_num(X_acoustic, copy=False)

    print("-- A) Per-representation -----------------------------------------------")
    lr_records = []
    for rep in reps:
        rec = fit_misclassification_lr(X_acoustic, (~correct[rep]).astype(int),
                                       rep, feat_cols)
        if rec:
            lr_records.append(rec)

    print("\n-- B) Universally hard -------------------------------------------------")
    rec = fit_misclassification_lr(X_acoustic, universally_hard.astype(int),
                                   "universally_hard", feat_cols)
    if rec:
        lr_records.append(rec)

    if not lr_records:
        print("No LR records to plot -- skipping coefficient heatmap.")
        return

    lr_df = pd.DataFrame(lr_records).set_index("group")
    coef_vals = lr_df[feat_cols].values
    vmax = np.abs(coef_vals).max()

    fig, ax = plt.subplots(figsize=(len(feat_cols) * 1.5, len(lr_df) * 0.85 + 1.2))
    im = ax.imshow(coef_vals, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(feat_cols)))
    ax.set_xticklabels(feat_cols, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(lr_df)))
    ax.set_yticklabels(lr_df.index, fontsize=9)
    plt.colorbar(im, ax=ax, label="LR coefficient  (red = harder, blue = easier)")

    for i, (grp, row) in enumerate(lr_df.iterrows()):
        for j, feat in enumerate(feat_cols):
            ax.text(j, i, f"{row[feat]:.2f}", ha="center", va="center", fontsize=8)
        ax.text(-0.6, i, f"AUC={row['auc_mean']:.3f}", ha="right", va="center",
                fontsize=8, color="dimgray")

    ax.set_title("LR coefficients: acoustic predictors of misclassification\n"
                 "(universally_hard = wrong by ALL reps)", fontsize=10)
    plt.tight_layout()
    savefig(fig, plots_dir, "error_lr_coefs.pdf", save)


def run_error_analysis(reps, y_test, preds_dict, features_dir, meta_dir,
                       plots_dir, save):
    print("\n=== Error analysis ===")
    correct, universally_hard = compute_correctness(reps, y_test, preds_dict)

    acoustic_df = load_acoustic_profiles(features_dir, meta_dir, y_test)
    print("\nAcoustic profile summary:")
    print(acoustic_df.describe().round(4).to_string())

    print("\nStatistical comparison (Mann-Whitney U):")
    results_df = statistical_comparison(reps, acoustic_df, correct,
                                        universally_hard, ACOUSTIC_FEAT_COLS)

    print("\nGenerating error-distribution plots")
    plot_error_distributions(reps, acoustic_df, correct, universally_hard,
                             results_df, ACOUSTIC_FEAT_COLS, plots_dir, save)

    print("\nFitting misclassification LR")
    plot_lr_coefficients(reps, acoustic_df, correct, universally_hard,
                         ACOUSTIC_FEAT_COLS, plots_dir, save)


# ---------------------------------------------------------------------------
# DeLong test for the difference between two correlated ROC AUCs
# (same test set, so the AUCs are not independent).
# Implementation follows Sun & Xu (2014), "Fast Implementation of DeLong's
# Algorithm for Comparing the Areas Under Correlated ROC Curves".
# ---------------------------------------------------------------------------
def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive = predictions_sorted_transposed[:, :m]
    negative = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = _compute_midrank(positive[r, :])
        ty[r, :] = _compute_midrank(negative[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_roc_test(y_true, scores_a, scores_b):
    """Return the two-sided p-value for AUC(a) == AUC(b) on the same labels."""
    from scipy import stats

    pos = y_true == 1
    idx = np.concatenate([np.where(pos)[0], np.where(~pos)[0]])
    label_1_count = int(np.sum(pos))
    preds = np.vstack([scores_a[idx], scores_b[idx]])
    aucs, cov = _fast_delong(preds, label_1_count)
    l = np.array([[1, -1]])
    var = l @ cov @ l.T
    if var <= 0:
        return float(aucs[0]), float(aucs[1]), 1.0
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(aucs[0]), float(aucs[1]), float(p[0][0] if np.ndim(p) else p)


def auc_significance_matrix(reps, probs_dict, y_test, plots_dir, save):
    """Pairwise DeLong p-values for all representation AUC differences."""
    mat = pd.DataFrame(np.nan, index=reps, columns=reps)
    for i, a in enumerate(reps):
        for b in reps[i + 1:]:
            _, _, pval = delong_roc_test(y_test, probs_dict[a], probs_dict[b])
            mat.loc[a, b] = pval
            mat.loc[b, a] = pval
    print("\nPairwise DeLong AUC significance (p-values):")
    print(mat.round(4).to_string())
    if save:
        plots_dir.mkdir(parents=True, exist_ok=True)
        out = plots_dir / "auc_significance.csv"
        mat.to_csv(out)
        print(f"  saved \u2192 {out}")
    return mat


def main():
    parser = argparse.ArgumentParser(description="Analyse saved predictions and generate plots.")
    parser.add_argument(
        "--representation",
        nargs="+",
        required=True,
        metavar="REP",
        help="One or more feature representations, e.g. prosodic whisper xlsr ensemble",
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
    parser.add_argument(
        "--error-analysis",
        action="store_true",
        help="Run acoustic error analysis (streams test audio once if not cached): "
             "MWU tests, distribution plots, misclassification-LR coefficient heatmap",
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

    if len(reps) > 1:
        auc_significance_matrix(reps, probs_dict, y_test, args.plots_dir, args.save_plots)
    print("Generating plots")
    plot_roc(reps, y_test, probs_dict, args.plots_dir, args.save_plots)
    plot_confusion_matrices(reps, y_test, preds_dict, args.plots_dir, args.save_plots)
    plot_error_heatmap(reps, y_test, preds_dict, models, args.plots_dir, args.save_plots)

    if not args.skip_pca:
        X_test_dict = {rep: load_test_features(args.features_dir, rep) for rep in reps}
        plot_pca_by_label(reps, y_test, X_test_dict, args.plots_dir, args.save_plots)

    if args.error_analysis:
        run_error_analysis(reps, y_test, preds_dict, args.features_dir,
                           args.meta_dir, args.plots_dir, args.save_plots)


if __name__ == "__main__":
    main()
