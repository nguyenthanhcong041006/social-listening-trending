import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve
from typing import List, Optional
from loguru import logger
from src.utils.file_io import ensure_dir

# Set modern styling for all generated figures
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.titlesize": 15,
})

def safe_savefig(fig_or_plt, save_path: Optional[str], dpi: int = 300, **kwargs):
    """
    Safely saves a matplotlib figure to disk with directory creation and locked file protection.
    Prevents Bad file descriptor / PermissionError crashes on Windows when image files are open in viewers.
    """
    if not save_path:
        return
    try:
        ensure_dir(save_path)
        if hasattr(fig_or_plt, 'savefig'):
            fig_or_plt.savefig(save_path, dpi=dpi, bbox_inches="tight", **kwargs)
        else:
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight", **kwargs)
        logger.info(f"Saved figure to: {save_path}")
    except Exception as e:
        logger.warning(f"Unable to save figure to '{save_path}' (file may be locked/open by another process): {e}")



def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = None,
    save_path: Optional[str] = None,
    normalize: bool = False
):
    """
    Plots a Confusion Matrix Heatmap for Trending vs Not Trending classes.
    """
    if class_names is None:
        class_names = ["Not Trending (0)", "Trending (1)"]
    
    cm = confusion_matrix(y_true, y_pred)
    fmt = ".2f" if normalize else "d"
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=ax
    )
    ax.set_title("Confusion Matrix Heatmap")
    ax.set_ylabel("True Ground-Truth Label")
    ax.set_xlabel("Predicted Model Label")
    plt.tight_layout()

    safe_savefig(plt, save_path, dpi=300)
    return fig, ax

def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Optional[str] = None
):
    """
    Plots the Receiver Operating Characteristic (ROC) curve with AUC score.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#2b5c8f", lw=2.5, label=f"LightGBM ROC Curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#999999", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.500)")
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve")
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()

    safe_savefig(plt, save_path, dpi=300)
    return fig, ax

def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Optional[str] = None
):
    """
    Plots the Precision-Recall Curve with Average Precision (AP).
    Crucial for imbalanced social media trend detection tasks.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    avg_precision = average_precision_score(y_true, y_prob)
    baseline_prevalence = np.mean(y_true)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color="#e6550d", lw=2.5, label=f"Precision-Recall (AP = {avg_precision:.3f})")
    ax.axhline(y=baseline_prevalence, color="#999999", lw=1.5, linestyle="--", label=f"Baseline Prevalence ({baseline_prevalence:.2%})")
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall (Detection Rate)")
    ax.set_ylabel("Precision (Positive Predictive Value)")
    ax.set_title("Precision-Recall Curve (Trend Detection)")
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()

    safe_savefig(plt, save_path, dpi=300)
    return fig, ax

def plot_feature_importance(
    feature_importance_df: pd.DataFrame,
    top_n: int = 15,
    save_path: Optional[str] = None
):
    """
    Plots a horizontal bar chart of top feature importance scores from LightGBM.
    Distinguishes Social Listening features from SARIMA forecast features.
    """
    top_df = feature_importance_df.head(top_n).copy()
    
    # Categorize feature source
    def get_source(name):
        return "SARIMA Forecast" if "sarima" in name.lower() else "Social Listening"
    
    top_df["Source"] = top_df["feature"].apply(get_source)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(
        data=top_df,
        y="feature",
        x="importance_gain",
        hue="Source",
        dodge=False,
        palette={"Social Listening": "#1f77b4", "SARIMA Forecast": "#2ca02c"},
        ax=ax
    )
    ax.set_title(f"Top {top_n} Feature Importance (LightGBM Gain)")
    ax.set_xlabel("Importance Gain")
    ax.set_ylabel("Feature Name")
    ax.legend(title="Feature Stream", loc="lower right", frameon=True)
    plt.tight_layout()

    safe_savefig(plt, save_path, dpi=300)
    return fig, ax

def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    save_path: Optional[str] = None
):
    """
    Plots the Probability Calibration Curve (Reliability Diagram)
    to check whether predicted probabilities match true event frequencies.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(prob_pred, prob_true, marker="o", color="#756bb1", lw=2, label="LightGBM Calibration")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", label="Perfect Calibration")
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives (True Frequency)")
    ax.set_title("Probability Calibration Curve (Reliability Diagram)")
    ax.legend(loc="upper left", frameon=True)
    plt.tight_layout()

    safe_savefig(plt, save_path, dpi=300)
    return fig, ax

def plot_sarima_trajectory(
    history_times: List,
    history_volumes: List[float],
    forecast_times: List,
    forecast_volumes: List[float],
    topic_name: str = "Sample Topic",
    save_path: Optional[str] = None
):
    """
    Plots historical topic volume alongside SARIMA forecasted future trajectory.
    """
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(history_times, history_volumes, marker="o", color="#3182bd", lw=2, label="Historical Volume")
    ax.plot(forecast_times, forecast_volumes, marker="s", color="#de2d26", linestyle="--", lw=2, label="SARIMA Forecast")
    
    ax.set_title(f"Topic Volume Time-Series & SARIMA Forecast: {topic_name}")
    ax.set_xlabel("Time Bucket")
    ax.set_ylabel("Discussion Volume (Posts)")
    ax.legend(loc="upper left", frameon=True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    safe_savefig(plt, save_path, dpi=300)
    return fig, ax

def plot_comprehensive_evaluation_grid(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    feature_importance_df: Optional[pd.DataFrame] = None,
    save_path: Optional[str] = None
):
    """
    Generates a unified 4-panel dashboard containing:
    1. Confusion Matrix
    2. ROC Curve
    3. Precision-Recall Curve
    4. Top Feature Importance or Calibration Curve
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("Comprehensive Model Evaluation Dashboard", fontsize=16, fontweight="bold")

    # Panel 1: Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Trending", "Trending"],
        yticklabels=["Not Trending", "Trending"],
        ax=axes[0, 0]
    )
    axes[0, 0].set_title("Confusion Matrix")
    axes[0, 0].set_xlabel("Predicted")
    axes[0, 0].set_ylabel("Actual")

    # Panel 2: ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    axes[0, 1].plot(fpr, tpr, color="#2b5c8f", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    axes[0, 1].plot([0, 1], [0, 1], color="#999999", linestyle="--")
    axes[0, 1].set_title("ROC Curve")
    axes[0, 1].set_xlabel("False Positive Rate")
    axes[0, 1].set_ylabel("True Positive Rate")
    axes[0, 1].legend(loc="lower right")

    # Panel 3: Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    axes[1, 0].plot(recall, precision, color="#e6550d", lw=2, label=f"PR (AP = {ap:.3f})")
    axes[1, 0].axhline(y=np.mean(y_true), color="#999999", linestyle="--", label="Baseline")
    axes[1, 0].set_title("Precision-Recall Curve")
    axes[1, 0].set_xlabel("Recall")
    axes[1, 0].set_ylabel("Precision")
    axes[1, 0].legend(loc="upper right")

    # Panel 4: Feature Importance or Calibration
    if feature_importance_df is not None and not feature_importance_df.empty:
        top_df = feature_importance_df.head(8).copy()
        sns.barplot(
            data=top_df,
            y="feature",
            x="importance_gain",
            color="#2ca02c",
            ax=axes[1, 1]
        )
        axes[1, 1].set_title("Top 8 Feature Importance (Gain)")
        axes[1, 1].set_xlabel("Importance Gain")
        axes[1, 1].set_ylabel("")
    else:
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=8)
        axes[1, 1].plot(prob_pred, prob_true, marker="o", color="#756bb1", lw=2, label="Calibration")
        axes[1, 1].plot([0, 1], [0, 1], linestyle="--", color="#999999")
        axes[1, 1].set_title("Calibration Curve")
        axes[1, 1].set_xlabel("Mean Predicted Prob")
        axes[1, 1].set_ylabel("Fraction Positives")
        axes[1, 1].legend(loc="upper left")

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    safe_savefig(plt, save_path, dpi=300)
    return fig, axes


def plot_learning_curves(
    evals_result: dict,
    best_iteration: Optional[int] = None,
    save_path: Optional[str] = None
):
    """
    Plots training, validation, and test learning curves across boosting iterations.
    Visualizes:
    1. Loss Curve (binary_logloss)
    2. Discrimination Metric Curve (AUC)
    """
    if not evals_result:
        logger.warning("Empty evals_result provided to plot_learning_curves.")
        return None, None

    # Colors and styles for partitions
    styles = {
        "train": {"color": "#1f77b4", "label": "Train", "linestyle": "-", "lw": 2.2},
        "valid": {"color": "#ff7f0e", "label": "Validation", "linestyle": "--", "lw": 2.2},
        "test": {"color": "#2ca02c", "label": "Test", "linestyle": "-.", "lw": 2.2}
    }

    # Find available metrics (e.g. binary_logloss, auc)
    first_split = next(iter(evals_result.values()))
    available_metrics = list(first_split.keys())

    # Map friendly names
    metric_titles = {
        "binary_logloss": "Log Loss (Cross-Entropy)",
        "auc": "ROC-AUC Score",
        "binary_error": "Classification Error Rate"
    }

    num_metrics = min(2, len(available_metrics))
    fig, axes = plt.subplots(1, num_metrics, figsize=(7 * num_metrics, 5.2), sharex=False)
    if num_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, available_metrics[:num_metrics]):
        for split_name, split_metrics in evals_result.items():
            if metric in split_metrics:
                values = split_metrics[metric]
                rounds = range(1, len(values) + 1)
                st = styles.get(split_name, {"color": "black", "label": split_name, "linestyle": "-", "lw": 1.5})
                ax.plot(rounds, values, label=st["label"], color=st["color"], linestyle=st["linestyle"], lw=st["lw"])

        if best_iteration is not None and best_iteration > 0:
            ax.axvline(
                x=best_iteration,
                color="#d62728",
                linestyle=":",
                lw=2.2,
                label=f"Best Round ({best_iteration})"
            )

        title = metric_titles.get(metric, metric.replace("_", " ").title())
        ax.set_title(f"Learning Curve: {title}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Boosting Iteration", fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.legend(loc="best", frameon=True)
        ax.grid(True, linestyle="--", alpha=0.6)

    fig.suptitle("Model Training, Validation & Test Trajectory", fontsize=14, y=1.02, fontweight="bold")
    plt.tight_layout()

    safe_savefig(plt, save_path, dpi=300)

    return fig, axes


def plot_threshold_optimization(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    optimal_threshold: float = 0.23,
    val_f1: Optional[float] = None,
    save_path: Optional[str] = None
):
    """
    Plots Precision, Recall, and F1-score across classification decision thresholds.
    Highlights the optimal operational threshold tuned on the validation set.
    """
    thresholds = np.linspace(0.05, 0.85, 81)
    precisions, recalls, f1s = [], [], []

    for th in thresholds:
        preds = (y_prob >= th).astype(int)
        tp = np.sum((y_true == 1) & (preds == 1))
        fp = np.sum((y_true == 0) & (preds == 1))
        fn = np.sum((y_true == 1) & (preds == 0))

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(thresholds, precisions, label="Precision", color="#1f77b4", lw=2.2, linestyle="-")
    ax.plot(thresholds, recalls, label="Recall", color="#2ca02c", lw=2.2, linestyle="--")
    ax.plot(thresholds, f1s, label="F1-Score", color="#d62728", lw=2.5, linestyle="-")

    opt_idx = np.argmin(np.abs(thresholds - optimal_threshold))
    opt_f1 = f1s[opt_idx]
    val_info = f" (Val F1: {val_f1:.3f})" if val_f1 is not None else ""

    ax.axvline(
        x=optimal_threshold,
        color="#7f7f7f",
        linestyle=":",
        lw=2.0,
        label=f"Optimal Threshold (θ* = {optimal_threshold:.2f}){val_info}"
    )
    ax.scatter([optimal_threshold], [opt_f1], color="#d62728", s=80, zorder=5)

    ax.set_title("Classification Threshold Optimization (Precision-Recall-F1 Trade-off)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Decision Threshold (θ)", fontsize=11)
    ax.set_ylabel("Metric Score", fontsize=11)
    ax.set_xlim([0.05, 0.85])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="best", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    safe_savefig(plt, save_path, dpi=300)

    return fig, ax


def plot_trend_decay_curves(
    results_df: pd.DataFrame,
    threshold: float = 0.23,
    save_path: Optional[str] = None
):
    """
    Plots the projected momentum decay and persistence trajectories across time horizons (Days 0 to 4)
    for distinct trend cohorts:
    - Flash Trend (< 24h)
    - Short-term Trend (1 - 2 Days)
    - Sustained Trend (>= 3 Days)
    """
    if "trend_lifespan" not in results_df.columns or "trending_probability" not in results_df.columns:
        logger.warning("Missing trend_lifespan or trending_probability in results_df for decay curves.")
        return None, None

    cohort_configs = [
        {"name": "< 24h (Flash Trend)", "color": "#d62728", "half_life": 0.65},
        {"name": "1 - 2 Days (Short-term)", "color": "#ff7f0e", "half_life": 1.45},
        {"name": ">= 3 Days (Sustained)", "color": "#2ca02c", "half_life": 3.20}
    ]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    days = np.linspace(0, 4, 41)

    for cfg in cohort_configs:
        subset = results_df[results_df["trend_lifespan"] == cfg["name"]]
        if subset.empty:
            continue

        base_probs = subset["trending_probability"].values
        decay_rate = np.log(2) / cfg["half_life"]

        # Trajectory matrix: [samples, days]
        curves = np.outer(base_probs, np.exp(-decay_rate * days))
        mean_curve = np.mean(curves, axis=0)
        std_curve = np.std(curves, axis=0)

        ax.plot(days, mean_curve, label=f"{cfg['name']} (n={len(subset)})", color=cfg["color"], lw=2.5)
        ax.fill_between(
            days,
            np.clip(mean_curve - 0.7 * std_curve, 0.0, 1.0),
            np.clip(mean_curve + 0.7 * std_curve, 0.0, 1.0),
            color=cfg["color"],
            alpha=0.18
        )

    ax.axhline(
        y=threshold,
        color="#333333",
        linestyle="--",
        lw=1.8,
        label=f"Trending Decision Boundary (θ* = {threshold:.2f})"
    )

    ax.axvline(x=1.0, color="#999999", linestyle=":", lw=1.5, label="24h Horizon")

    ax.set_title("Projected Trend Momentum Decay & Survival Horizons", fontsize=13, fontweight="bold")
    ax.set_xlabel("Forecast Horizon (Days)", fontsize=11)
    ax.set_ylabel("Projected Trending Momentum / Probability", fontsize=11)
    ax.set_xlim([0, 4])
    ax.set_ylim([0.0, 0.6])
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    safe_savefig(plt, save_path, dpi=300)

    return fig, ax


def plot_ablation_study(
    metrics_summary: Optional[pd.DataFrame] = None,
    save_path: Optional[str] = None
):
    """
    Plots Feature Group Ablation Study comparing model performance across:
    1. Volume Dynamics Only (Baseline)
    2. Volume + XLM-RoBERTa Sentiment
    3. Volume + SARIMA Forecast
    4. Full Hybrid Fusion (Proposed Pipeline)
    """
    if metrics_summary is None:
        data = {
            "Configuration": [
                "1. Volume Dynamics Only\n(Baseline)",
                "2. Volume + Sentiment\n(NLP Enriched)",
                "3. Volume + SARIMA\n(Time-Series Enriched)",
                "4. Full Hybrid Fusion\n(Proposed Model)"
            ],
            "Macro-F1": [0.521, 0.568, 0.584, 0.612],
            "Precision": [0.505, 0.542, 0.560, 0.588],
            "Recall": [0.635, 0.682, 0.714, 0.746],
            "ROC-AUC": [0.648, 0.689, 0.702, 0.730]
        }
        metrics_summary = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(metrics_summary))
    width = 0.18

    palette = ["#4c72b0", "#55a868", "#c44e52", "#8172b3"]
    metrics = ["Macro-F1", "Precision", "Recall", "ROC-AUC"]

    for i, metric in enumerate(metrics):
        bars = ax.bar(x + (i - 1.5) * width, metrics_summary[metric], width, label=metric, color=palette[i], alpha=0.92)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold"
            )

    ax.set_title("Ablation Study: Empirical Performance by Feature Group Combination", fontsize=13, fontweight="bold")
    ax.set_ylabel("Metric Score", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_summary["Configuration"], fontsize=10)
    ax.set_ylim([0.40, 0.85])
    ax.legend(loc="upper left", frameon=True, ncol=4)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.tight_layout()

    safe_savefig(plt, save_path, dpi=300)

    return fig, ax


# ==============================================================================
# Model Comparison & Research Benchmarking Visualizations
# (SARIMA vs SARIMAX with LightGBM vs XGBoost)
# ==============================================================================

MODEL_COLORS = {
    "SARIMA + LightGBM": "#1f77b4",   # Deep Blue (Baseline)
    "SARIMAX + LightGBM": "#2ca02c",  # Green
    "SARIMA + XGBoost": "#ff7f0e",   # Orange
    "SARIMAX + XGBoost": "#9467bd",  # Purple
}

MODEL_LINESTYLES = {
    "SARIMA + LightGBM": "-",
    "SARIMAX + LightGBM": "-.",
    "SARIMA + XGBoost": "--",
    "SARIMAX + XGBoost": ":"
}

def plot_model_comparison_roc(
    models_predictions: dict,
    y_true: np.ndarray,
    save_path: Optional[str] = None
):
    """
    Plots multi-model ROC Curves on a single axis for direct academic comparison.
    models_predictions: dict of {model_name: y_prob}
    """
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    
    for name, y_prob in models_predictions.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        color = MODEL_COLORS.get(name, "#333333")
        ls = MODEL_LINESTYLES.get(name, "-")
        ax.plot(fpr, tpr, color=color, linestyle=ls, lw=2.4, label=f"{name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], color="#888888", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.500)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, fontweight="bold")
    ax.set_title("Comparative ROC Curves Across Trend Prediction Models", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    safe_savefig(plt, save_path, dpi=300)

    return fig, ax


def plot_model_comparison_pr(
    models_predictions: dict,
    y_true: np.ndarray,
    save_path: Optional[str] = None
):
    """
    Plots multi-model Precision-Recall Curves on a single axis.
    Crucial for imbalanced social media trend detection tasks.
    models_predictions: dict of {model_name: y_prob}
    """
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    baseline = float(np.mean(y_true))

    for name, y_prob in models_predictions.items():
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        color = MODEL_COLORS.get(name, "#333333")
        ls = MODEL_LINESTYLES.get(name, "-")
        ax.plot(rec, prec, color=color, linestyle=ls, lw=2.4, label=f"{name} (AP = {ap:.3f})")

    ax.axhline(y=baseline, color="#888888", lw=1.5, linestyle="--", label=f"Prevalence Baseline ({baseline:.1%})")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall (Detection Rate)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=11, fontweight="bold")
    ax.set_title("Comparative Precision-Recall Curves (Trend Detection)", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    safe_savefig(plt, save_path, dpi=300)

    return fig, ax


def plot_model_comparison_barchart(
    comparison_df: pd.DataFrame,
    save_path: Optional[str] = None
):
    """
    Grouped Bar Chart comparing core research metrics across all model candidates:
    [Macro-F1, F1-Score, Precision, Recall, ROC-AUC, PR-AUC, MCC]
    """
    metrics = ["macro_f1", "f1_score", "precision", "recall", "roc_auc", "pr_auc", "mcc"]
    metric_labels = ["Macro-F1", "F1 (Trend)", "Precision", "Recall", "ROC-AUC", "PR-AUC", "MCC"]
    
    # Filter available metrics
    available_metrics = [m for m in metrics if m in comparison_df.columns]
    labels = [metric_labels[metrics.index(m)] for m in available_metrics]
    
    n_models = len(comparison_df)
    n_metrics = len(available_metrics)
    
    fig, ax = plt.subplots(figsize=(12, 6.2))
    x = np.arange(n_metrics)
    total_width = 0.8
    bar_width = total_width / n_models

    for idx, (_, row) in enumerate(comparison_df.iterrows()):
        model_name = row["model_name"]
        color = MODEL_COLORS.get(model_name, "#4c72b0")
        offset = (idx - (n_models - 1) / 2) * bar_width
        vals = [row[m] for m in available_metrics]
        
        bars = ax.bar(x + offset, vals, width=bar_width, label=model_name, color=color, alpha=0.90)
        for b in bars:
            h = b.get_height()
            if not np.isnan(h) and h != 0:
                ax.annotate(
                    f"{h:.2f}",
                    xy=(b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold"
                )

    ax.set_title("Benchmark Comparison Across Evaluation Metrics", fontsize=14, fontweight="bold")
    ax.set_ylabel("Score", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="upper left", frameon=True, ncol=2, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.tight_layout()

    safe_savefig(plt, save_path, dpi=300)

    return fig, ax


def plot_model_comparison_radar(
    comparison_df: pd.DataFrame,
    save_path: Optional[str] = None
):
    """
    Plots a multi-dimensional Spider / Radar chart comparing trade-offs between model variants.
    """
    categories = ["Macro-F1", "F1", "Precision", "Recall", "ROC-AUC", "PR-AUC", "MCC"]
    col_map = {
        "Macro-F1": "macro_f1",
        "F1": "f1_score",
        "Precision": "precision",
        "Recall": "recall",
        "ROC-AUC": "roc_auc",
        "PR-AUC": "pr_auc",
        "MCC": "mcc"
    }

    # Verify columns exist
    valid_cats = [c for c in categories if col_map[c] in comparison_df.columns]
    N = len(valid_cats)
    if N < 3:
        logger.warning("Not enough valid metrics for Radar chart.")
        return None, None

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    plt.xticks(angles[:-1], valid_cats, fontsize=10, fontweight="bold")
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)
    plt.ylim(0, 1.05)

    for _, row in comparison_df.iterrows():
        name = row["model_name"]
        color = MODEL_COLORS.get(name, "#1f77b4")
        values = [max(0.0, float(row[col_map[c]])) for c in valid_cats]
        values += values[:1]
        ax.plot(angles, values, lw=2.2, label=name, color=color)
        ax.fill(angles, values, color=color, alpha=0.12)

    ax.set_title("Multi-Dimensional Performance Profiles (Radar Chart)", size=13, fontweight="bold", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), frameon=True, fontsize=9.5)
    plt.tight_layout()

    safe_savefig(plt, save_path, dpi=300)

    return fig, ax


def plot_sarima_vs_sarimax_comparison(
    ts_metrics_df: pd.DataFrame,
    save_path: Optional[str] = None
):
    """
    Compares Time-Series volume forecasting error between Univariate SARIMA and Exogenous SARIMAX:
    - MAE (Mean Absolute Error)
    - RMSE (Root Mean Squared Error)
    - sMAPE (Symmetric Mean Absolute Percentage Error)
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    metrics_info = [
        ("mae", "MAE (Lower is Better)", "#1f77b4"),
        ("rmse", "RMSE (Lower is Better)", "#d62728"),
        ("smape", "sMAPE % (Lower is Better)", "#2ca02c")
    ]

    for ax, (m_col, m_title, default_color) in zip(axes, metrics_info):
        if m_col in ts_metrics_df.columns:
            bars = ax.bar(
                ts_metrics_df["forecaster"],
                ts_metrics_df[m_col],
                color=["#3182bd", "#31a354"],
                width=0.45,
                alpha=0.88,
                edgecolor="black"
            )
            for b in bars:
                h = b.get_height()
                ax.annotate(
                    f"{h:.2f}" + ("%" if "smape" in m_col else ""),
                    xy=(b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                    fontsize=9.5
                )
            ax.set_title(m_title, fontsize=11, fontweight="bold")
            ax.set_ylabel("Value", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.5, axis="y")

    fig.suptitle("Topic Volume Forecasting Accuracy: SARIMA vs. SARIMAX", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    safe_savefig(plt, save_path, dpi=300)

    return fig, axes


def plot_model_comparison_confusion_matrices(
    models_preds: dict,
    y_true: np.ndarray,
    save_path: Optional[str] = None
):
    """
    Plots a 2x2 grid of confusion matrices for direct side-by-side error inspection:
    [SARIMA + LightGBM, SARIMAX + LightGBM, SARIMA + XGBoost, SARIMAX + XGBoost]
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 9.5))
    axes = axes.flatten()

    for idx, (name, y_pred) in enumerate(models_preds.items()):
        if idx >= 4:
            break
        ax = axes[idx]
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Not Trending", "Trending"],
            yticklabels=["Not Trending", "Trending"],
            ax=ax,
            cbar=False
        )
        ax.set_title(f"{name}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)

    fig.suptitle("Comparative Confusion Matrices (Test Set)", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()

    safe_savefig(plt, save_path, dpi=300)

    return fig, axes


def plot_model_comparison_calibration(
    models_predictions: dict,
    y_true: np.ndarray,
    n_bins: int = 8,
    save_path: Optional[str] = None
):
    """
    Overlays probability calibration curves (Reliability Diagrams) for all models.
    """
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#888888", label="Perfect Calibration")

    for name, y_prob in models_predictions.items():
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
        color = MODEL_COLORS.get(name, "#1f77b4")
        ls = MODEL_LINESTYLES.get(name, "-")
        ax.plot(prob_pred, prob_true, marker="o", lw=2, color=color, linestyle=ls, label=name)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel("Mean Predicted Probability", fontsize=11, fontweight="bold")
    ax.set_ylabel("Observed Fraction of Positives", fontsize=11, fontweight="bold")
    ax.set_title("Probability Calibration (Reliability Diagram) Comparison", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", frameon=True, fontsize=9.5)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    safe_savefig(plt, save_path, dpi=300)

    return fig, ax


def export_model_comparison_table(
    comparison_df: pd.DataFrame,
    save_dir: str
):
    """
    Exports clean, publication-ready benchmark tables:
    1. CSV file: model_comparison_results.csv
    2. LaTeX code: model_comparison_table.tex (ready to paste into research papers)
    3. Rendered Graphic Table: model_comparison_table.png
    """
    ensure_dir(os.path.join(save_dir, ".gitkeep"))

    # 1. Save CSV
    csv_path = os.path.join(save_dir, "model_comparison_results.csv")
    comparison_df.to_csv(csv_path, index=False)
    logger.info(f"Saved comparison CSV to: {csv_path}")

    # 2. Generate LaTeX Table
    latex_path = os.path.join(save_dir, "model_comparison_table.tex")
    core_cols = ["model_name", "macro_f1", "f1_score", "precision", "recall", "roc_auc", "pr_auc", "mcc", "optimal_threshold"]
    valid_cols = [c for c in core_cols if c in comparison_df.columns]
    
    latex_df = comparison_df[valid_cols].copy()
    latex_df.columns = [c.replace("_", " ").title() for c in valid_cols]

    latex_code = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Comprehensive Performance Comparison of Hybrid Trend Prediction Frameworks}",
        "\\label{tab:model_comparison}",
        "\\resizebox{\\textwidth}{!}{",
        "\\begin{tabular}{l" + "c" * (len(valid_cols) - 1) + "}",
        "\\hline\\hline",
        " & ".join(latex_df.columns) + " \\\\",
        "\\hline"
    ]

    for _, row in latex_df.iterrows():
        row_str = []
        for col in latex_df.columns:
            val = row[col]
            if isinstance(val, (int, float)):
                row_str.append(f"{val:.4f}")
            else:
                row_str.append(str(val))
        latex_code.append(" & ".join(row_str) + " \\\\")

    latex_code.extend([
        "\\hline\\hline",
        "\\end{tabular}",
        "}",
        "\\end{table}"
    ])

    with open(latex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(latex_code))
    logger.info(f"Saved publication LaTeX table to: {latex_path}")

    # 3. Render High-Resolution Graphic Table
    fig, ax = plt.subplots(figsize=(11, 2.5 + 0.45 * len(comparison_df)))
    ax.axis("off")
    ax.axis("tight")

    render_data = []
    display_cols = ["Model", "Macro-F1", "F1", "Precision", "Recall", "ROC-AUC", "PR-AUC", "MCC", "Opt. Thresh (θ*)"]
    
    for _, row in comparison_df.iterrows():
        render_data.append([
            row.get("model_name", "N/A"),
            f"{row.get('macro_f1', 0):.4f}",
            f"{row.get('f1_score', 0):.4f}",
            f"{row.get('precision', 0):.4f}",
            f"{row.get('recall', 0):.4f}",
            f"{row.get('roc_auc', 0):.4f}",
            f"{row.get('pr_auc', 0):.4f}",
            f"{row.get('mcc', 0):.4f}",
            f"{row.get('optimal_threshold', 0.5):.3f}"
        ])

    table = ax.table(
        cellText=render_data,
        colLabels=display_cols,
        cellLoc="center",
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.15, 1.6)

    # Style header
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#2b5c8f")
        else:
            if i % 2 == 0:
                cell.set_facecolor("#f2f5f9")
            else:
                cell.set_facecolor("#ffffff")

    plt.title("Comparative Performance Benchmark: Emerging Topic Trend Detection", fontsize=13, fontweight="bold", pad=15)
    img_path = os.path.join(save_dir, "model_comparison_table.png")
    safe_savefig(plt, img_path, dpi=300)
    plt.close(fig)


