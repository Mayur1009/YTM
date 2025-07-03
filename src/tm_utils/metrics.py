import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    hamming_loss,
    multilabel_confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def plot_heatmap(arr, **hmapargs):
    figsize = (max(4, 1.2 * arr.shape[0]), max(4, 1.2 * arr.shape[1]))
    fig, ax = plt.subplots(1, 1, figsize=figsize, layout="compressed", dpi=120)
    sns.heatmap(
        arr,
        annot=True,
        annot_kws={"fontsize": "small"},
        fmt=".4g",
        linewidths=0.1,
        cbar=True,
        square=True,
        ax=ax,
        **hmapargs,
    )
    ax.tick_params(axis="both", which="major", labelsize="small")
    ax.tick_params(axis="x", which="major", rotation=90)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    return fig


def multiclass_metrics(true, pred, prob, class_names):
    assert true.ndim == 1, "True labels should be a 1D array."
    assert pred.ndim == 1, "Predicted labels should be a 1D array."
    assert prob.ndim == 2, "Probabilities should be a 2D array."
    N = true.shape[0]

    acc = accuracy_score(true, pred)
    hml = hamming_loss(true, pred)
    cm = confusion_matrix(true, pred)
    cm_plot = plot_heatmap(cm, xticklabels=class_names, yticklabels=class_names)

    macro_pre, macro_rec, macro_f1, _ = precision_recall_fscore_support(true, pred, average="macro", zero_division=0)
    per_class_pre, per_class_rec, per_class_f1, _ = precision_recall_fscore_support(
        true,
        pred,
        average=None,
        zero_division=0,
    )

    true_bin = np.zeros((N, len(class_names)))
    true_bin[np.arange(N), true] = 1
    macro_roc = roc_auc_score(true_bin, prob, average="macro", multi_class="ovr")
    per_class_roc = roc_auc_score(true_bin, prob, average=None, multi_class="ovr")

    metrics = {
        "accuracy": acc,
        "precision/macro": macro_pre,
        "recall/macro": macro_rec,
        "f1/macro": macro_f1,
        "roc_auc/macro": macro_roc,
        **{f"precision/class.{name}": pre for name, pre in zip(class_names, per_class_pre)},  # pyright:ignore[reportArgumentType]
        **{f"recall/class.{name}": rec for name, rec in zip(class_names, per_class_rec)},  # pyright:ignore[reportArgumentType]
        **{f"f1/class.{name}": f1 for name, f1 in zip(class_names, per_class_f1)},  # pyright:ignore[reportArgumentType]
        **{f"roc_auc/class.{name}": roc for name, roc in zip(class_names, per_class_roc)},  # pyright:ignore[reportArgumentType]
        "confusion_matrix": cm_plot,
        "hamming_loss": hml,
    }

    return metrics


def multilabel_metrics(true, pred, prob, class_names):
    assert true.ndim == 2, "True labels should be a 2D array."
    assert pred.ndim == 2, "Predicted labels should be a 2D array."
    assert prob.ndim == 2, "Probabilities should be a 2D array."

    N = true.shape[0]

    acc = 0
    for c in range(len(class_names)):
        acc += accuracy_score(true[:, c], pred[:, c])
        acc /= len(class_names)

    cm = multilabel_confusion_matrix(true, pred)
    cm_plots = [plot_heatmap(cm[i], xticklabels=["0", "1"], yticklabels=["0", "1"]) for i in range(len(class_names))]

    hml = hamming_loss(true, pred)
    macro_pre, macro_rec, macro_f1, _ = precision_recall_fscore_support(true, pred, average="macro", zero_division=0)
    per_class_pre, per_class_rec, per_class_f1, _ = precision_recall_fscore_support(
        true,
        pred,
        average=None,
        zero_division=0,
    )

    macro_roc = roc_auc_score(true, prob, average="macro", multi_class="ovr")
    per_class_roc = roc_auc_score(true, prob, average=None, multi_class="ovr")

    metrics = {
        "accuracy": acc,
        "precision/macro": macro_pre,
        "recall/macro": macro_rec,
        "f1/macro": macro_f1,
        "roc_auc/macro": macro_roc,
        **{f"precision/class.{name}": pre for name, pre in zip(class_names, per_class_pre)},  # pyright:ignore[reportArgumentType]
        **{f"recall/class.{name}": rec for name, rec in zip(class_names, per_class_rec)},  # pyright:ignore[reportArgumentType]
        **{f"f1/class.{name}": f1 for name, f1 in zip(class_names, per_class_f1)},  # pyright:ignore[reportArgumentType]
        **{f"roc_auc/class.{name}": roc for name, roc in zip(class_names, per_class_roc)},  # pyright:ignore[reportArgumentType]
        **{f"confusion_matrix/class.{name}": cm_plot for name, cm_plot in zip(class_names, cm_plots)},  # pyright:ignore[reportArgumentType]
        "hamming_loss": hml,
    }

    return metrics
