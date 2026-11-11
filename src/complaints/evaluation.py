"""Metrics, plots and error analysis; test access is orchestrated after selection."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

from .data import write_json


def metrics_from_predictions(truth, prediction, names):
    labels = np.arange(len(names))
    result = {"accuracy": float(accuracy_score(truth, prediction)), "records": len(truth)}
    for average in ["macro", "weighted"]:
        precision, recall, f1, _ = precision_recall_fscore_support(truth, prediction, labels=labels, average=average, zero_division=0)
        result.update({f"{average}_precision": float(precision), f"{average}_recall": float(recall), f"{average}_f1": float(f1)})
    result["per_class"] = classification_report(truth, prediction, labels=labels, target_names=names, output_dict=True, zero_division=0)
    return result


def save_evaluation(output_dir, probabilities, indices, data, loss, name, split):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    truth = data.targets[indices]
    prediction = probabilities.argmax(axis=1)
    metrics = metrics_from_predictions(truth, prediction, data.label_names)
    metrics["loss"] = float(loss)
    metrics["split"] = split
    metrics["model"] = name
    write_json(output_dir / "metrics.json", metrics)
    pd.DataFrame(metrics["per_class"]).T.to_csv(output_dir / "classification_report.csv")
    cm = confusion_matrix(truth, prediction, labels=np.arange(len(data.label_names)))
    normalized = np.divide(cm, cm.sum(axis=1, keepdims=True), out=np.zeros_like(cm, dtype=float), where=cm.sum(axis=1, keepdims=True) != 0)
    for label, matrix in [("counts", cm), ("normalized", normalized)]:
        pd.DataFrame(matrix, index=data.label_names, columns=data.label_names).to_csv(output_dir / f"confusion_matrix_{label}.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.7))
    for ax, matrix, title, formatter in [(axes[0], cm, "Counts", "d"), (axes[1], normalized, "Row-normalized", ".2f")]:
        im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1 if formatter == ".2f" else None)
        ax.set_xticks(range(len(data.label_names)), data.label_names, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(len(data.label_names)), data.label_names, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{name} — {split} — {title}")
        for row in range(len(data.label_names)):
            for col in range(len(data.label_names)):
                ax.text(col, row, format(matrix[row, col], formatter), ha="center", va="center", fontsize=8,
                        color="white" if matrix[row, col] > matrix.max() / 2 else "black")
        fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrices.png", dpi=150)
    plt.close(fig)
    records = data.records.iloc[indices].copy()
    records["predicted_id"] = prediction
    records["predicted_label"] = [data.label_names[i] for i in prediction]
    records["confidence"] = probabilities.max(axis=1)
    records["correct"] = truth == prediction
    records["recurrent_truncated"] = data.encoded["recurrent_full_lengths"][indices] > data.preprocessing["recurrent_max_length"]
    records["transformer_truncated"] = data.encoded["transformer_full_lengths"][indices] > data.preprocessing["transformer_max_length"]
    records.drop(columns=["narrative"]).to_csv(output_dir / "predictions.csv.gz", index=False)
    np.savez_compressed(output_dir / "probabilities.npz", probabilities=probabilities, indices=indices)
    # This is diagnostic output, not another source for selecting the deployment winner.
    examples = pd.concat([records.loc[~records.correct].sort_values("confidence", ascending=False).head(15),
                          records.loc[records.correct].sort_values("confidence").head(5)])
    examples.to_csv(output_dir / "error_and_correct_examples.csv", index=False)
    strata = {}
    for col in ["recurrent_truncated", "transformer_truncated"]:
        strata[col] = {}
        for value in [False, True]:
            group = records.loc[records[col].eq(value)]
            strata[col][str(value)] = {"records": len(group), "accuracy": float(group.correct.mean()) if len(group) else None}
    write_json(output_dir / "truncation_analysis.json", strata)
    return metrics


def save_learning_curves(history, output):
    if not history:
        return
    frame = pd.DataFrame(history)
    frame.to_csv(Path(output) / "history.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(frame.epoch, frame.training_loss, label="Training")
    axes[0].plot(frame.epoch, frame.validation_loss, label="Validation")
    axes[0].set_ylabel("Class-weighted loss")
    axes[0].legend()
    axes[1].plot(frame.epoch, frame.validation_macro_f1, label="Macro F1")
    axes[1].plot(frame.epoch, frame.validation_accuracy, label="Accuracy")
    axes[1].set_ylabel("Validation score")
    axes[1].legend()
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_xticks(frame.epoch)
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(Path(output) / "learning_curves.png", dpi=150)
    plt.close(fig)
