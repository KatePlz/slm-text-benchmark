#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль evaluate_metrics.py
Расчет комплексных метрик: Accuracy, Macro-F1, Cross-Entropy Loss, Brier Score,
а также построение диаграмм надежности (Reliability Diagrams) и калибровочных кривых
и графика сходимости функции потерь.
"""

import os
import json
import argparse

import numpy as np
import torch

from sklearn.metrics import accuracy_score, f1_score, log_loss, brier_score_loss
from sklearn.calibration import calibration_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODEL_ORDER = ["modernbert", "roberta", "qwen", "smollm2"]

MODEL_STYLES = {
    "modernbert": ("ModernBERT-base", "tab:blue", "s", "-"),
    "roberta":    ("RoBERTa-base",    "tab:green", "^", "--"),
    "qwen":       ("Qwen2.5-0.5B",    "tab:orange", "o", "-."),
    "smollm2":    ("SmolLM2-360M",    "tab:red", "d", ":"),
}

_FALLBACK_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
_FALLBACK_MARKERS = ["s", "^", "o", "d", "v", "P"]
_FALLBACK_LSTYLES = ["-", "--", "-.", ":"]


def _model_key(model_name: str) -> str:
    lowered = model_name.lower()
    for key in MODEL_ORDER:
        if key in lowered:
            return key
    return lowered


def display_model_name(model_name: str) -> str:
    key = _model_key(model_name)
    if key in MODEL_STYLES:
        return MODEL_STYLES[key][0]
    name = model_name.split("/")[-1]
    if name.lower() == "roberta-base":
        return "RoBERTa-base"
    return name


def _ordered_results(results_dict):
    def rank(item):
        key = _model_key(item[0])
        return (MODEL_ORDER.index(key) if key in MODEL_ORDER else len(MODEL_ORDER),)
    return sorted(results_dict.items(), key=rank)


def compute_classification_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]

    probabilities = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    predictions = np.argmax(probabilities, axis=-1)

    acc = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average="macro")

    num_classes = probabilities.shape[1]
    clipped = np.clip(probabilities, 1e-15, 1 - 1e-15)
    ce_loss = log_loss(labels, clipped, labels=list(range(num_classes)))

    if num_classes == 2:
        brier = brier_score_loss(labels, probabilities[:, 1])
    else:
        one_hot_labels = np.eye(num_classes)[labels]
        brier = np.mean(np.sum((probabilities - one_hot_labels) ** 2, axis=1))

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "cross_entropy": ce_loss,
        "brier_score": brier
    }


def plot_calibration_curves(results_dict, dataset_label="SST-2",
                            output_path="calibration_comparison.png"):
    ordered = _ordered_results(results_dict)

    plt.figure(figsize=(7, 6), dpi=300)
    plt.plot([0, 1], [0, 1], "k--", label="Идеальная калибровка")

    for idx, (model_name, (y_true, y_prob)) in enumerate(ordered):
        key = _model_key(model_name)
        prob_true, prob_pred = calibration_curve(
            np.asarray(y_true), np.asarray(y_prob), n_bins=10, strategy="uniform"
        )
        brier = brier_score_loss(np.asarray(y_true), np.asarray(y_prob))
        if key in MODEL_STYLES:
            _, color, marker, lstyle = MODEL_STYLES[key]
        else:
            color = _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)]
            marker = _FALLBACK_MARKERS[idx % len(_FALLBACK_MARKERS)]
            lstyle = _FALLBACK_LSTYLES[idx % len(_FALLBACK_LSTYLES)]
        label = f"{display_model_name(model_name)} (Brier: {brier:.3f})"
        plt.plot(prob_pred, prob_true, marker + lstyle, color=color,
                 markerfacecolor=color, markeredgecolor="white", label=label)

    plt.xlabel("Средняя предсказанная вероятность", fontsize=11)
    plt.ylabel("Доля истинных положительных исходов", fontsize=11)
    plt.title(f"Калибровочные кривые ({dataset_label})", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper left", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[*] График калибровочных кривых сохранен в {output_path}")


def plot_loss_curves(results_dict, output_path="loss_convergence.png", max_steps=None):

    ordered = _ordered_results(results_dict)

    plt.figure(figsize=(7, 6), dpi=300)
    for idx, (model_name, history) in enumerate(ordered):
        key = _model_key(model_name)
        steps = [entry["step"] for entry in history if "loss" in entry]
        losses = [entry["loss"] for entry in history if "loss" in entry]
        if max_steps is not None:
            filtered = [(s, l) for s, l in zip(steps, losses) if s <= max_steps]
            steps, losses = (list(zip(*filtered)) if filtered else ([], []))
        if key in MODEL_STYLES:
            _, color, _, lstyle = MODEL_STYLES[key]
        else:
            color = _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)]
            lstyle = _FALLBACK_LSTYLES[idx % len(_FALLBACK_LSTYLES)]
        marker = "o" if len(steps) < 3 else None
        plt.plot(steps, losses, linestyle=lstyle, color=color, linewidth=1.8,
                 marker=marker, markersize=4, label=display_model_name(model_name))

    plt.xlabel("Шаг оптимизации (Steps)", fontsize=11)
    plt.ylabel("Кросс-энтропийная функция потерь", fontsize=11)
    plt.title("Динамика сходимости потерь при LoRA", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[*] График сходимости потерь сохранен в {output_path}")


def load_run_artifacts(run_dir):
    probs_path = os.path.join(run_dir, "eval_predictions.npz")
    history_path = os.path.join(run_dir, "history.json")
    labels, probabilities, history = None, None, None
    if os.path.isfile(probs_path):
        data = np.load(probs_path)
        labels, probabilities = data["labels"], data["probabilities"]
    if os.path.isfile(history_path):
        with open(history_path, encoding="utf-8") as f:
            history = json.load(f)
    return labels, probabilities, history


def build_figures(results_dir, dataset_name="sst2", output_dir=None, max_steps=None):
    output_dir = output_dir or results_dir
    os.makedirs(output_dir, exist_ok=True)
    dataset_label = "SST-2" if dataset_name == "sst2" else "AG News"

    runs = {}
    if os.path.isdir(results_dir):
        for entry in sorted(os.listdir(results_dir)):
            run_dir = os.path.join(results_dir, entry)
            if os.path.isdir(run_dir) and entry.endswith(f"_{dataset_name}"):
                labels, probabilities, history = load_run_artifacts(run_dir)
                model_name = entry[: -(len(dataset_name) + 1)]
                runs[model_name] = (labels, probabilities, history)

    if not runs:
        print(f"[!] В каталоге {results_dir} не найдено прогонов для датасета '{dataset_name}'.")
        return

    if dataset_name == "sst2":
        calibration_input = {}
        for model_name, (labels, probabilities, _) in runs.items():
            if labels is None or probabilities is None:
                continue
            if probabilities.shape[1] == 2:
                calibration_input[model_name] = (labels, probabilities[:, 1])
        if calibration_input:
            plot_calibration_curves(
                calibration_input,
                dataset_label=dataset_label,
                output_path=os.path.join(output_dir, f"calibration_{dataset_name}.png"),
            )
        else:
            print("[!] Не найдены сохраненные вероятности для калибровочных кривых.")

    loss_input = {
        model_name: history
        for model_name, (_, _, history) in runs.items()
        if history
    }
    if loss_input:
        plot_loss_curves(
            loss_input,
            output_path=os.path.join(output_dir, f"loss_convergence_{dataset_name}.png"),
            max_steps=max_steps,
        )
    else:
        print("[!] Не найдена история потерь (history.json) для графика сходимости.")


def parse_args():
    parser = argparse.ArgumentParser(description="Построение графиков статьи по результатам прогонов")
    parser.add_argument("--results_dir", type=str, default="./results",
                        help="Каталог с результатами train_benchmark.py")
    parser.add_argument("--dataset_name", type=str, default="sst2", choices=["sst2", "ag_news"],
                        help="Датасет, для которого строятся графики")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Каталог для сохранения графиков (по умолчанию — results_dir)")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Ограничение оси X на графике сходимости (например, 1000, как в статье)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_figures(args.results_dir, args.dataset_name, args.output_dir, args.max_steps)
