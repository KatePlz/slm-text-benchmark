#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль evaluate_metrics.py
Расчет комплексных метрик:
Accuracy, Macro-F1, Cross-Entropy Loss и Brier Score.
"""

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, log_loss, brier_score_loss

def compute_classification_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]

    logits = np.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)

    probabilities = torch.softmax(torch.tensor(logits), dim=-1).numpy()

    num_classes = probabilities.shape[1]
    probabilities = np.nan_to_num(probabilities, nan=1.0 / num_classes)

    prob_sum = np.sum(probabilities, axis=1, keepdims=True)
    prob_sum[prob_sum == 0] = 1.0
    probabilities = probabilities / prob_sum

    predictions = np.argmax(probabilities, axis=-1)

    acc = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average="macro")

    clipped = np.clip(probabilities, 1e-15, 1.0 - 1e-15)
    ce_loss = log_loss(labels, clipped, labels=list(range(num_classes)))

    if num_classes == 2:
        brier = brier_score_loss(labels, probabilities[:, 1])
    else:
        one_hot_labels = np.eye(num_classes)[labels]
        brier = float(np.mean(np.sum((probabilities - one_hot_labels) ** 2, axis=1)))

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "cross_entropy": float(ce_loss),
        "brier_score": float(brier)
    }

if __name__ == "__main__":
    print("[*] Модуль вычисления метрик готов к использованию.")
