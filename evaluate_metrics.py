#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль evaluate_metrics.py
Расчет комплексных метрик для Hugging Face Trainer:
Accuracy, Macro-F1, Cross-Entropy Loss и Brier Score.
"""

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, log_loss, brier_score_loss


def compute_classification_metrics(eval_pred):
    """
    Callback-функция вычисления метрик для Hugging Face Trainer.
    """
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
        # Для бинарной классификации: Brier по вероятности положительного класса
        brier = brier_score_loss(labels, probabilities[:, 1])
    else:
        # Для многоклассовой классификации: мультиклассовый Brier Score
        # BS = (1/N) * sum_i sum_c (p_ic - y_ic)^2 
        one_hot_labels = np.eye(num_classes)[labels]
        brier = np.mean(np.sum((probabilities - one_hot_labels) ** 2, axis=1))

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "cross_entropy": ce_loss,
        "brier_score": brier
    }


if __name__ == "__main__":
    print("[*] Модуль вычисления метрик готов к использованию.")
