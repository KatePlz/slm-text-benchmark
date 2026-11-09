#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль train_benchmark.py
Универсальный запуск LoRA для ModernBERT, RoBERTa, SmolLM2 и Qwen2.5.
Поддерживает датасеты: sst2, ag_news, dair-ai/emotion.
"""

import os
import json
import time
import argparse
import inspect

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed
)
from peft import LoraConfig, get_peft_model, TaskType
from evaluate_metrics import compute_classification_metrics

def parse_args():
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning Benchmark")
    parser.add_argument("--model_name", type=str, default="answerdotai/ModernBERT-base")
    parser.add_argument("--dataset_name", type=str, default="sst2", choices=["sst2", "ag_news", "emotion"])
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_samples", type=int, default=10000)
    return parser.parse_args()

def get_target_and_save_modules(model_name):
    mid = model_name.lower()
    if "modernbert" in mid:
        return ["Wqkv", "Wo"], None
    elif "roberta" in mid:
        return ["query", "value"], None
    else:  # SmolLM2 и Qwen2.5 (декодеры)
        return ["q_proj", "v_proj"], ["score"]

def prepare_data(dataset_name, tokenizer, max_length, max_train_samples=10000):
    if dataset_name == "sst2":
        raw_dataset = load_dataset("nyu-mll/glue", "sst2")
        text_column = "sentence"
        num_labels = 2
        id2label = {0: "negative", 1: "positive"}
        label2id = {"negative": 0, "positive": 1}
        eval_split = "validation"
    elif dataset_name == "ag_news":
        raw_dataset = load_dataset("fancyzhx/ag_news")
        text_column = "text"
        num_labels = 4
        id2label = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
        label2id = {"World": 0, "Sports": 1, "Business": 2, "Sci/Tech": 3}
        eval_split = "test"
    elif dataset_name == "emotion":
        raw_dataset = load_dataset("dair-ai/emotion")
        text_column = "text"
        num_labels = 6
        id2label = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}
        label2id = {v: k for k, v in id2label.items()}
        eval_split = "test"
    else:
        raise ValueError(f"Неподдерживаемый датасет: {dataset_name}")

    if max_train_samples is not None and max_train_samples < len(raw_dataset["train"]):
        raw_dataset["train"] = raw_dataset["train"].select(range(max_train_samples))

    def tokenize_fn(examples):
        return tokenizer(examples[text_column], truncation=True, max_length=max_length)

    tokenized_dataset = raw_dataset.map(tokenize_fn, batched=True)
    return tokenized_dataset, num_labels, id2label, label2id, eval_split

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[*] Старт: {args.model_name} на {args.dataset_name} | Устройство: {device}")

    # Загрузка токенизатора
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    mid = args.model_name.lower()
    if "smollm" in mid or "qwen" in mid:
        tokenizer.padding_side = "left"

    # Подготовка данных
    tokenized_datasets, num_labels, id2label, label2id, eval_split = prepare_data(
        args.dataset_name, tokenizer, args.max_length, args.max_train_samples
    )

    # Загрузка модели в float32
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    model = model.float()

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Наложение LoRA
    targets, save_modules = get_target_and_save_modules(args.model_name)
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=targets,
        modules_to_save=save_modules,
        bias="none"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Очистка видеопамяти
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    run_dir = os.path.join(args.output_dir, f"{args.model_name.replace('/', '_')}_{args.dataset_name}")

    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    eval_key = "eval_strategy" if "eval_strategy" in ta_params else "evaluation_strategy"
    current_lr = 1e-4 if ("smollm" in mid or "qwen" in mid) else args.learning_rate

    raw_training_args = {
        "output_dir": run_dir,
        eval_key: "epoch",
        "save_strategy": "no",
        "learning_rate": current_lr,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size * 2,
        "num_train_epochs": args.epochs,
        "weight_decay": 0.01,
        "logging_steps": 50,
        "fp16": torch.cuda.is_available(),
        "report_to": "none"
    }

    if "max_grad_norm" in ta_params:
        raw_training_args["max_grad_norm"] = 1.0

    valid_training_args = {k: v for k, v in raw_training_args.items() if k in ta_params}
    training_args = TrainingArguments(**valid_training_args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets[eval_split],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_classification_metrics,
        processing_class=tokenizer
    )

    # Обучение
    start_time = time.time()
    trainer.train()
    total_train_time = time.time() - start_time
    sec_per_epoch = total_train_time / args.epochs

    # Память
    max_memory_gb = 0.0
    if torch.cuda.is_available():
        max_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    # Оценка
    eval_metrics = trainer.evaluate()
    eval_metrics["sec_per_epoch"] = sec_per_epoch
    eval_metrics["max_vram_gb"] = max_memory_gb

    print("=" * 60)
    print(f"ИТОГИ: {args.model_name} / {args.dataset_name}")
    for k, v in eval_metrics.items():
        if isinstance(v, float):
            print(f"  {k:25s}: {v:.4f}")
        else:
            print(f"  {k:25s}: {v}")
    print("=" * 60)

    # Сохранение
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "benchmark_metrics.txt"), "w", encoding="utf-8") as f:
        for k, v in eval_metrics.items():
            f.write(f"{k}: {v}\n")
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v))
                   for k, v in eval_metrics.items()}, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
