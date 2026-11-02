#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль train_benchmark.py
Параметризованный запуск адаптивного дообучения (LoRA) компактных языковых моделей
для классификации текстов с профилированием времени обучения и потребления памяти VRAM.

Гиперпараметры по умолчанию:
  ранг LoRA r=8, alpha=16, dropout=0.05, LR=2e-4, batch=32 (16 для Qwen2.5-0.5B),
  max_length=128, epochs=3, seed=42, FP16 (при наличии CUDA).
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

LORA_TARGET_MODULES = {
    # ModernBERT: слитая проекция QKV + выходная проекция внимания
    "modernbert": ["attn.Wqkv", "attn.Wo"],
    # RoBERTa/BERT: q/k/v-проекции self-attention + выходная проекция внимания
    "roberta": ["query", "key", "value", "attention.output.dense"],
    "bert": ["query", "key", "value", "attention.output.dense"],
    # Каузальные декодеры семейств Llama (SmolLM2) и Qwen2
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "qwen2": ["q_proj", "k_proj", "v_proj", "o_proj"],
}

# Резервный список на случай незнакомой архитектуры
LORA_TARGET_FALLBACK = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "query", "key", "value", "attention.output.dense",
    "Wqkv", "attn.Wqkv", "attn.Wo",
]


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning Benchmark for Compact LMs")
    parser.add_argument("--model_name", type=str, default="answerdotai/ModernBERT-base",
                        help="Hugging Face model ID or path (e.g. answerdotai/ModernBERT-base, "
                             "Qwen/Qwen2.5-0.5B, HuggingFaceTB/SmolLM2-360M, roberta-base)")
    parser.add_argument("--dataset_name", type=str, default="sst2", choices=["sst2", "ag_news"],
                        help="Dataset name: 'sst2' or 'ag_news'")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save checkpoints and metrics")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Per-device train batch size (в статье: 32, для Qwen2.5-0.5B — 16)")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate for AdamW")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA rank dimension")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha scaling factor")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout rate")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Maximum token sequence length")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Ограничение размера train-выборки (для отладки; по умолчанию — весь корпус)")
    parser.add_argument("--max_eval_samples", type=int, default=None,
                        help="Ограничение размера eval-выборки (для отладки; по умолчанию — весь корпус)")
    return parser.parse_args()


def prepare_data(dataset_name, tokenizer, max_length,
                 max_train_samples=None, max_eval_samples=None):
    if dataset_name == "sst2":
        raw_dataset = load_dataset("nyu-mll/glue", "sst2")
        text_column = "sentence"
        label_column = "label"
        num_labels = 2
        id2label = {0: "negative", 1: "positive"}
        label2id = {"negative": 0, "positive": 1}
    elif dataset_name == "ag_news":
        raw_dataset = load_dataset("fancyzhx/ag_news")
        text_column = "text"
        label_column = "label"
        num_labels = 4
        id2label = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
        label2id = {"World": 0, "Sports": 1, "Business": 2, "Sci/Tech": 3}
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if max_train_samples is not None and "train" in raw_dataset:
        n = min(max_train_samples, len(raw_dataset["train"]))
        raw_dataset["train"] = raw_dataset["train"].select(range(n))
    if max_eval_samples is not None:
        for split in ("validation", "test"):
            if split in raw_dataset:
                n = min(max_eval_samples, len(raw_dataset[split]))
                raw_dataset[split] = raw_dataset[split].select(range(n))

    def tokenize_fn(examples):
        return tokenizer(examples[text_column], truncation=True, max_length=max_length)

    original_columns = raw_dataset["train"].column_names
    columns_to_remove = [c for c in original_columns if c != label_column]
    tokenized_dataset = raw_dataset.map(
        tokenize_fn, batched=True, remove_columns=columns_to_remove
    )
    return tokenized_dataset, num_labels, id2label, label2id


def resolve_lora_targets(model):
    """Подбор целевых модулей LoRA по архитектуре модели с проверкой наличия."""
    model_type = getattr(model.config, "model_type", None)
    if model_type in LORA_TARGET_MODULES:
        targets = LORA_TARGET_MODULES[model_type]
    else:
        module_names = {name for name, _ in model.named_modules()}
        targets = [
            t for t in LORA_TARGET_FALLBACK
            if any(name == t or name.endswith("." + t) for name in module_names)
        ]
        if not targets:
            raise RuntimeError(
                f"Не удалось подобрать целевые модули LoRA для model_type='{model_type}'. "
                f"Укажите LORA_TARGET_MODULES вручную."
            )

    module_names = {name for name, _ in model.named_modules()}
    matched = [
        t for t in targets
        if any(name == t or name.endswith("." + t) for name in module_names)
    ]
    if not matched:
        raise RuntimeError(
            f"Целевые модули LoRA {targets} не найдены в модели "
            f"(model_type='{model_type}')."
        )
    return matched


def build_training_args(args, run_dir):
    """Создание TrainingArguments с учётом различий API transformers v4/v5."""
    ta_params = inspect.signature(TrainingArguments.__init__).parameters

    # В transformers>=4.46 evaluation_strategy переименован в eval_strategy,
    # в v5 старое имя удалено полностью.
    eval_key = "eval_strategy" if "eval_strategy" in ta_params else "evaluation_strategy"

    kwargs = dict(
        output_dir=run_dir,
        **{eval_key: "epoch"},
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )
    if "overwrite_output_dir" in ta_params:
        kwargs["overwrite_output_dir"] = True
    if "logging_dir" in ta_params:
        kwargs["logging_dir"] = os.path.join(run_dir, "logs")
    return TrainingArguments(**kwargs)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Инициализация эксперимента: модель={args.model_name}, "
          f"датасет={args.dataset_name}, устройство={device}")

    # Загрузка токенизатора
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Подготовка данных (с ограничением выборок — для отладки)
    tokenized_datasets, num_labels, id2label, label2id = prepare_data(
        args.dataset_name, tokenizer, args.max_length,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )
    eval_split = "validation" if "validation" in tokenized_datasets else "test"

    # Загрузка базовой модели
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Конфигурация LoRA: целевые модули определяются по архитектуре модели
    target_modules = resolve_lora_targets(model)
    print(f"[*] Целевые модули LoRA ({getattr(model.config, 'model_type', '?')}): {target_modules}")

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none"
    )

    model = get_peft_model(model, peft_config)
    print("[*] Модель успешно обернута адаптерами LoRA.")
    model.print_trainable_parameters()

    # Сброс счетчика памяти VRAM
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    run_dir = os.path.join(args.output_dir, f"{args.model_name.replace('/', '_')}_{args.dataset_name}")
    training_args = build_training_args(args, run_dir)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer_params = inspect.signature(Trainer.__init__).parameters
    processor_key = "processing_class" if "processing_class" in trainer_params else "tokenizer"

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets[eval_split],
        data_collator=data_collator,
        compute_metrics=compute_classification_metrics,
        **{processor_key: tokenizer},
    )

    # Профилирование времени
    start_time = time.time()
    train_result = trainer.train()
    total_train_time = time.time() - start_time
    sec_per_epoch = total_train_time / args.epochs

    # Пиковое потребление памяти GPU
    max_memory_gb = 0.0
    if torch.cuda.is_available():
        max_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    # Финальная оценка
    eval_metrics = trainer.evaluate()
    eval_metrics["sec_per_epoch"] = sec_per_epoch
    eval_metrics["max_vram_gb"] = max_memory_gb

    os.makedirs(run_dir, exist_ok=True)

    print("=" * 60)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ ОЦЕНКИ:")
    for k, v in eval_metrics.items():
        if isinstance(v, float):
            print(f"  {k:25s}: {v:.4f}")
        else:
            print(f"  {k:25s}: {v}")
    print("=" * 60)

    # Сохранение результатов
    with open(os.path.join(run_dir, "benchmark_metrics.txt"), "w", encoding="utf-8") as f:
        for k, v in eval_metrics.items():
            f.write(f"{k}: {v}\n")
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v))
                   for k, v in eval_metrics.items()}, f, ensure_ascii=False, indent=2)
    print(f"[*] Результаты сохранены в {run_dir}")


if __name__ == "__main__":
    main()
