# Сравнительный анализ компактных языковых моделей при классификации текстовых данных (LoRA Benchmark)

Данный репозиторий содержит воспроизводимый программный комплекс для экспериментального исследования эффективности, надежности калибровки вероятностей и вычислительной трудоемкости современных компактных языковых моделей (энкодеров и декодеров) при параметрически-эффективном дообучении методом LoRA (Low-Rank Adaptation).

## Структура репозитория

```
.
├── train_benchmark.py       # Основной параметризованный скрипт дообучения и профилирования
├── evaluate_metrics.py      # Модуль расчета метрик (Accuracy, F1, Cross-Entropy, Brier Score, диаграммы калибровки)
├── requirements.txt         # Зависимости Python
├── README.md                # Описание и руководство по запуску
└── results/                 # Директория для сохранения весов LoRA и метрик
```

## Исследуемые архитектуры моделей
1. **ModernBERT-base** (`answerdotai/ModernBERT-base`, ~149M параметров) — современный энкодер (2024/2025) с поддержкой контекста до 8192 токенов, FlashAttention-2 и вращающимися позиционными эмбеддингами (RoPE).
2. **RoBERTa-base** (`roberta-base`, ~125M параметров) — классический бейзлайн двунаправленного трансформера.
3. **SmolLM2-360M** (`HuggingFaceTB/SmolLM2-360M`, ~360M параметров) — сверхлегкая каузальная декодерная модель (2024/2025).
4. **Qwen2.5-0.5B** (`Qwen/Qwen2.5-0.5B`, ~490M параметров) — компактная авторегрессионная архитектура от Alibaba Cloud.

## Наборы данных
* **SST-2** (Stanford Sentiment Treebank) — бинарная классификация тональности предложений.
* **AG News** — многоклассовая тематическая классификация новостей (4 категории).

## Инструкция по локальному запуску

1. Клонируйте репозиторий и перейдите в рабочую директорию:
```bash
git clone https://github.com/username/slm-text-benchmark.git
cd slm-text-benchmark
```

2. Создайте и активируйте виртуальное окружение Python (рекомендуется Python 3.10+):
```bash
python -m venv venv
source venv/bin/activate  # Для Linux/macOS
# venv\Scripts\activate  # Для Windows
```

3. Установите зависимости:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. Запустите скрипт бенчмаркинга:
```bash
# ModernBERT на корпусе SST-2
python train_benchmark.py --model_name "answerdotai/ModernBERT-base" --dataset_name sst2 --batch_size 32 --epochs 3

# ModernBERT на корпусе AG News
python train_benchmark.py --model_name "answerdotai/ModernBERT-base" --dataset_name ag_news --batch_size 32 --epochs 3

# RoBERTa на корпусе SST-2
python train_benchmark.py --model_name "roberta-base" --dataset_name sst2 --batch_size 32 --epochs 3

# RoBERTa на корпусе AG News
python train_benchmark.py --model_name "roberta-base" --dataset_name ag_news --batch_size 32 --epochs 3

# SmolLM2-360M на корпусе SST-2
python train_benchmark.py --model_name "HuggingFaceTB/SmolLM2-360M" --dataset_name sst2 --batch_size 16 --epochs 3

# SmolLM2-360M на корпусе AG News
python train_benchmark.py --model_name "HuggingFaceTB/SmolLM2-360M" --dataset_name ag_news --batch_size 16 --epochs 3

# Qwen2.5-0.5B на корпусе SST-2
python train_benchmark.py --model_name "Qwen/Qwen2.5-0.5B" --dataset_name sst2 --batch_size 16 --epochs 3

# Qwen2.5-0.5B на корпусе AG News
python train_benchmark.py --model_name "Qwen/Qwen2.5-0.5B" --dataset_name ag_news --batch_size 16 --epochs 3
```

## Вычисляемые метрики
* **Качество классификации:** Accuracy, Macro-F1.
* **Вероятностная калибровка:** Cross-Entropy Loss, Brier Score (оценка надежности апостериорных вероятностей).
* **Вычислительная эффективность:** Время обучения на эпоху (сек/эпоха) и пиковое потребление видеопамяти VRAM (ГБ).
