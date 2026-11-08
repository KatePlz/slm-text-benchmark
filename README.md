# Сравнительный анализ компактных языковых моделей при классификации текстовых данных

Данный репозиторий содержит воспроизводимый программный комплекс для экспериментального исследования эффективности, надежности калибровки вероятностей и вычислительной трудоемкости современных компактных языковых моделей (энкодеров и декодеров) при параметрически-эффективном дообучении методом LoRA (Low-Rank Adaptation).

## Структура репозитория

```
.
├── train_benchmark.py       # Основной параметризованный скрипт дообучения и профилирования
├── evaluate_metrics.py      # Модуль расчета метрик (Accuracy, Macro-F1, Cross-Entropy, Brier Score)
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
* **SST-2** (Stanford Sentiment Treebank) — бинарная классификация тональности предложений; загружается из репозитория `nyu-mll/glue` (67 349 обучающих / 872 валидационных примеров).
* **AG News** — многоклассовая тематическая классификация новостей (4 категории); загружается из репозитория `fancyzhx/ag_news` (120 000 обучающих / 7 600 тестовых примеров).

Оба набора данных скачиваются автоматически с Hugging Face Hub при первом запуске.

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

## Параметры запуска

| Параметр | По умолчанию | Назначение |
|---|---|---|
| `--model_name` | `answerdotai/ModernBERT-base` | Идентификатор модели на Hugging Face Hub или локальный путь |
| `--dataset_name` | `sst2` | Датасет: `sst2` или `ag_news` |
| `--output_dir` | `./results` | Каталог для чекпоинтов и метрик |
| `--batch_size` | `32` | Размер мини-пакета (в статье 32, для Qwen2.5-0.5B — 16) |
| `--learning_rate` | `2e-4` | Скорость обучения AdamW |
| `--epochs` | `3` | Количество эпох обучения |
| `--lora_r` | `8` | Ранг адаптера LoRA |
| `--lora_alpha` | `16` | Множитель масштабирования LoRA |
| `--lora_dropout` | `0.05` | Вероятность dropout в адаптере |
| `--max_length` | `128` | Максимальная длина последовательности (токенов) |
| `--seed` | `42` | Зерно генератора случайных чисел |
| `--max_train_samples` | нет | Ограничение обучающей выборки (для отладки) |
| `--max_eval_samples` | нет | Ограничение оценочной выборки (для отладки) |

Целевые модули LoRA определяются автоматически по архитектуре модели (`model_type`): `attn.Wqkv`/`attn.Wo` для ModernBERT, `query`/`key`/`value`/`attention.output.dense` для RoBERTa, `q_proj`/`k_proj`/`v_proj`/`o_proj` для декодеров SmolLM2 и Qwen.

## Вычисляемые метрики
* **Качество классификации:** Accuracy, Macro-F1.
* **Вероятностная калибровка:** Cross-Entropy Loss, Brier Score (оценка надежности апостериорных вероятностей).
* **Вычислительная эффективность:** Время обучения на эпоху (сек/эпоха) и пиковое потребление видеопамяти VRAM (ГБ).

Скрипты вычисляют только числовые метрики; построение графиков не выполняется.

## Результаты запуска

Для каждого запуска создается каталог `results/{модель}_{датасет}/`, содержащий:
* `benchmark_metrics.txt` и `metrics.json` — итоговые метрики оценки, время эпохи и пиковое потребление VRAM;
* `checkpoint-*/` — чекпоинты с весами адаптеров LoRA (сохраняется лучшая модель по метрике Macro-F1).

Примечание: режим FP16 и замер VRAM активируются автоматически только при наличии CUDA-устройства; на CPU значение `max_vram_gb` равно 0.
