# Материалы к ВКР

Репозиторий содержит код, данные и результаты экспериментов, использованные в выпускной квалификационной работе:

**«Повышение качества решения математических задач олимпиадного уровня с помощью LLM с открытыми весами»**

## Структура репозитория

- `notebooks/01_finetune_kaggle.ipynb` — запуск обучения в Kaggle.
- `scripts/finetune_tinker.py` — основной скрипт обучения.
- `notebooks/02_tinker_to_peft.ipynb` — конвертация Tinker-адаптера в PEFT-формат.
- `scripts/merge_adapter.py` — слияние PEFT-адаптера с весами базовой модели.
- `notebooks/03_validate_nemotron.ipynb` — валидация Nemotron Nano 30B.
- `notebooks/04_validate_gpt_oss.ipynb` — валидация gpt-oss-20b.
- `notebooks/05_check_data_leakage.ipynb` — проверка отсутствия пересечений между обучающими, валидационными и тестовыми задачами.
- `data/` — валидационный и тестовый наборы задач.
- `results/` — результаты валидационных и тестовых запусков.

## Среда запуска

Основные эксперименты выполнялись в Kaggle Notebook. В ноутбуках используются пути и подключаемые артефакты Kaggle.

## Веса моделей

Итоговые веса моделей доступны в Kaggle Datasets:

- **dapo40**: https://www.kaggle.com/datasets/tkaggl/merged-dapo40/
- **sb20**: https://www.kaggle.com/datasets/tkaggl/merged-sb20/
