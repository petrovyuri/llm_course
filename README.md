# LLM своими руками - код курса

Учебный код к курсу [«LLM - Создаём Большую Языковую Модель своими руками»](https://stepik.org/a/276236) на Stepik. Автор курса - Юрий Петров (Friflex).

Здесь мы шаг за шагом собираем миниатюрную GPT-модель на PyTorch: от подготовки текста и токенизации до механизма внимания, трансформер-блока, обучения и генерации текста с управляемым сэмплированием.

## Как устроен репозиторий

Курс последовательный - каждый модуль продолжает предыдущий. Поэтому **каждая папка `module-NN-*` - это полный рабочий снимок проекта на конец соответствующего модуля**. Любой модуль можно открыть и запустить отдельно, а сравнив соседние папки, легко увидеть, какой код добавился на очередном шаге.

Сквозной пример через весь курс - предложение «The cat loved sour cream» («Кот любит сметану») и текст `cat_story.txt`.

| Модуль | Что добавляется | Ключевые файлы |
|--------|-----------------|----------------|
| [module-06-text-prep](module-06-text-prep/) | Скользящее окно, `Dataset`, `DataLoader` | `dataset.py`, `loader.py` |
| [module-07-embeddings](module-07-embeddings/) | Токенные и позиционные эмбеддинги | `embedding.py` |
| [module-08-attention-basic](module-08-attention-basic/) | Внимание без обучаемых весов | `attention_simple.py` |
| [module-09-attention-weights](module-09-attention-weights/) | Внимание с обучаемыми Q/K/V и каузальной маской | `attention.py` |
| [module-10-multi-head](module-10-multi-head/) | Multi-Head Attention | `attention_multihead.py` |
| [module-11-feed-forward](module-11-feed-forward/) | Feed-Forward сеть | `feed_forward.py` |
| [module-12-transformer-block](module-12-transformer-block/) | Трансформер-блок и модель `SimpleLLM` | `transformer_block.py`, `model.py` |
| [module-13-training](module-13-training/) | Цикл обучения (loss, backprop, оптимизатор) | `train.py` |
| [module-14-generation](module-14-generation/) | Авторегрессионная генерация текста | `generate.py` |
| [module-15-sampling](module-15-sampling/) | Температура, top-k, top-p | `sampling.py` |

## Требования

- Python 3.11
- Зависимости из [requirements.txt](requirements.txt): `torch` 2.10 (CPU), `transformers` 5.3

При первом запуске `transformers` скачает токенизатор `gpt2` из интернета.

## Установка

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## Запуск

Каждый модуль запускается из своей папки (импорты внутри модуля плоские):

```bash
cd module-15-sampling

# Демонстрационный прогон
python main.py

# Обучение модели (сохранит best_model.pth / final_model.pth)
python train.py

# Генерация текста (нужен обученный best_model.pth)
python generate.py

# Тесты
python -m pytest -q
```

> На Windows для корректного вывода символов вроде `×·Σ` задайте кодировку:
> `set PYTHONIOENCODING=utf-8` (cmd) или `$env:PYTHONIOENCODING="utf-8"` (PowerShell).

## Тесты

Начиная с модуля 9 к коду прилагаются тесты (`test_*.py`). Прогнать все тесты во всех модулях:

```bash
for dir in module-*/; do
  ls "$dir"test_*.py >/dev/null 2>&1 && (cd "$dir" && python -m pytest -q)
done
```

Те же тесты запускаются автоматически в GitHub Actions - см. [.github/workflows/tests.yml](.github/workflows/tests.yml).
