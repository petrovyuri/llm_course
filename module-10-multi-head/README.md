# Модуль 10 - Multi-Head Attention

Запускаем несколько «голов» внимания параллельно. Каждая голова смотрит на текст под своим углом, затем результаты объединяются (concat) и проходят через выходную матрицу `W_O`.

## Что нового в этом модуле

- `attention_multihead.py` - Multi-Head Attention: разбиение на головы, внимание в каждой, `concat` + `W_O`.
- `demo_multihead_detailed.py` - подробный разбор на две головы; функция `demo_reproduce_manual()` воспроизводит ручной пример (`embed_dim = 4`).
- `test_multihead.py` - тесты форм и числа параметров.

## Как запустить

```bash
cd module-10-multi-head
python demo_multihead_detailed.py
python -m pytest -q
```
