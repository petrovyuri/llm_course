# Модуль 08 - Внимание без обучаемых весов

Знакомимся с механизмом внимания в самом простом виде - без обучаемых весов. Каждый токен «смотрит» на другие и собирает контекст: scores -> softmax -> контекстный вектор.

## Что нового в этом модуле

- `attention_simple.py` - self-attention без весов: `Q = K = V = x`, `scores = Q·Kᵀ`, `softmax`, `context = weights·V`.
- `demo_attention_manual.py` - разбор вычислений на примере «The cat loved sour cream», числа совпадают с ручным расчётом из урока.

## Как запустить

```bash
cd module-08-attention-basic
python demo_attention_manual.py
```
