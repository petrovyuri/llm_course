# Модуль 12 - Трансформер-блок и выходной слой

Собираем всё вместе в трансформер-блок (Pre-LN): нормализация -> Multi-Head Attention -> residual -> нормализация -> FFN -> residual. Сверху добавляем выходной слой: `logits -> softmax`.

## Что нового в этом модуле

- `transformer_block.py` - трансформер-блок с архитектурой Pre-LN и остаточными связями.
- `model.py` - класс `SimpleLLM`: эмбеддинги + стек блоков + выходной слой (`lm_head`).
- `test_model_shapes.py` - тест сквозных форм: `[batch, seq]` -> `logits [batch, seq, vocab]`.

## Как запустить

```bash
cd module-12-transformer-block
python -m pytest -q
```
