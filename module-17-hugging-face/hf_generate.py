# -*- coding: utf-8 -*-
"""
Генерация текста готовой моделью: метод generate().

В модуле 14 мы написали цикл генерации руками, в модуле 15 добавили
температуру, top-k и top-p. У transformers всё это лежит в одном методе.
Здесь мы смотрим на его параметры и убеждаемся, что внутри - наш же код.

Запуск: python hf_generate.py
"""
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from sampling import sample_next_token

sys.stdout.reconfigure(encoding="utf-8")

torch.manual_seed(42)

PROMPT = "The cat loved sour cream"

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.eval()

inputs = tokenizer(PROMPT, return_tensors="pt")

print(f"Промпт: {PROMPT!r}")
print(f"Токенов на входе: {inputs['input_ids'].shape[1]}")
print()


def show(label, **kwargs):
    """Генерирует продолжение и печатает его с подписью."""
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=20,
            pad_token_id=tokenizer.eos_token_id,
            **kwargs,
        )
    print(f"[{label}]")
    print(f"  {tokenizer.decode(output[0])}")
    print()


# === 1. Жадный выбор: то же, что наш модуль 14 ===
show("жадно, do_sample=False", do_sample=False)

# === 2. Сэмплирование: параметры из модуля 15 ===
show("T=0.7", do_sample=True, temperature=0.7)
show("top-k=50", do_sample=True, top_k=50)
show("top-p=0.9", do_sample=True, top_p=0.9)

# === 3. Параметры, которых у нас не было ===
show("штраф за повторы 1.5", do_sample=False, repetition_penalty=1.5)
show("запрет повторов биграмм", do_sample=False, no_repeat_ngram_size=2)
show("лучевой поиск, 4 луча", do_sample=False, num_beams=4)

# === 4. Сверка: наш сэмплер против generate ===
print("=" * 60)
print("Сверяем: наш sample_next_token против generate(do_sample=False)")
print("=" * 60)

with torch.no_grad():
    library = model.generate(
        **inputs, max_new_tokens=10, do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

ours = inputs["input_ids"]
for _ in range(10):
    with torch.no_grad():
        logits = model(ours).logits

    # Тот самый код из модуля 15, без единой правки
    next_token = sample_next_token(logits[:, -1, :], do_sample=False)
    ours = torch.cat([ours, next_token], dim=1)

print(f"generate:          {tokenizer.decode(library[0])!r}")
print(f"наш sample_next_token: {tokenizer.decode(ours[0])!r}")
print(f"Совпадает: {torch.equal(library, ours)}")

# === 5. pipeline: самый короткий путь ===
print()
print("=" * 60)
print("То же самое через pipeline")
print("=" * 60)

generator = pipeline("text-generation", model="gpt2", tokenizer=tokenizer)
result = generator(PROMPT, max_new_tokens=20, do_sample=False,
                   pad_token_id=tokenizer.eos_token_id)
print(result[0]["generated_text"])
