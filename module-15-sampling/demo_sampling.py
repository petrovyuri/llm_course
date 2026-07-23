# -*- coding: utf-8 -*-
"""
Демо: сравниваем режимы генерации на обученной модели.

Один и тот же промт генерируется по-разному:
greedy, чистое сэмплирование, температура, top-k, top-p.
Перед каждой генерацией фиксируем seed, чтобы вывод был воспроизводим.
"""
import sys
import torch

from generate import model, tokenizer, generate

sys.stdout.reconfigure(encoding="utf-8")

PROMPT = "The cat was"
N = 15  # сколько новых токенов генерируем


def run(title, seed=42, **params):
    torch.manual_seed(seed)
    text = generate(model, tokenizer, PROMPT, max_new_tokens=N, **params)
    print(f"[{title}]")
    print(f"  {text}")


print(f"Промт: '{PROMPT}'")

print("\n=== 1. Greedy: два запуска подряд ===")
run("greedy, запуск 1")
run("greedy, запуск 2")

print("\n=== 2. Сэмплирование T=1.0: три запуска ===")
run("T=1.0, seed=1", seed=1, do_sample=True, temperature=1.0)
run("T=1.0, seed=2", seed=2, do_sample=True, temperature=1.0)
run("T=1.0, seed=4", seed=4, do_sample=True, temperature=1.0)

print("\n=== 3. Температура: 0.5 / 1.0 / 2.0 ===")
run("T=0.5", do_sample=True, temperature=0.5)
run("T=1.0", do_sample=True, temperature=1.0)
run("T=2.0", do_sample=True, temperature=2.0)

print("\n=== 4. Отсекаем хвост: top-k и top-p ===")
run("T=2.0 + top-k=5", do_sample=True, temperature=2.0, top_k=5)
run("T=1.0 + top-p=0.9, seed=2", seed=2, do_sample=True, temperature=1.0, top_p=0.9)
