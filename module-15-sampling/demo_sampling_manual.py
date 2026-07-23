# -*- coding: utf-8 -*-
"""
Демо: воспроизводим ручные расчёты уроков 2-5 модуля 15.

Словарь и логиты берём из ручного примера модуля 12:
слова ["кот", "любил", "сметану", "он", "был"],
logits = [0.085, 0.201, 0.944, 0.334, -0.155].
"""
import sys
import torch

from sampling import filter_top_k, filter_top_p

# Windows-консоль по умолчанию не умеет печатать Unicode
sys.stdout.reconfigure(encoding="utf-8")

WORDS = ["кот", "любил", "сметану", "он", "был"]
logits = torch.tensor([[0.085, 0.201, 0.944, 0.334, -0.155]])


def show(tag, probs):
    """Печатает распределение в одну строку."""
    row = ", ".join(f"{w}={p:.3f}" for w, p in zip(WORDS, probs[0]))
    print(f"{tag}: {row}  (сумма={probs.sum():.3f})")


print("=== Урок 2-3. Softmax и температура ===")
for T in (0.5, 1.0, 2.0):
    show(f"T={T}", torch.softmax(logits / T, dim=-1))

print()
print("=== Урок 4. Top-k=2 ===")
filtered_k = filter_top_k(logits, k=2)
print("логиты после фильтра:", filtered_k[0].tolist())
show("top-k=2", torch.softmax(filtered_k, dim=-1))

print()
print("=== Урок 5. Top-p=0.7 ===")
filtered_p = filter_top_p(logits, p=0.7)
print("логиты после фильтра:", filtered_p[0].tolist())
show("top-p=0.7", torch.softmax(filtered_p, dim=-1))
