# -*- coding: utf-8 -*-
"""
Демо: одна и та же модель до и после дообучения.

Слева базовая модель (best_model.pth), справа дообученная (chat_model.pth).
Веса разные, вопросы одинаковые. Хорошо видно, что дообучение поменяло
не знания, а поведение.

Запуск: python demo_finetune.py
"""
import sys

import torch

from config import settings
from model import SimpleLLM
from qa_data import build_prompt
from tokenizer_utils import get_tokenizer

sys.stdout.reconfigure(encoding="utf-8")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = get_tokenizer()


def load(weights):
    model = SimpleLLM(
        vocab_size=tokenizer.vocab_size,
        max_seq_length=settings.max_length,
        embed_dim=64,
        num_heads=2,
        num_layers=2,
        ff_dim=256,
    ).to(device)
    model.load_state_dict(
        torch.load(weights, map_location=device, weights_only=True)
    )
    model.eval()
    return model


def answer(model, question, max_new_tokens=20, stop_on_eos=True):
    """Задаёт вопрос и возвращает то, что модель дописала после «Answer:»."""
    prompt_ids = tokenizer.encode(build_prompt(question))
    input_ids = torch.tensor([prompt_ids], device=device)
    produced = []

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(input_ids[:, -settings.max_length:])

        next_id = torch.argmax(logits[:, -1, :], dim=-1).item()

        if stop_on_eos and next_id == tokenizer.eos_token_id:
            break

        produced.append(next_id)
        input_ids = torch.cat(
            [input_ids, torch.tensor([[next_id]], device=device)], dim=1
        )

    return tokenizer.decode(produced).strip()


base = load("best_model.pth")
chat = load("chat_model.pth")

QUESTIONS = [
    "What is the cat name?",
    "Where did the cat live?",
    "Who loved the cat?",
]

print("=== 1. Базовая модель: продолжает текст вместо ответа ===")
for question in QUESTIONS:
    print(f"  {question}")
    print(f"    -> {answer(base, question)!r}")

print()
print("=== 2. Дообученная модель: отвечает и замолкает ===")
for question in QUESTIONS:
    print(f"  {question}")
    print(f"    -> {answer(chat, question)!r}")

print()
print("=== 3. Что будет, если не слушать EOS ===")
question = QUESTIONS[0]
print(f"  {question}")
print(f"    со стопом:  {answer(chat, question)!r}")
print(f"    без стопа:  {answer(chat, question, stop_on_eos=False)!r}")

print()
print("=== 4. Вопрос не из обучающего набора ===")
for question in ["What was the cat doing?", "What is the dog name?"]:
    print(f"  {question}")
    print(f"    -> {answer(chat, question)!r}")

print()
print("=== 5. Тот же факт, но вопрос задан другими словами ===")
for trained, rephrased in [
    ("What is the cat name?", "What is the name of the cat?"),
    ("Where did the cat live?", "Where does Barsik live?"),
]:
    print(f"  как учили:  {trained}")
    print(f"    -> {answer(chat, trained)!r}")
    print(f"  иначе:      {rephrased}")
    print(f"    -> {answer(chat, rephrased)!r}")
