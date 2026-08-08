# -*- coding: utf-8 -*-
"""
Разговор с дообученной моделью.

Отличие от generate.py модуля 15: там мы просто продолжали текст заданной
длины. Здесь модель должна сама решить, где закончить ответ, и мы её
слушаем: как только выпал токен EOS, генерацию останавливаем.

Запуск: python chat.py
Нужен файл chat_model.pth от finetune.py.
"""
import sys

import torch

from config import settings
from model import SimpleLLM
from qa_data import build_prompt
from sampling import sample_next_token
from tokenizer_utils import get_tokenizer

sys.stdout.reconfigure(encoding="utf-8")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = get_tokenizer()

model = SimpleLLM(
    vocab_size=tokenizer.vocab_size,
    max_seq_length=settings.max_length,
    embed_dim=64,
    num_heads=2,
    num_layers=2,
    ff_dim=256,
).to(device)

model.load_state_dict(
    torch.load("chat_model.pth", map_location=device, weights_only=True)
)
model.eval()


def ask(question, max_new_tokens=20, do_sample=False, temperature=1.0,
        top_k=None, top_p=None):
    """
    Задаёт модели вопрос и возвращает ответ.

    Генерация останавливается на токене EOS - том самом, которым
    заканчивался каждый ответ в обучающих примерах.
    """
    prompt_ids = tokenizer.encode(build_prompt(question))
    input_ids = torch.tensor([prompt_ids], device=device)

    answer_ids = []

    for _ in range(max_new_tokens):
        context = input_ids[:, -settings.max_length:]

        with torch.no_grad():
            logits = model(context)

        next_token = sample_next_token(
            logits[:, -1, :],
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        # Модель сказала «я закончил»
        if next_token.item() == tokenizer.eos_token_id:
            break

        answer_ids.append(next_token.item())
        input_ids = torch.cat([input_ids, next_token], dim=1)

    return tokenizer.decode(answer_ids).strip()


if __name__ == "__main__":
    print("Спросите модель о коте Барсике (пустая строка - выход).")
    print("Например: What is the cat name?")
    print()

    while True:
        question = input("Вопрос: ").strip()

        if not question:
            print("Пока!")
            break

        print(f"Ответ:  {ask(question)}")
        print()
