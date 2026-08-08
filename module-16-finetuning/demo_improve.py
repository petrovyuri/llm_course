# -*- coding: utf-8 -*-
"""
Два приёма, которые улучшают дообучение, и замер их эффекта.

Сравниваем три набора данных на одной и той же базовой модели:
    A - как в модуле: 12 пар, по одной формулировке на факт
    B - аугментация: у каждого факта две разные формулировки вопроса
    C - аугментация плюс примеры отказа «I do not know»

Меряем на трёх наборах вопросов:
    обучающие      - выучила ли модель то, что ей показывали
    новый перефраз - третья формулировка, её в обучении НЕ было
    не по теме     - вопросы, ответа на которые в рассказе нет

Запуск: python demo_improve.py
"""
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import settings
from model import SimpleLLM
from qa_dataset import QADataset
from tokenizer_utils import get_tokenizer

sys.stdout.reconfigure(encoding="utf-8")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = get_tokenizer()

# По три формулировки на каждый факт.
# Первая идёт в обучение всегда, вторая - только при аугментации,
# третья не участвует в обучении никогда: на ней и проверяем обобщение.
FACTS = [
    (["What is the cat name?",
      "What is the name of the cat?",
      "How is the cat called?"], "Barsik."),
    (["What did Barsik like?",
      "What did the cat like?",
      "What was Barsik fond of?"], "Barsik liked to sleep."),
    (["What did Barsik love?",
      "What did the cat love?",
      "What did Barsik love to do?"], "Barsik loved to eat."),
    (["Where did the cat live?",
      "Where does Barsik live?",
      "Where was the cat living?"], "He lived in the house."),
    (["What was the cat catching?",
      "What does the cat catch?",
      "Who did the cat catch?"], "The cat was catching mice."),
    (["Who was afraid of the cat?",
      "Who is afraid of the cat?",
      "Who feared the cat?"], "The mice were afraid of the cat."),
    (["What was the cat drinking?",
      "What did the cat drink?",
      "What does the cat drink?"], "He was drinking milk."),
    (["Where was the cat walking?",
      "Where did the cat walk?",
      "Where does the cat walk?"], "The cat was walking in the garden."),
    (["Who loved the cat?",
      "Who loves the cat?",
      "Who was fond of the cat?"], "The mistress loved the cat."),
    (["What was the cat doing in the evening?",
      "What did the cat do in the evening?",
      "What happens in the evening?"], "He was going to bed."),
    (["What was the cat dreaming about?",
      "What did the cat dream about?",
      "What was in the dream?"], "The cat was flying."),
    (["Where was the cat basking?",
      "Where did the cat bask?",
      "Where does the cat bask?"], "The cat was basking in the window."),
]

# Примеры отказа: учим модель говорить «не знаю» вместо выдумки
REFUSALS = [
    ("What is the dog name?", "I do not know."),
    ("How old is the cat?", "I do not know."),
    ("What is the capital of France?", "I do not know."),
    ("Do you like pizza?", "I do not know."),
]

# Вопросы не по теме, которых нет ни в одном обучающем наборе
OFFTOPIC = ["Who is the president?", "What is the weather today?"]


def build_pairs(mode):
    """Собирает набор пар для режима A, B или C."""
    pairs = [(forms[0], answer) for forms, answer in FACTS]

    if mode in ("B", "C"):
        pairs += [(forms[1], answer) for forms, answer in FACTS]

    if mode == "C":
        pairs += REFUSALS

    return pairs


def ask(model, question, max_new_tokens=20):
    """Задаёт вопрос дообученной модели, останавливается на EOS."""
    from qa_data import build_prompt

    model.eval()
    prompt_ids = tokenizer.encode(build_prompt(question))
    input_ids = torch.tensor([prompt_ids], device=device)
    produced = []

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(input_ids[:, -settings.max_length:])

        next_id = torch.argmax(logits[:, -1, :], dim=-1).item()

        if next_id == tokenizer.eos_token_id:
            break

        produced.append(next_id)
        input_ids = torch.cat(
            [input_ids, torch.tensor([[next_id]], device=device)], dim=1
        )

    return tokenizer.decode(produced).strip()


def finetune(pairs, num_epochs=200):
    """Дообучает свежую копию базовой модели на переданных парах."""
    torch.manual_seed(settings.seed)

    model = SimpleLLM(
        vocab_size=tokenizer.vocab_size,
        max_seq_length=settings.max_length,
        embed_dim=64,
        num_heads=2,
        num_layers=2,
        ff_dim=256,
    ).to(device)

    model.load_state_dict(
        torch.load("best_model.pth", map_location=device, weights_only=True)
    )

    loader = DataLoader(
        QADataset(pairs, tokenizer, settings.max_length),
        batch_size=4,
        shuffle=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for _ in range(num_epochs):
        model.train()
        for input_ids, labels in loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            loss = F.cross_entropy(
                model(input_ids).reshape(-1, tokenizer.vocab_size),
                labels.reshape(-1),
                ignore_index=-100,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


def matches(got, expected):
    """Сравнивает ответы, не придираясь к точке на конце."""
    return got.rstrip(".") == expected.rstrip(".")


results = {}

print(f"{'режим':>6} | {'пар':>4} | {'сек':>4} | обучающие | новый перефраз | отказы")

for mode in ("A", "B", "C"):
    pairs = build_pairs(mode)

    started = time.time()
    model = finetune(pairs)
    elapsed = time.time() - started

    trained = sum(matches(ask(model, f[0][0]), f[1]) for f in FACTS)
    rephrased = sum(matches(ask(model, f[0][2]), f[1]) for f in FACTS)
    refused = sum("know" in ask(model, q).lower() for q in OFFTOPIC)

    results[mode] = model
    print(f"{mode:>6} | {len(pairs):>4} | {elapsed:>4.0f} | "
          f"{trained:>6}/12 | {rephrased:>10}/12 | {refused}/2")

print()
print("=== Вопрос не по теме: было и стало ===")
for question in OFFTOPIC:
    print(f"  {question}")
    print(f"    A (12 пар):        {ask(results['A'], question)!r}")
    print(f"    C (с отказами):    {ask(results['C'], question)!r}")

print()
print("=== Осторожность имеет цену ===")
question = FACTS[1][0][2]
print(f"  {question}")
print(f"    A: {ask(results['A'], question)!r}")
print(f"    C: {ask(results['C'], question)!r}")
print(f"    правильный ответ: {FACTS[1][1]!r}")
