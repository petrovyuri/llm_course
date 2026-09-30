# -*- coding: utf-8 -*-
"""
Дообучение (fine-tuning) базовой модели на парах «вопрос - ответ».

Обучение с нуля (train.py) дало модели язык: она знает слова из рассказа
и умеет продолжать текст. Здесь мы берём эти же веса и показываем модели
новый формат поведения: на вопрос отвечают коротко и по делу.

Запуск: python finetune.py
Нужен файл best_model.pth от train.py с тем же max_length.
"""
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import settings
from model import SimpleLLM
from qa_data import QA_PAIRS
from qa_dataset import QADataset, IGNORE_INDEX
from tokenizer_utils import get_tokenizer

sys.stdout.reconfigure(encoding="utf-8")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = get_tokenizer()
vocab_size = tokenizer.vocab_size

# === Данные ===
dataset = QADataset(QA_PAIRS, tokenizer, settings.max_length)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

print(f"Устройство: {device}")
print(f"Пар вопрос-ответ: {len(dataset)}")

# === Модель: та же архитектура, что и при обучении с нуля ===
model = SimpleLLM(
    vocab_size=vocab_size,
    max_seq_length=settings.max_length,
    embed_dim=64,
    num_heads=2,
    num_layers=2,
    ff_dim=256,
).to(device)

# Ключевой момент: мы не создаём модель заново, а берём обученные веса
model.load_state_dict(
    torch.load("best_model.pth", map_location=device, weights_only=True)
)
print("Загружены веса базовой модели: best_model.pth")

# Learning rate меньше, чем при обучении с нуля (там был 3e-4).
# Модель уже умеет говорить, нам нужно аккуратно подправить поведение,
# а не переучить её заново.
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 200
best_loss = float("inf")

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for input_ids, labels in loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        logits = model(input_ids)

        # ignore_index=-100: позиции промпта не влияют на loss
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)

    if (epoch + 1) % 25 == 0 or epoch == 0:
        print(f"Epoch {epoch + 1}: loss = {avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss

torch.save(model.state_dict(), "chat_model.pth")

print()
print(f"Дообучение завершено. Лучший loss: {best_loss:.4f}")
print("Модель сохранена: chat_model.pth")
