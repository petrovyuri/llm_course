import torch
import torch.nn.functional as F
import os
import json
from model import SimpleLLM
from config import settings
from loader import create_dataloader
from tokenizer_utils import get_tokenizer

# Читаем текст из файла
with open(settings.text_file, "r", encoding="utf-8") as f:
    raw_text = f.read()

# Получаем токенизатор и размер словаря
tokenizer = get_tokenizer()
vocab_size = tokenizer.vocab_size
max_seq_length = settings.max_length

# Создаем загрузчик данных для обучения
train_loader = create_dataloader(raw_text, shuffle=True, drop_last=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Новое. Добавим блок для вывода информации о состоянии обучения и размере словаря.
print(f"Обучение на устройстве: {device}")  # Новое
print(f"Размер словаря: {vocab_size}")  # Новое
print(f"Максимальная длина последовательности: {max_seq_length}")  # Новое

model = SimpleLLM(
    vocab_size=vocab_size,
    max_seq_length=max_seq_length,
    embed_dim=64,
    num_heads=2,
    num_layers=2,
    ff_dim=256,
).to(device)

print(f"Количество параметров модели: {model.get_num_parameters():,}")

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

# Количество эпох
num_epochs = settings.num_epochs

# Отслеживаем лучший loss
best_loss = float('inf')  # Новое

for epoch in range(num_epochs):
    model.train()  # переводим модель в режим обучения

    total_loss = 0.0

    for batch_x, batch_y in train_loader:
        input_ids = batch_x.to(device)
        target_ids = batch_y.to(device)

        # 1. Forward pass
        logits = model(input_ids)   # [B, T, vocab_size]

        # 2. Loss
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),   # [B*T, vocab_size]
            target_ids.reshape(-1)                 # [B*T]
        )

        # 3. Обнуляем старые градиенты
        optimizer.zero_grad()

        # 4. Backward pass
        loss.backward()

        # 5. Обновляем веса
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}: loss = {avg_loss:.4f}")

    # Новое. Сохраняем лучшую модель
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "best_model.pth")
        print(f"  -> Сохранена лучшая модель с потерей: {best_loss:.4f}")

# Новое. Сохраняем финальную модель
torch.save(model.state_dict(), "final_model.pth")
print(f"\nОбучение завершено!")
print(f"Лучшая потеря: {best_loss:.4f}")
print(f"Модели сохранены: best_model.pth, final_model.pth")

# Сохраняем информацию о токенизаторе и настройках
config_info = {
    "vocab_size": vocab_size,
    "max_seq_length": max_seq_length,
    "embed_dim": 64,
    "num_heads": 2,
    "num_layers": 2,
    "ff_dim": 256,
    "model_name": settings.model_name,
}

with open("model_config.json", "w", encoding="utf-8") as f:
    json.dump(config_info, f, indent=2)

print(f"Конфигурация сохранена: model_config.json")
