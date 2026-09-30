# -*- coding: utf-8 -*-
"""
Что лежит в репозитории модели на Hugging Face.

Скачиваем файлы GPT-2 по одному, читаем конфигурацию, считаем, сколько
должен весить файл весов, и разбираемся, откуда берётся разница
с настоящим размером.

Запуск: python hf_files.py
"""
import json
import os
import sys

from huggingface_hub import hf_hub_download
from safetensors import safe_open

sys.stdout.reconfigure(encoding="utf-8")

REPO = "gpt2"

# === 1. Конфигурация: описание модели без единого веса ===
config_path = hf_hub_download(REPO, "config.json")

with open(config_path, encoding="utf-8") as f:
    config = json.load(f)

print(f"Скачали config.json: {os.path.getsize(config_path)} байт")
print()
for key in ("model_type", "vocab_size", "n_positions", "n_embd", "n_head", "n_layer"):
    print(f"  {key:12s} = {config[key]}")

# === 2. Файл весов ===
weights_path = hf_hub_download(REPO, "model.safetensors")
file_mb = os.path.getsize(weights_path) / 1e6

print()
print(f"Скачали model.safetensors: {file_mb:.1f} МБ")

# === 3. Заглядываем внутрь, не загружая веса в память ===
params = 0
mask_shapes = []
mask_bytes = 0

with safe_open(weights_path, framework="pt") as f:
    names = list(f.keys())

    for name in names:
        shape = tuple(f.get_slice(name).get_shape())

        count = 1
        for dim in shape:
            count *= dim

        # Маска внимания - не параметр: её не учат, а вычисляют.
        # GPT-2 сохранила её прямо в файле.
        if name.endswith(".attn.bias"):
            mask_shapes.append(shape)
            mask_bytes += count * 4
        else:
            params += count

print(f"Тензоров в файле: {len(names)}")
print(f"Обучаемых параметров: {params:,}")
print()

params_mb = params * 4 / 1e6
masks_mb = mask_bytes / 1e6

print("Считаем размер файла:")
print(f"  параметры:      {params:,} x 4 байта = {params_mb:.1f} МБ")
print(f"  маски внимания: {len(mask_shapes)} шт. x {masks_mb / len(mask_shapes):.2f} МБ = {masks_mb:.1f} МБ")
print(f"  вместе:         {params_mb + masks_mb:.1f} МБ")
print(f"  файл на диске:  {file_mb:.1f} МБ")
print()
print(f"Форма одной маски: {mask_shapes[0]}")
print("Это причинная маска из модуля 9, только на 1024 позиции вместо наших 32.")
