# -*- coding: utf-8 -*-
"""
Сохраняем свою модель в формате Hugging Face.

Раньше мы клали веса в chat_model.pth через torch.save. Это pickle: файл
умеет исполнять код при загрузке, поэтому мы и писали weights_only=True.
Safetensors устроен проще и безопаснее: заголовок JSON с описанием тензоров
плюс сырые байты. Кода внутри нет, исполнять нечего.

Запуск: python save_safetensors.py
Нужен файл chat_model.pth от finetune.py.
"""
import json
import sys

import torch
from safetensors.torch import load_file, save_file

from model import SimpleLLM

sys.stdout.reconfigure(encoding="utf-8")

# Конфигурация нашей модели: то же, что мы передавали руками в chat.py.
# Теперь она поедет вместе с весами, как config.json у настоящих моделей.
MODEL_CONFIG = {
    "vocab_size": 50257,
    "max_seq_length": 32,
    "embed_dim": 64,
    "num_heads": 2,
    "num_layers": 2,
    "ff_dim": 256,
}


def save_model_to_safetensors(pth_path, weights_path, config_path):
    """Переводит веса из .pth в .safetensors и кладёт рядом конфигурацию."""
    state_dict = torch.load(pth_path, map_location="cpu", weights_only=True)

    # safetensors хранит сырые байты, поэтому тензоры должны лежать
    # в памяти сплошным куском
    state_dict = {name: tensor.contiguous() for name, tensor in state_dict.items()}
    save_file(state_dict, weights_path)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(MODEL_CONFIG, f, indent=2)

    return state_dict


def load_model_from_safetensors(weights_path, config_path):
    """Собирает SimpleLLM по конфигурации и заливает в неё веса."""
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    model = SimpleLLM(**config)
    model.load_state_dict(load_file(weights_path))
    model.eval()
    return model


if __name__ == "__main__":
    import os

    state_dict = save_model_to_safetensors(
        "chat_model.pth", "chat_model.safetensors", "simple_llm_config.json"
    )

    params = sum(t.numel() for t in state_dict.values())
    print(f"Параметров: {params:,}")
    print(f"chat_model.pth:         {os.path.getsize('chat_model.pth') / 1e6:.2f} МБ")
    print(f"chat_model.safetensors: {os.path.getsize('chat_model.safetensors') / 1e6:.2f} МБ")
    print()

    model = load_model_from_safetensors("chat_model.safetensors", "simple_llm_config.json")
    print(f"Модель собрана заново: {model.get_num_parameters():,} параметров")
