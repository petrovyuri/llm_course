# -*- coding: utf-8 -*-
"""
Публикуем свою модель на Hugging Face Hub.

Собираем папку ровно такого вида, какой мы видели в уроке 2 у GPT-2:
веса в safetensors, конфигурация в JSON, карточка в README.md.
Потом отправляем её в репозиторий.

Нужен аккаунт и токен с правом записи:
    hf auth login

Запуск: python publish.py
"""
import json
import os
import shutil
import sys

from huggingface_hub import HfApi

sys.stdout.reconfigure(encoding="utf-8")

# Подставьте своё имя пользователя на Hugging Face
REPO_ID = "ВАШ_НИК/simple-llm-barsik"

FOLDER = "hub_upload"

CARD = """---
license: mit
language:
- en
tags:
- text-generation
- educational
pipeline_tag: text-generation
---

# SimpleLLM Barsik

Учебная языковая модель, написанная с нуля на чистом PyTorch в курсе
«Математика LLM - Создаём ChatGPT своими руками».

## Что это

Decoder-only трансформер на 6 534 528 параметров: 2 блока, 2 головы
внимания, размер эмбеддинга 64, контекстное окно 32 токена.
Токенизатор - стандартный BPE от GPT-2, словарь 50 257 токенов.

Модель обучена на коротком рассказе про кота Барсика, а затем дообучена
на 12 парах «вопрос - ответ» по этому рассказу.

## Как пользоваться

Модель не совместима с классами transformers: это собственная архитектура
из курса. Нужен файл model.py оттуда же.

```python
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

weights = hf_hub_download("{repo_id}", "chat_model.safetensors")
config = hf_hub_download("{repo_id}", "simple_llm_config.json")
```

## Границы применимости

Модель отвечает на 12 вопросов, на которых её дообучали, и уверенно
выдумывает ответы на все остальные. Это учебный пример, показывающий
устройство языковой модели, а не рабочий инструмент.
"""


def build_folder():
    """Собирает папку для загрузки: веса, конфигурация, карточка."""
    if os.path.exists(FOLDER):
        shutil.rmtree(FOLDER)
    os.makedirs(FOLDER)

    shutil.copy("chat_model.safetensors", FOLDER)
    shutil.copy("simple_llm_config.json", FOLDER)

    with open(os.path.join(FOLDER, "README.md"), "w", encoding="utf-8") as f:
        f.write(CARD.format(repo_id=REPO_ID))

    return FOLDER


if __name__ == "__main__":
    folder = build_folder()

    print(f"Папка {folder} собрана:")
    for name in sorted(os.listdir(folder)):
        size = os.path.getsize(os.path.join(folder, name))
        print(f"  {name:28s} {size / 1e6:8.2f} МБ")

    with open("simple_llm_config.json", encoding="utf-8") as f:
        print()
        print("Конфигурация, которая поедет вместе с весами:")
        print(json.dumps(json.load(f), indent=2))

    if REPO_ID.startswith("ВАШ_НИК"):
        print()
        print("Впишите своё имя пользователя в REPO_ID и запустите ещё раз.")
        sys.exit(0)

    api = HfApi()
    api.create_repo(REPO_ID, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=folder, repo_id=REPO_ID, repo_type="model")

    print()
    print(f"Готово: https://huggingface.co/{REPO_ID}")
