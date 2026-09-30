# -*- coding: utf-8 -*-
"""
Скачиваем свою модель с Hub и разговариваем с ней.

Круг замкнулся: модель обучена, дообучена, переведена в safetensors,
опубликована - и теперь возвращается из интернета обратно к нам.

Запуск: python load_from_hub.py
"""
import sys

from huggingface_hub import hf_hub_download

from quant_manual import ask
from save_safetensors import load_model_from_safetensors

sys.stdout.reconfigure(encoding="utf-8")

# То же имя, что в publish.py
REPO_ID = "ВАШ_НИК/simple-llm-barsik"

if REPO_ID.startswith("ВАШ_НИК"):
    print("Впишите имя своего репозитория в REPO_ID.")
    sys.exit(0)

weights = hf_hub_download(REPO_ID, "chat_model.safetensors")
config = hf_hub_download(REPO_ID, "simple_llm_config.json")

print(f"Скачали веса: {weights}")
print(f"Скачали конфигурацию: {config}")
print()

model = load_model_from_safetensors(weights, config)
print(f"Модель собрана: {model.get_num_parameters():,} параметров")
print()

for question in ("What is the cat name?", "Where did the cat live?",
                 "What was the cat dreaming about?"):
    print(f"Вопрос: {question}")
    print(f"Ответ:  {ask(model, question)}")
    print()
