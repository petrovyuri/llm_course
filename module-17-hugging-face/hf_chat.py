# -*- coding: utf-8 -*-
"""
Разговор с настоящей чат-моделью.

GPT-2 - базовая модель: она продолжает текст, как наша до модуля 16.
SmolLM2-135M-Instruct уже прошла дообучение, поэтому отвечает на вопросы.
Смотрим, чем её архитектура отличается от GPT-2 и как ей задают вопрос.

Запуск: python hf_chat.py
"""
import sys
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(encoding="utf-8")

REPO = "HuggingFaceTB/SmolLM2-135M-Instruct"

# === 1. Чем эта архитектура отличается от GPT-2 ===
config = AutoConfig.from_pretrained(REPO)

print(f"Модель: {REPO}")
print(f"  класс архитектуры : {config.architectures[0]}")
print(f"  размер эмбеддинга : {config.hidden_size}")
print(f"  блоков            : {config.num_hidden_layers}")
print(f"  голов внимания    : {config.num_attention_heads}")
print(f"  голов для K и V   : {config.num_key_value_heads}  (grouped-query attention)")
print(f"  скрытый слой FFN  : {config.intermediate_size}")
print(f"  размер словаря    : {config.vocab_size}")
print(f"  контекстное окно  : {config.max_position_embeddings}")
print(f"  функция активации : {config.hidden_act}")
print(f"  веса связаны      : {config.tie_word_embeddings}")
print()
print("Таблицы позиций здесь нет вообще: позиция кодируется поворотом Q и K (RoPE).")
print()

tokenizer = AutoTokenizer.from_pretrained(REPO)
model = AutoModelForCausalLM.from_pretrained(REPO)
model.eval()

print(f"Параметров: {model.num_parameters():,}")
print(f"Тип чисел по умолчанию: {model.dtype}")
print()

# === 2. Как задают вопрос: шаблон разговора ===
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France? Answer in one sentence."},
]

as_text = tokenizer.apply_chat_template(messages, tokenize=False,
                                        add_generation_prompt=True)

print("Наши роли превратились в одну строку:")
print("-" * 60)
print(as_text)
print("-" * 60)
print()
print(f"Токен конца ответа: {tokenizer.eos_token!r} (id {tokenizer.eos_token_id})")
print("Это тот же EOS, которым мы заканчивали ответы в модуле 16.")
print()

# === 3. Ответ ===
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
)

start = time.perf_counter()
with torch.no_grad():
    output = model.generate(
        **inputs, max_new_tokens=30, do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
elapsed = time.perf_counter() - start

answer_ids = output[0, inputs["input_ids"].shape[1]:]

print(f"Ответ целиком (со служебными токенами): {tokenizer.decode(answer_ids)!r}")
print(f"Ответ для человека: {tokenizer.decode(answer_ids, skip_special_tokens=True)!r}")
print(f"Токенов сгенерировано: {len(answer_ids)}, время: {elapsed:.2f} с")
print()
print("Модель сама остановилась на EOS - ровно как наша chat-модель в модуле 16.")
