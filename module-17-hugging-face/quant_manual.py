# -*- coding: utf-8 -*-
"""
Квантизация своими руками: из float32 в int8.

Каждый вес модели занимает 4 байта. Если хранить его в одном байте,
модель похудеет вчетверо. Цена - округление: вместо любого числа
мы храним одно из 256 значений и множитель, который возвращает масштаб.

Запуск: python quant_manual.py
"""
import os
import sys

import torch
from safetensors.torch import load_file, save_file

from config import settings
from qa_data import QA_PAIRS, build_prompt
from sampling import sample_next_token
from save_safetensors import load_model_from_safetensors
from tokenizer_utils import get_tokenizer

sys.stdout.reconfigure(encoding="utf-8")

WEIGHTS = "chat_model.safetensors"
CONFIG = "simple_llm_config.json"
INT8_WEIGHTS = "chat_model_int8.safetensors"

tokenizer = get_tokenizer()


def quantize_int8(tensor):
    """
    Симметричная квантизация по строкам.

    Для каждой строки находим самый большой по модулю вес и растягиваем
    диапазон так, чтобы он попал в 127. Множитель (scale) храним отдельно.

    Возвращает целые числа и множители.
    """
    if tensor.dim() < 2:
        scale = tensor.abs().max().clamp(min=1e-8) / 127
    else:
        scale = tensor.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127

    quantized = torch.round(tensor / scale).clamp(-127, 127).to(torch.int8)
    return quantized, scale


def dequantize(quantized, scale):
    """Возвращает веса в float32: целое число умножаем на множитель."""
    return quantized.float() * scale


def build_int8_weights_file():
    """Квантует все веса модели и сохраняет их рядом с множителями в INT8_WEIGHTS."""
    original = load_file(WEIGHTS)

    packed = {}
    for name, weight in original.items():
        q, scale = quantize_int8(weight)
        packed[name] = q
        packed[f"{name}.scale"] = scale

    save_file(packed, INT8_WEIGHTS)


def build_fp32_model():
    return load_model_from_safetensors(WEIGHTS, CONFIG)


def build_int8_model():
    """Собирает модель из int8-файла: читаем целые, умножаем на множители."""
    saved = load_file(INT8_WEIGHTS)

    state_dict = {}
    for name in [n for n in saved if not n.endswith(".scale")]:
        state_dict[name] = dequantize(saved[name], saved[f"{name}.scale"])

    model = load_model_from_safetensors(WEIGHTS, CONFIG)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def ask(model, question, max_new_tokens=20):
    """Задаёт вопрос дообученной модели - как chat.py из модуля 16."""
    input_ids = torch.tensor([tokenizer.encode(build_prompt(question))])
    answer_ids = []

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(input_ids[:, -settings.max_length:])

        next_token = sample_next_token(logits[:, -1, :], do_sample=False)

        if next_token.item() == tokenizer.eos_token_id:
            break

        answer_ids.append(next_token.item())
        input_ids = torch.cat([input_ids, next_token], dim=1)

    return tokenizer.decode(answer_ids).strip()


def answer_all(model):
    """Ответы на все 12 обучающих вопросов."""
    return [ask(model, question) for question, _ in QA_PAIRS]


if __name__ == "__main__":
    original = load_file(WEIGHTS)

    # === 1. Одна матрица: смотрим, что теряем ===
    print("Квантуем одну матрицу и смотрим на ошибку:")
    print()
    for name in ("token_embedding.weight", "lm_head.weight"):
        weight = original[name]
        q, scale = quantize_int8(weight)
        restored = dequantize(q, scale)

        print(f"  {name}")
        print(f"    самый большой вес : {weight.abs().max().item():.4f}")
        print(f"    шаг квантования   : {scale.max().item():.6f}")
        print(f"    худшая ошибка     : {(restored - weight).abs().max().item():.6f}")
    print()

    # === 2. Вся модель ===
    build_int8_weights_file()

    fp32_size = os.path.getsize(WEIGHTS) / 1e6
    int8_size = os.path.getsize(INT8_WEIGHTS) / 1e6

    params = sum(t.numel() for t in original.values())
    print(f"Параметров: {params:,}")
    print(f"  float32: {fp32_size:.2f} МБ")
    print(f"  int8 плюс множители: {int8_size:.2f} МБ")
    print(f"  сжатие в {fp32_size / int8_size:.2f} раза")
    print()

    # === 3. Проверяем на деле: те же ответы или нет ===
    fp32_answers = answer_all(build_fp32_model())
    int8_answers = answer_all(build_int8_model())

    same = sum(a == b for a, b in zip(fp32_answers, int8_answers))

    print(f"{'вопрос':40s} | {'float32':32s} | int8")
    print("-" * 110)
    for (question, _), a32, a8 in zip(QA_PAIRS, fp32_answers, int8_answers):
        mark = " " if a32 == a8 else "!"
        print(f"{mark}{question:39s} | {a32:32s} | {a8}")

    print()
    print(f"Совпало ответов: {same} из {len(QA_PAIRS)}")
