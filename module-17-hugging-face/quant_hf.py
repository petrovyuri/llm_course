# -*- coding: utf-8 -*-
"""
Квантизация настоящих моделей средствами библиотек.

Руками мы уже умеем. Теперь то же самое, но готовыми инструментами:
bfloat16 прямо при загрузке и int8 через optimum-quanto.

Запуск: python quant_hf.py
"""
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, QuantoConfig

sys.stdout.reconfigure(encoding="utf-8")

SMOL = "HuggingFaceTB/SmolLM2-135M-Instruct"


def memory_mb(model):
    """Сколько весов модели лежит в памяти."""
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    return total / 1e6


def timed_answer(model, tokenizer, inputs, label):
    start = time.perf_counter()
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=20, do_sample=False,
                                pad_token_id=tokenizer.pad_token_id)
    elapsed = time.perf_counter() - start

    new_tokens = output[0, inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True)

    print(f"[{label}]")
    print(f"  в памяти: {memory_mb(model):.1f} МБ | время: {elapsed:.2f} с")
    print(f"  ответ: {answer!r}")
    print()


# === 1. GPT-2: float32 против bfloat16 ===
print("=" * 60)
print("GPT-2: тот же файл, разные типы чисел")
print("=" * 60)

gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
gpt2_inputs = gpt2_tokenizer("The cat loved sour cream", return_tensors="pt")

for dtype, label in ((torch.float32, "float32"), (torch.bfloat16, "bfloat16")):
    model = AutoModelForCausalLM.from_pretrained("gpt2", dtype=dtype)
    model.eval()

    start = time.perf_counter()
    with torch.no_grad():
        output = model.generate(**gpt2_inputs, max_new_tokens=20, do_sample=False,
                                pad_token_id=gpt2_tokenizer.eos_token_id)
    elapsed = time.perf_counter() - start

    print(f"[{label}] в памяти: {memory_mb(model):.1f} МБ | время: {elapsed:.2f} с")
    print(f"  {gpt2_tokenizer.decode(output[0])}")
    print()

# === 2. SmolLM2: float32, bfloat16 и int8 ===
print("=" * 60)
print("SmolLM2-135M-Instruct: добавляем int8")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(SMOL)
messages = [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                       return_tensors="pt", return_dict=True)

model = AutoModelForCausalLM.from_pretrained(SMOL, dtype=torch.float32)
model.eval()
timed_answer(model, tokenizer, inputs, "float32")

model = AutoModelForCausalLM.from_pretrained(SMOL, dtype=torch.bfloat16)
model.eval()
timed_answer(model, tokenizer, inputs, "bfloat16, он же тип по умолчанию")

model = AutoModelForCausalLM.from_pretrained(SMOL, quantization_config=QuantoConfig(weights="int8"))
model.eval()
timed_answer(model, tokenizer, inputs, "int8 через optimum-quanto")

weight = model.model.layers[0].self_attn.q_proj.weight
print(f"Тип весов после квантизации: {type(weight).__name__}, {weight.qtype}")
print()

# === 3. memory_mb() выше соврала - разбираемся, почему ===
print("=" * 60)
print("В памяти получилось столько же, сколько у bfloat16. Почему?")
print("=" * 60)

real_bytes = 0
for p in model.parameters():
    if hasattr(p, "_data"):
        real_bytes += p._data.numel() * p._data.element_size()
        real_bytes += p._scale.numel() * p._scale.element_size()
    else:
        real_bytes += p.numel() * p.element_size()

print(f"memory_mb(), через element_size() : {memory_mb(model):.1f} МБ")
print(f"по-настоящему, через _data        : {real_bytes / 1e6:.1f} МБ")
print()
print("WeightQBytesTensor снаружи притворяется исходным типом (bfloat16),")
print("чтобы обычные операции PyTorch работали без переделки, - element_size()")
print("верит этой маскировке и не видит int8 внутри, он лежит в приватном")
print("атрибуте _data. Экономия при этом настоящая, просто не вчетверо:")
print("quanto квантует только nn.Linear, а таблицу эмбеддингов (пятую часть")
print("всех параметров) и веса LayerNorm оставляет как есть.")
print()
print("Ответ тот же, памяти меньше, а времени больше: на обычном процессоре")
print("нет быстрых команд для int8, поэтому числа приходится разворачивать обратно.")
print("Выигрыш по скорости даёт видеокарта или специальный формат вроде GGUF.")
