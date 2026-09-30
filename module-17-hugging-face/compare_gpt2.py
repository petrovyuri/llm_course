# -*- coding: utf-8 -*-
"""
Наша модель и настоящая GPT-2: одно устройство, разные числа.

Открываем GPT-2 через transformers и раскладываем её веса рядом с нашими.
Имена другие, смысл тот же.

Запуск: python compare_gpt2.py
"""
import json
import sys

from transformers import AutoConfig, AutoModelForCausalLM

from model import SimpleLLM

sys.stdout.reconfigure(encoding="utf-8")

# === 1. Две конфигурации рядом ===
gpt2_config = AutoConfig.from_pretrained("gpt2")

with open("simple_llm_config.json", encoding="utf-8") as f:
    ours_config = json.load(f)

rows = [
    ("размер эмбеддинга", ours_config["embed_dim"], gpt2_config.n_embd),
    ("голов внимания", ours_config["num_heads"], gpt2_config.n_head),
    ("блоков трансформера", ours_config["num_layers"], gpt2_config.n_layer),
    ("контекстное окно", ours_config["max_seq_length"], gpt2_config.n_positions),
    ("скрытый слой FFN", ours_config["ff_dim"], 4 * gpt2_config.n_embd),
    ("размер словаря", ours_config["vocab_size"], gpt2_config.vocab_size),
]

print(f"{'параметр':22s} | {'наша':>11s} | {'GPT-2':>11s}")
print("-" * 50)
for name, ours_value, gpt2_value in rows:
    print(f"{name:22s} | {ours_value:>11,} | {gpt2_value:>11,}")

# === 2. Две модели рядом ===
ours = SimpleLLM(**ours_config)
gpt2 = AutoModelForCausalLM.from_pretrained("gpt2")

print()
print(f"Параметров у нашей: {ours.get_num_parameters():,}")
print(f"Параметров у GPT-2: {gpt2.num_parameters():,}")
print(f"GPT-2 больше в {gpt2.num_parameters() / ours.get_num_parameters():.0f} раз")

# === 3. Кто есть кто: сопоставляем веса по именам ===
# Левая колонка - имена из нашей модели, правая - из GPT-2.
# У GPT-2 нет отдельных матриц для Q, K и V - все три являются срезами
# одной и той же c_attn, поэтому в правой колонке эта строка повторяется
# трижды с одной и той же формой.
pairs = [
    ("token_embedding.weight", "transformer.wte.weight", "таблица токенов"),
    ("position_embedding.weight", "transformer.wpe.weight", "таблица позиций"),
    ("blocks.0.norm_1.weight", "transformer.h.0.ln_1.weight", "нормализация перед вниманием"),
    ("blocks.0.attention.W_query.weight", "transformer.h.0.attn.c_attn.weight", "матрица запросов"),
    ("blocks.0.attention.W_key.weight", "transformer.h.0.attn.c_attn.weight", "матрица ключей"),
    ("blocks.0.attention.W_value.weight", "transformer.h.0.attn.c_attn.weight", "матрица значений"),
    ("blocks.0.attention.W_output.weight", "transformer.h.0.attn.c_proj.weight", "выход внимания"),
    ("blocks.0.norm_2.weight", "transformer.h.0.ln_2.weight", "нормализация перед FFN"),
    ("blocks.0.ffn.linear_1.weight", "transformer.h.0.mlp.c_fc.weight", "первый слой FFN"),
    ("blocks.0.ffn.linear_2.weight", "transformer.h.0.mlp.c_proj.weight", "второй слой FFN"),
    ("final_norm.weight", "transformer.ln_f.weight", "финальная нормализация"),
    ("lm_head.weight", "lm_head.weight", "проекция в словарь"),
]

ours_state = ours.state_dict()
gpt2_state = gpt2.state_dict()

print()
print(f"{'что это':30s} | {'у нас':>16s} | {'у GPT-2':>18s}")
print("-" * 72)
for our_name, gpt2_name, meaning in pairs:
    our_shape = tuple(ours_state[our_name].shape)
    gpt2_shape = tuple(gpt2_state[gpt2_name].shape)
    print(f"{meaning:30s} | {str(our_shape):>16s} | {str(gpt2_shape):>18s}")

n_embd = gpt2_config.n_embd
print()
print(f"Три строки с c_attn - не совпадение: Q, K и V - срезы одной матрицы {n_embd} x {3 * n_embd},")
print(f"а не три отдельные матрицы по {n_embd} x {n_embd}, как у нас.")

# === 4. Weight tying: выходной слой и эмбеддинги - один тензор ===
print()
tied = gpt2.lm_head.weight.data_ptr() == gpt2.transformer.wte.weight.data_ptr()
print(f"У GPT-2 lm_head и wte - один и тот же тензор: {tied}")

saved = gpt2_config.vocab_size * gpt2_config.n_embd
ours_both = ours_config["vocab_size"] * ours_config["embed_dim"] * 2
print(f"Это экономит {saved:,} параметров: выходной слой не хранится отдельно.")
print(f"У нас эти матрицы разные, на вход и выход уходит {ours_both:,} параметров.")
