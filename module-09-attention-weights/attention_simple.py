"""
Модуль простого механизма самовнимания без обучаемых весов.
Используется только для образовательных целей, чтобы понять концепцию.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleSelfAttention(nn.Module):
    """
    Упрощённый механизм самовнимания без обучаемых весов.
    Query, Key, Value - это просто входные векторы.
    В полноценной версии будут обучаемые матрицы W_Q, W_K, W_V.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size, seq_length, embed_dim = x.shape

        # Шаг 1: Query, Key, Value = входные векторы (без преобразований)
        Q = x
        K = x
        V = x

        # Шаг 2: Attention Scores = Q @ K^T
        attention_scores = torch.matmul(Q, K.transpose(-2, -1))

        # Шаг 3: Attention Weights = softmax(scores)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Шаг 4: Context Vectors = weights @ V
        context_vectors = torch.matmul(attention_weights, V)

        if squeeze_output:
            context_vectors = context_vectors.squeeze(0)
            attention_weights = attention_weights.squeeze(0)
            attention_scores = attention_scores.squeeze(0)

        # Возвращаем 3 значения: контекст, веса и scores
        return context_vectors, attention_weights, attention_scores


def compute_attention_step_by_step(x, tokenizer=None, token_names=None):
    seq_length, embed_dim = x.shape
    if token_names is None:
        token_names = [f"Token_{i}" for i in range(seq_length)]

    print("=" * 70)
    print("МЕХАНИЗМ САМОВНИМАНИЯ - ПОШАГОВОЕ ВЫЧИСЛЕНИЕ")
    print("=" * 70)

    print("\n[ШАГ 0] Входные векторы:")
    print(f"Форма: {x.shape} ({seq_length} токенов, {embed_dim} измерений)")
    print()
    for i, name in enumerate(token_names):
        values = ", ".join([f"{v:.2f}" for v in x[i].tolist()])
        print(f"  {name:>8}: [{values}]")

    print("\n" + "-" * 70)
    print("[ШАГ 1] Attention Scores (скалярное произведение)")
    print("-" * 70)
    attention_scores = torch.empty(seq_length, seq_length)
    for i in range(seq_length):
        for j in range(seq_length):
            attention_scores[i, j] = torch.dot(x[i], x[j])
    print("Матрица scores:")
    header = "         " + "".join([f"{name:>10}" for name in token_names])
    print(header)
    for i, name in enumerate(token_names):
        row = f"{name:>8}" + "".join([f"{attention_scores[i, j]:>10.4f}" for j in range(seq_length)])
        print(row)

    print("\n" + "-" * 70)
    print("[ШАГ 2] Attention Weights (softmax по строкам)")
    print("-" * 70)
    attention_weights = F.softmax(attention_scores, dim=-1)
    print("Матрица weights:")
    print(header)
    for i, name in enumerate(token_names):
        row = f"{name:>8}" + "".join([f"{attention_weights[i, j]:>10.4f}" for j in range(seq_length)])
        print(row)
    print("\nПроверка сумм по строкам:")
    for i, name in enumerate(token_names):
        row_sum = attention_weights[i].sum().item()
        print(f"  {name:>8}: {row_sum:.6f} {'OK' if abs(row_sum - 1.0) < 0.0001 else 'X'}")

    print("\n" + "-" * 70)
    print("[ШАГ 3] Context Vectors (взвешенная сумма)")
    print("-" * 70)
    context_vectors = attention_weights @ x
    print("Контекстные векторы:")
    for i, name in enumerate(token_names):
        values = ", ".join([f"{v:.4f}" for v in context_vectors[i].tolist()])
        print(f"  {name:>8}: [{values}]")

    print("\n" + "=" * 70)
    print("ПОДРОБНЫЙ РАЗБОР для токена '" + token_names[1] + "'")
    print("=" * 70)
    mid_idx = 1
    print(f"\nВеса внимания для '{token_names[mid_idx]}':")
    for j, name in enumerate(token_names):
        print(f"  На '{name}': {attention_weights[mid_idx, j]:.4f} ({attention_weights[mid_idx, j] * 100:.1f}%)")
    print("\nВычисление по измерениям:")
    for dim in range(min(embed_dim, 3)):
        calc_parts = []
        result = 0
        for j in range(seq_length):
            term = attention_weights[mid_idx, j] * x[j, dim]
            result += term
            calc_parts.append(f"{attention_weights[mid_idx, j]:.3f}x{x[j, dim]:.2f}")
        print(f"  dim[{dim}] = " + " + ".join(calc_parts) + f" = {result:.4f}")

    return {
        "input": x,
        "attention_scores": attention_scores,
        "attention_weights": attention_weights,
        "context_vectors": context_vectors,
    }
