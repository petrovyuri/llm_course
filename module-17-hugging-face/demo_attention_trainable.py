# demo_attention_trainable.py
"""
Демонстрация механизма самовнимания с обучаемыми весами.
Показывает все промежуточные вычисления пошагово.
"""

import torch
import torch.nn as nn
from attention import SelfAttention, CausalMask


def demo_with_custom_weights():
    """
    Демонстрация с заданными весами (как в ручном примере из книги).
    """
    print("\n" + "=" * 70)
    print("ДЕМОНСТРАЦИЯ: Внимание с обучаемыми весами")
    print("=" * 70)
    
    # ============================================================
    # Исходные данные (те же векторы, что в книге)
    # ============================================================
    inputs = torch.tensor([
        [0.43, 0.15, 0.89],  # The
        [0.55, 0.87, 0.66],  # cat
        [0.57, 0.85, 0.64],  # loved
    ], dtype=torch.float32)
    
    token_names = ["The", "cat", "loved"]
    embed_dim = inputs.shape[1]
    
    print(f"\nВходные векторы ({len(token_names)} токенов × {embed_dim} измерений):")
    for i, name in enumerate(token_names):
        values = ", ".join([f"{v:.2f}" for v in inputs[i].tolist()])
        print(f"  {name:>8}: [{values}]")
    
    # ============================================================
    # Создаём механизм внимания
    # ============================================================
    attention = SelfAttention(embed_dim=embed_dim)
    attention.eval()  # Режим оценки (без dropout и т.д.)
    
    # ============================================================
    # Устанавливаем веса вручную (как в примере книги)
    # ============================================================
    print("\n" + "-" * 70)
    print("Устанавливаем веса матриц (как в ручном примере)")
    print("-" * 70)
    
    # W_Q
    W_Q_manual = torch.tensor([
        [0.5, 0.2, 0.1],
        [0.3, 0.6, 0.2],
        [0.1, 0.1, 0.7],
    ], dtype=torch.float32)
    
    # W_K
    W_K_manual = torch.tensor([
        [0.4, 0.3, 0.2],
        [0.2, 0.5, 0.3],
        [0.3, 0.2, 0.6],
    ], dtype=torch.float32)
    
    # W_V
    W_V_manual = torch.tensor([
        [0.6, 0.1, 0.2],
        [0.1, 0.7, 0.1],
        [0.2, 0.2, 0.5],
    ], dtype=torch.float32)
    
    # Копируем веса в модель.
    # nn.Linear вычисляет x @ weight.T, а в уроках формула была Q = x @ W_Q.
    # Поэтому в weight кладём ТРАНСПОНИРОВАННУЮ матрицу, чтобы получить те же числа.
    with torch.no_grad():
        attention.W_query.weight = nn.Parameter(W_Q_manual.t().contiguous())
        attention.W_key.weight = nn.Parameter(W_K_manual.t().contiguous())
        attention.W_value.weight = nn.Parameter(W_V_manual.t().contiguous())
    
    print("\nМатрица W_Q:")
    print(W_Q_manual)
    print("\nМатрица W_K:")
    print(W_K_manual)
    print("\nМатрица W_V:")
    print(W_V_manual)
    
    # ============================================================
    # Запускаем механизм внимания
    # ============================================================
    print("\n" + "=" * 70)
    print("ВЫЧИСЛЕНИЕ МЕХАНИЗМА ВНИМАНИЯ")
    print("=" * 70)
    
    # scale=False - чтобы точно воспроизвести ручной пример из уроков 4-8
    # (там мы для наглядности не делили scores на sqrt(d_k)).
    context, weights, scores = attention(inputs, scale=False)
    
    # ------------------------------------------------------------
    # ШАГ 1: Query, Key, Value
    # ------------------------------------------------------------
    print("\n[ШАГ 1] Query, Key, Value векторы:")
    
    Q = attention.W_query(inputs)
    K = attention.W_key(inputs)
    V = attention.W_value(inputs)
    
    print("\nQuery (Q = X @ W_Q):")
    for i, name in enumerate(token_names):
        values = ", ".join([f"{v:.3f}" for v in Q[i].tolist()])
        print(f"  {name:>8}: [{values}]")
    
    print("\nKey (K = X @ W_K):")
    for i, name in enumerate(token_names):
        values = ", ".join([f"{v:.3f}" for v in K[i].tolist()])
        print(f"  {name:>8}: [{values}]")
    
    print("\nValue (V = X @ W_V):")
    for i, name in enumerate(token_names):
        values = ", ".join([f"{v:.3f}" for v in V[i].tolist()])
        print(f"  {name:>8}: [{values}]")
    
    # ------------------------------------------------------------
    # ШАГ 2: Attention Scores
    # ------------------------------------------------------------
    print("\n" + "-" * 70)
    print("[ШАГ 2] Attention Scores (Q @ K^T)")
    print("-" * 70)
    print("В этом демо scale=False - показываем те же scores, что в ручном примере")
    print("(без деления на sqrt(d_k)). В реальной модели масштабирование включено.")

    print("\nМатрица scores:")
    header = "         " + "".join([f"{name:>10}" for name in token_names])
    print(header)
    for i, name in enumerate(token_names):
        row = f"{name:>8}" + "".join([f"{scores[i,j]:>10.4f}" for j in range(len(token_names))])
        print(row)
    
    # ------------------------------------------------------------
    # ШАГ 3: Attention Weights
    # ------------------------------------------------------------
    print("\n" + "-" * 70)
    print("[ШАГ 3] Attention Weights (softmax)")
    print("-" * 70)
    
    print("\nМатрица weights:")
    print(header)
    for i, name in enumerate(token_names):
        row = f"{name:>8}" + "".join([f"{weights[i,j]:>10.4f}" for j in range(len(token_names))])
        print(row)
    
    print("\nПроверка сумм по строкам:")
    for i, name in enumerate(token_names):
        row_sum = weights[i].sum().item()
        print(f"  {name:>8}: {row_sum:.6f} {'✓' if abs(row_sum - 1.0) < 0.0001 else '✗'}")
    
    # ------------------------------------------------------------
    # ШАГ 4: Context Vectors
    # ------------------------------------------------------------
    print("\n" + "-" * 70)
    print("[ШАГ 4] Context Vectors (weights @ V)")
    print("-" * 70)
    
    print("\nКонтекстные векторы:")
    for i, name in enumerate(token_names):
        values = ", ".join([f"{v:.4f}" for v in context[i].tolist()])
        print(f"  {name:>8}: [{values}]")
    
    # ------------------------------------------------------------
    # Подробный разбор для одного токена
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ПОДРОБНЫЙ РАЗБОР для токена 'cat'")
    print("=" * 70)
    
    mid_idx = 1
    print(f"\nВеса внимания для 'cat':")
    for j, name in enumerate(token_names):
        print(f"  На '{name}': {weights[mid_idx, j]:.4f} ({weights[mid_idx, j]*100:.1f}%)")
    
    print(f"\nФормула контекстного вектора:")
    formula_parts = []
    for j, name in enumerate(token_names):
        formula_parts.append(f"{weights[mid_idx, j]:.3f} × {name}")
    print(f"  z(cat) = " + " + ".join(formula_parts))
    
    # ------------------------------------------------------------
    # Сравнение с версией без весов
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("СРАВНЕНИЕ: С весами vs Без весов")
    print("=" * 70)
    
    from attention_simple import SimpleSelfAttention
    simple_attention = SimpleSelfAttention()
    simple_attention.eval()
    
    # Исправлено: теперь unpack 3 значения (как в SelfAttention)
    context_simple, weights_simple, scores_simple = simple_attention(inputs)
    
    print("\nВеса внимания (диагональ - само-внимание):")
    print(f"  Без весов: The={weights_simple[0,0]:.3f}, cat={weights_simple[1,1]:.3f}, loved={weights_simple[2,2]:.3f}")
    print(f"  С весами:  The={weights[0,0]:.3f}, cat={weights[1,1]:.3f}, loved={weights[2,2]:.3f}")
    
    print("\nВывод: С обучаемыми весами модель может научиться уделять")
    print("       больше внимания ДРУГИМ токенам, а не только себе!")
    
    return {
        'Q': Q,
        'K': K,
        'V': V,
        'scores': scores,
        'weights': weights,
        'context': context
    }


def demo_with_causal_mask():
    """
    Демонстрация каузальной маски для генерации текста.
    """
    print("\n" + "=" * 70)
    print("ДЕМОНСТРАЦИЯ: Каузальная маска")
    print("=" * 70)
    
    seq_length = 4
    mask = CausalMask.create(seq_length)
    
    CausalMask.visualize(mask)
    
    print("\nОбъяснение:")
    print("  - '0' означает: токен виден (прошлое и текущий)")
    print("  - '-inf' означает: токен скрыт (будущее)")
    print("  - После softmax, -inf превращается в 0 (токен не получает внимания)")
    
    print("\nПример для токена 2:")
    print("  Видит токены: 0, 1, 2 (прошлое и текущий)")
    print("  Не видит: 3 (будущее)")


def demo_parameter_count():
    """
    Показывает количество обучаемых параметров.
    """
    print("\n" + "=" * 70)
    print("ДЕМОНСТРАЦИЯ: Количество параметров")
    print("=" * 70)
    
    embed_dims = [16, 64, 128, 256, 768]
    
    print(f"\n{'embed_dim':>10} | {'Параметров':>12} | {'W_Q':>8} | {'W_K':>8} | {'W_V':>8}")
    print("-" * 50)
    
    for dim in embed_dims:
        attention = SelfAttention(embed_dim=dim)
        total = attention.get_num_parameters()
        per_matrix = dim * dim
        print(f"{dim:>10} | {total:>12} | {per_matrix:>8} | {per_matrix:>8} | {per_matrix:>8}")
    
    print("\nФормула: 3 × (embed_dim × embed_dim) = 3 × d²")


if __name__ == "__main__":
    # Запускаем все демонстрации
    demo_with_custom_weights()
    demo_with_causal_mask()
    demo_parameter_count()
    
    print("\n" + "=" * 70)
    print("ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА")
    print("=" * 70)
    print("\nСледующий шаг: Multi-Head Attention (несколько голов внимания)")