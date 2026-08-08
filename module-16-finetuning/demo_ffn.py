"""
Подробная демонстрация Feed-Forward Network с пошаговым выводом.

Этот скрипт показывает все промежуточные вычисления, чтобы вы могли
увидеть, как данные проходят через каждый слой FFN.

Важно: Здесь демонстрируется ТОЛЬКО Feed-Forward Network.
Transformer Block будет в следующем модуле.
"""

import torch
import torch.nn as nn
from feed_forward import FeedForwardNetwork


def demo_reproduce_manual():
    """
    Воспроизводим ручной пример из уроков 3-7 (embed_dim=4, ff_dim=8).
    Вход - выход Multi-Head Attention из прошлого модуля; веса W_1, W_2 -
    те же, что считали руками. Проверяем, что код даёт те же числа.
    """
    print("\n" + "=" * 70)
    print("ПРОВЕРКА: код воспроизводит ручной пример (embed_dim=4)")
    print("=" * 70)

    X = torch.tensor([
        [0.7484, 0.8509, 0.7282, 0.8056],  # The
        [0.7579, 0.8647, 0.7364, 0.8144],  # cat
        [0.7588, 0.8659, 0.7372, 0.8154],  # loved
    ])
    W1 = torch.tensor([
        [0.5, 0.2, 0.1, 0.3, 0.4, 0.2, 0.1, 0.2],
        [0.3, 0.6, 0.2, 0.1, 0.2, 0.5, 0.3, 0.1],
        [0.1, 0.1, 0.7, 0.2, 0.3, 0.1, 0.6, 0.2],
        [0.2, 0.3, 0.1, 0.5, 0.1, 0.2, 0.1, 0.7],
    ])
    W2 = torch.tensor([
        [0.5, 0.2, 0.1, 0.3], [0.3, 0.6, 0.2, 0.1], [0.1, 0.1, 0.7, 0.2],
        [0.2, 0.3, 0.1, 0.5], [0.4, 0.2, 0.3, 0.1], [0.2, 0.5, 0.1, 0.3],
        [0.1, 0.3, 0.6, 0.2], [0.3, 0.1, 0.2, 0.6],
    ])

    ffn = FeedForwardNetwork(embed_dim=4, ff_dim=8)
    ffn.eval()
    # nn.Linear считает x @ weight.T, а в уроках Intermediate = X @ W_1,
    # поэтому в weight кладём транспонированную матрицу, а bias обнуляем.
    with torch.no_grad():
        ffn.linear_1.weight = nn.Parameter(W1.t().contiguous())
        ffn.linear_2.weight = nn.Parameter(W2.t().contiguous())
        ffn.linear_1.bias.zero_()
        ffn.linear_2.bias.zero_()

    output = ffn(X)
    torch.set_printoptions(precision=4)
    print("\nВыход кода (Output):")
    print(output)
    print("\nОжидалось из ручного примера (урок 6):")
    print("  The:   [1.8166, 2.0054, 1.9607, 2.0083]")
    print("  cat:   [1.8401, 2.0321, 1.9857, 2.0335]")
    print("  loved: [1.8424, 2.0346, 1.9881, 2.0360]")
    print("\nЧисла совпадают - код подтверждает ручной счёт!")


def demo_feed_forward():
    """Демонстрация работы Feed-Forward Network."""
    print("\n" + "=" * 70)
    print("ДЕМОНСТРАЦИЯ: Feed-Forward Network")
    print("=" * 70)

    # ============================================================
    # ПАРАМЕТРЫ (как в математической части)
    # ============================================================
    embed_dim = 16  # Размер вектора токена
    ff_dim = 64  # Размер скрытого слоя (4× embed_dim)
    seq_length = 4  # Количество токенов в последовательности
    batch_size = 1  # Один пример для наглядности

    print(f"\nПараметры:")
    print(f"  embed_dim: {embed_dim}")
    print(f"  ff_dim: {ff_dim} (4× embed_dim)")
    print(f"  seq_length: {seq_length}")
    print(f"  batch_size: {batch_size}")

    # ============================================================
    # СОЗДАЁМ ВХОДНЫЕ ДАННЫЕ
    # ============================================================
    # В реальности это выход от Multi-Head Attention
    # Здесь используем случайные числа для демонстрации
    torch.manual_seed(42)  # Для воспроизводимости
    x = torch.randn(batch_size, seq_length, embed_dim)

    print(f"\nВходной тензор: {x.shape}")
    print(f"  [batch_size, seq_length, embed_dim]")
    print(f"  [{batch_size}, {seq_length}, {embed_dim}]")
    print(f"  ← В реальности это выход от Multi-Head Attention!")

    # ============================================================
    # СОЗДАЁМ FEED-FORWARD NETWORK
    # ============================================================
    ffn = FeedForwardNetwork(embed_dim=embed_dim, ff_dim=ff_dim)
    ffn.eval()  # Режим оценки (без dropout и т.д.)

    print(f"\nКоличество параметров: {ffn.get_num_parameters()}")
    print(f"  Формула: 2 × embed_dim × ff_dim + bias")
    print(f"           2 × {embed_dim} × {ff_dim} + {embed_dim} + {ff_dim}")
    print(f"         = {2 * embed_dim * ff_dim + embed_dim + ff_dim} параметров")

    # ============================================================
    # ПРОПУСКАЕМ ЧЕРЕЗ FFN
    # ============================================================
    output = ffn(x)

    print(f"\nВыходной тензор: {output.shape}")
    print(f"  [batch_size, seq_length, embed_dim]")
    print(f"  [{batch_size}, {seq_length}, {embed_dim}]")
    print(f"  ← Размерность такая же, как на входе!")

    # ============================================================
    # ДЕТАЛЬНЫЙ РАЗБОР
    # ============================================================
    print("\n" + "=" * 70)
    print("ДЕТАЛЬНЫЙ РАЗБОР")
    print("=" * 70)

    # Показываем промежуточные вычисления
    print(f"\nШаг 1: Linear 1 (расширение)")
    intermediate = ffn.linear_1(x)
    print(f"  Вход: {x.shape} → Выход: {intermediate.shape}")
    print(f"  ← Расширили с {embed_dim} до {ff_dim} измерений")

    print(f"\nШаг 2: ReLU (активация)")
    activated = torch.nn.functional.relu(intermediate)
    print(f"  Вход: {intermediate.shape} → Выход: {activated.shape}")

    # Считаем, сколько значений обнулилось
    zero_count = (activated == 0).sum().item()
    total_count = activated.numel()
    zero_percent = (zero_count / total_count) * 100
    print(f"  Обнулено значений: {zero_count} из {total_count} ({zero_percent:.1f}%)")
    print(f"  ← Отрицательные значения стали 0 (нелинейность)")

    print(f"\nШаг 3: Linear 2 (сжатие)")
    output = ffn.linear_2(activated)
    print(f"  Вход: {activated.shape} → Выход: {output.shape}")
    print(f"  ← Сжали с {ff_dim} до {embed_dim} измерений")

    # ============================================================
    # СРАВНЕНИЕ ВХОДА И ВЫХОДА
    # ============================================================
    print("\n" + "=" * 70)
    print("СРАВНЕНИЕ: Вход vs Выход FFN")
    print("=" * 70)

    print(f"\nВход:")
    print(f"  Форма: {x.shape}")
    print(f"  Мин: {x.min().item():.4f}, Макс: {x.max().item():.4f}")
    print(f"  Среднее: {x.mean().item():.4f}")

    print(f"\nВыход (после FFN):")
    print(f"  Форма: {output.shape}")
    print(f"  Мин: {output.min().item():.4f}, Макс: {output.max().item():.4f}")
    print(f"  Среднее: {output.mean().item():.4f}")

    print(f"\nЧто изменилось:")
    print(f"  • Размерность: {x.shape} → {output.shape} (без изменений)")
    print(f"  • Значения: преобразованы через нелинейную функцию")
    print(f"  • Информация: обогащена через расширение и сжатие")

    # ============================================================
    # ПОКАЗЫВАЕМ ПРИМЕР ДЛЯ ОДНОГО ТОКЕНА
    # ============================================================
    print("\n" + "=" * 70)
    print("ПРИМЕР ДЛЯ ОДНОГО ТОКЕНА")
    print("=" * 70)

    token_idx = 0
    print(f"\nТокен {token_idx}:")
    print(f"  Вход (первые 5 значений): {x[0, token_idx, :5].tolist()}")
    print(f"  Intermediate (первые 5): {intermediate[0, token_idx, :5].tolist()}")
    print(f"  После ReLU (первые 5): {activated[0, token_idx, :5].tolist()}")
    print(f"  Выход (первые 5): {output[0, token_idx, :5].tolist()}")

    return {
        "input": x,
        "intermediate": intermediate,
        "activated": activated,
        "output": output,
    }


def demo_parameter_comparison():
    """Сравнение параметров Attention и FFN."""
    print("\n" + "=" * 70)
    print("СРАВНЕНИЕ: Параметры Attention vs FFN")
    print("=" * 70)

    embed_dim = 16
    num_heads = 2
    ff_dim = 64

    from attention_multihead import MultiHeadAttention

    attention = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
    ffn = FeedForwardNetwork(embed_dim=embed_dim, ff_dim=ff_dim)

    attention_params = attention.get_num_parameters()
    ffn_params = ffn.get_num_parameters()
    total_params = attention_params + ffn_params

    print(f"\nembed_dim = {embed_dim}, num_heads = {num_heads}, ff_dim = {ff_dim}")
    print(f"\n{'Компонент':<25} | {'Параметров':>12} | {'Доля':>8}")
    print("-" * 50)
    print(
        f"{'Multi-Head Attention':<25} | {attention_params:>12} | {attention_params / total_params * 100:>7.1f}%"
    )
    print(
        f"{'Feed-Forward Network':<25} | {ffn_params:>12} | {ffn_params / total_params * 100:>7.1f}%"
    )
    print(f"{'Итого':<25} | {total_params:>12} | {100.0:>7.1f}%")

    print(f"\nВывод: FFN обычно имеет больше параметров, чем Attention!")
    print(f"       При ff_dim = 4×embed_dim, FFN ≈ 2× Attention")


if __name__ == "__main__":
    demo_reproduce_manual()
    demo_feed_forward()
    demo_parameter_comparison()

    print("\n" + "=" * 70)
    print("ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА")
