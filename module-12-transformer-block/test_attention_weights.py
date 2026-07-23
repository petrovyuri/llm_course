# test_attention_weights.py
"""
Тесты для проверки механизма внимания с обучаемыми весами.
"""

import torch
import torch.nn as nn
from attention import SelfAttention, CausalMask


def test_attention_output_shape():
    """
    Тест 1: Проверка размерности выхода.
    """
    print("\n" + "=" * 70)
    print("ТЕСТ 1: Размерность выхода")
    print("=" * 70)

    batch_size = 2
    seq_length = 5
    embed_dim = 16

    attention = SelfAttention(embed_dim=embed_dim)

    # Создаём случайный вход
    x = torch.randn(batch_size, seq_length, embed_dim)

    # Пропускаем через внимание
    context, weights, scores = attention(x)

    # Проверяем размерности
    assert context.shape == (batch_size, seq_length, embed_dim), (
        f"Контекст: ожидалось {(batch_size, seq_length, embed_dim)}, получено {context.shape}"
    )

    assert weights.shape == (batch_size, seq_length, seq_length), (
        f"Веса: ожидалось {(batch_size, seq_length, seq_length)}, получено {weights.shape}"
    )

    assert scores.shape == (batch_size, seq_length, seq_length), (
        f"Scores: ожидалось {(batch_size, seq_length, seq_length)}, получено {scores.shape}"
    )

    print(f"✓ Вход: {x.shape}")
    print(f"✓ Контекст: {context.shape}")
    print(f"✓ Веса: {weights.shape}")
    print(f"✓ Scores: {scores.shape}")
    print("\nТест 1: ПРОЙДЕН ✓")


def test_attention_weights_sum():
    """
    Тест 2: Сумма весов внимания должна быть 1.
    """
    print("\n" + "=" * 70)
    print("ТЕСТ 2: Сумма весов внимания = 1")
    print("=" * 70)

    seq_length = 4
    embed_dim = 8

    attention = SelfAttention(embed_dim=embed_dim)
    attention.eval()

    x = torch.randn(1, seq_length, embed_dim)
    _, weights, _ = attention(x)

    # Сумма по каждой строке должна быть 1
    row_sums = weights.sum(dim=-1)

    print(f"Суммы по строкам: {row_sums[0].tolist()}")

    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), (
        "Сумма весов должна быть 1.0"
    )

    print("✓ Все строки суммируются в 1.0")
    print("\nТест 2: ПРОЙДЕН ✓")


def test_causal_mask():
    """
    Тест 3: Каузальная маска скрывает будущее.
    """
    print("\n" + "=" * 70)
    print("ТЕСТ 3: Каузальная маска")
    print("=" * 70)

    seq_length = 4
    mask = CausalMask.create(seq_length)

    print("Маска:")
    print(mask)

    # Проверяем, что верхний треугольник = -inf
    upper_triangle = torch.triu(mask, diagonal=1)
    assert torch.all(upper_triangle[upper_triangle != 0] == float("-inf")), (
        "Верхний треугольник должен быть -inf"
    )

    # Проверяем, что диагональ и нижний треугольник = 0
    lower_with_diag = torch.tril(mask)
    assert torch.all(lower_with_diag == 0), (
        "Диагональ и нижний треугольник должны быть 0"
    )

    print("✓ Верхний треугольник: -inf (будущее скрыто)")
    print("✓ Диагональ и низ: 0 (прошлое видно)")
    print("\nТест 3: ПРОЙДЕН ✓")


def test_parameter_count():
    """
    Тест 4: Количество обучаемых параметров.
    """
    print("\n" + "=" * 70)
    print("ТЕСТ 4: Количество параметров")
    print("=" * 70)

    embed_dim = 16

    attention = SelfAttention(embed_dim=embed_dim)
    num_params = attention.get_num_parameters()

    expected = 3 * (embed_dim * embed_dim)  # W_Q + W_K + W_V

    print(f"embed_dim: {embed_dim}")
    print(f"Ожидаемо параметров: {expected}")
    print(f"Получено параметров: {num_params}")

    assert num_params == expected, f"Ожидалось {expected}, получено {num_params}"

    print(f"✓ Формула: 3 × {embed_dim}² = {expected}")
    print("\nТест 4: ПРОЙДЕН ✓")


def test_gradient_flow():
    """
    Тест 5: Градиенты протекают через механизм внимания.
    """
    print("\n" + "=" * 70)
    print("ТЕСТ 5: Протекание градиентов")
    print("=" * 70)

    embed_dim = 8
    seq_length = 3

    attention = SelfAttention(embed_dim=embed_dim)
    attention.train()

    x = torch.randn(1, seq_length, embed_dim, requires_grad=True)

    # Пропускаем через внимание
    context, _, _ = attention(x)

    # Создаём фиктивную функцию потерь
    loss = context.sum()

    # Обратное распространение
    loss.backward()

    # Проверяем, что градиенты есть
    assert x.grad is not None, "Градиент входа должен существовать"
    assert x.grad.shape == x.shape, f"Форма градиента: {x.grad.shape}"

    # Проверяем градиенты весов
    assert attention.W_query.weight.grad is not None, "Градиент W_Q должен существовать"
    assert attention.W_key.weight.grad is not None, "Градиент W_K должен существовать"
    assert attention.W_value.weight.grad is not None, "Градиент W_V должен существовать"

    print(f"✓ Градиент входа: {x.grad.shape}")
    print(f"✓ Градиент W_Q: {attention.W_query.weight.grad.shape}")
    print(f"✓ Градиент W_K: {attention.W_key.weight.grad.shape}")
    print(f"✓ Градиент W_V: {attention.W_value.weight.grad.shape}")
    print("\nТест 5: ПРОЙДЕН ✓")


def run_all_tests():
    """
    Запускает все тесты.
    """
    print("\n" + "=" * 70)
    print("ЗАПУСК ВСЕХ ТЕСТОВ: Механизм внимания с обучаемыми весами")
    print("=" * 70)

    try:
        test_attention_output_shape()
        test_attention_weights_sum()
        test_causal_mask()
        test_parameter_count()
        test_gradient_flow()

        print("\n" + "=" * 70)
        print("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ! 🎉")
        print("=" * 70)
        print("\nМеханизм внимания работает корректно.")
        print("Готов к интеграции в полноценную модель трансформера.")

    except AssertionError as e:
        print(f"\n❌ ТЕСТ ПРОВАЛЕН: {e}")
    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}")


if __name__ == "__main__":
    run_all_tests()
