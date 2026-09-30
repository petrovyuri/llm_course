import torch
from attention_multihead import MultiHeadAttention, CausalMask


def test_output_shape():
    """Тест 1: Проверка размерности выхода."""
    print("\n" + "=" * 70)
    print("ТЕСТ 1: Размерность выхода")
    print("=" * 70)

    batch_size = 2
    seq_length = 5
    embed_dim = 16
    num_heads = 2

    attention = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
    x = torch.randn(batch_size, seq_length, embed_dim)

    output, weights = attention(x)

    # Проверяем, что выход имеет ту же форму, что и вход
    assert output.shape == (batch_size, seq_length, embed_dim), (
        f"Выход: ожидалось {(batch_size, seq_length, embed_dim)}, получено {output.shape}"
    )

    # Проверяем форму весов внимания
    assert weights.shape == (batch_size, num_heads, seq_length, seq_length), (
        f"Веса: ожидалось {(batch_size, num_heads, seq_length, seq_length)}, получено {weights.shape}"
    )

    print(f"✓ Вход: {x.shape}")
    print(f"✓ Выход: {output.shape}")
    print(f"✓ Веса: {weights.shape}")
    print(f"  ← Размерность выхода совпадает с входом!")
    print("\nТест 1: ПРОЙДЕН ✓")


def test_attention_weights_sum():
    """Тест 2: Сумма весов = 1."""
    print("\n" + "=" * 70)
    print("ТЕСТ 2: Сумма весов внимания = 1")
    print("=" * 70)

    seq_length = 4
    embed_dim = 8
    num_heads = 2

    attention = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
    attention.eval()

    x = torch.randn(1, seq_length, embed_dim)
    _, weights = attention(x)

    # Сумма по каждой строке должна быть 1 (свойство softmax)
    row_sums = weights.sum(dim=-1)

    # Проверяем для всех батчей, голов и строк
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), (
        "Сумма весов должна быть 1.0"
    )


    print(f"✓ Суммы по строкам: {row_sums[0, 0].detach().numpy()}")
    print(f"  ← Все суммы равны 1.0 (проверка softmax)")
    print("\nТест 2: ПРОЙДЕН ✓")


def test_causal_mask_effect():
    """Тест 3: Каузальная маска скрывает будущее."""
    print("\n" + "=" * 70)
    print("ТЕСТ 3: Эффект каузальной маски")
    print("=" * 70)

    seq_length = 4
    embed_dim = 8
    num_heads = 2

    attention = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
    attention.eval()

    x = torch.randn(1, seq_length, embed_dim)
    mask = CausalMask.create(seq_length)

    _, weights_with_mask = attention(x, mask=mask)

    # Проверяем, что будущие токены имеют 0 внимания
    # Для токена 0, токены 1,2,3 должны быть ~0
    future_weights = weights_with_mask[0, 0, 0, 1:]

    assert torch.all(future_weights < 0.01), (
        f"Будущие токены должны иметь 0 внимания, получено {future_weights}"
    )

    print(f"✓ Веса на будущие токены: {future_weights.tolist()}")
    print(f"  ← Будущие токены скрыты (веса ≈ 0)")
    print("\nТест 3: ПРОЙДЕН ✓")


def test_parameter_count():
    """Тест 4: Количество параметров."""
    print("\n" + "=" * 70)
    print("ТЕСТ 4: Количество параметров")
    print("=" * 70)

    embed_dim = 16
    num_heads = 2

    attention = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
    num_params = attention.get_num_parameters()

    # 4 линейных слоя: W_Q, W_K, W_V, W_O
    # Каждый: embed_dim × embed_dim
    expected = 4 * (embed_dim * embed_dim)

    print(f"embed_dim: {embed_dim}")
    print(f"Ожидаемо параметров: {expected}")
    print(f"Получено параметров: {num_params}")
    print(f"  ← Формула: 4 × embed_dim² = 4 × {embed_dim}² = {expected}")

    assert num_params == expected, f"Ожидалось {expected}, получено {num_params}"

    print("\nТест 4: ПРОЙДЕН ✓")


def test_gradient_flow():
    """Тест 5: Протекание градиентов."""
    print("\n" + "=" * 70)
    print("ТЕСТ 5: Протекание градиентов")
    print("=" * 70)

    embed_dim = 8
    num_heads = 2
    seq_length = 3

    attention = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
    attention.train()  # Режим обучения

    x = torch.randn(1, seq_length, embed_dim, requires_grad=True)
    output, _ = attention(x)

    # Создаём фиктивную функцию потерь
    loss = output.sum()

    # Обратное распространение
    loss.backward()

    # Проверяем, что градиенты есть
    assert x.grad is not None, "Градиент входа должен существовать"
    assert attention.W_query.weight.grad is not None, "Градиент W_Q должен существовать"
    assert attention.W_key.weight.grad is not None, "Градиент W_K должен существовать"
    assert attention.W_value.weight.grad is not None, "Градиент W_V должен существовать"
    assert attention.W_output.weight.grad is not None, (
        "Градиент W_O должен существовать"
    )

    print(f"✓ Градиент входа: {x.grad.shape}")
    print(f"✓ Градиент W_Q: {attention.W_query.weight.grad.shape}")
    print(f"✓ Градиент W_K: {attention.W_key.weight.grad.shape}")
    print(f"✓ Градиент W_V: {attention.W_value.weight.grad.shape}")
    print(f"✓ Градиент W_O: {attention.W_output.weight.grad.shape}")
    print(f"  ← Все градиенты протекают корректно!")
    print("\nТест 5: ПРОЙДЕН ✓")


def run_all_tests():
    """Запускает все тесты."""
    print("\n" + "=" * 70)
    print("ЗАПУСК ВСЕХ ТЕСТОВ: Multi-Head Attention")
    print("=" * 70)

    try:
        test_output_shape()
        test_attention_weights_sum()
        test_causal_mask_effect()
        test_parameter_count()
        test_gradient_flow()

        print("\n" + "=" * 70)
        print("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ! 🎉")
        print("=" * 70)
        print("\nMulti-Head Attention работает корректно.")
        print("Готов к интеграции в Transformer Block.")

    except AssertionError as e:
        print(f"\n❌ ТЕСТ ПРОВАЛЕН: {e}")
    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}")


if __name__ == "__main__":
    run_all_tests()

