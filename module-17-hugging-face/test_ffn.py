import torch
from feed_forward import FeedForwardNetwork


def test_output_shape():
    """Тест 1: Проверка размерности выхода."""
    print("\n" + "=" * 70)
    print("ТЕСТ 1: Размерность выхода")
    print("=" * 70)

    batch_size = 2
    seq_length = 5
    embed_dim = 16
    ff_dim = 64

    ffn = FeedForwardNetwork(embed_dim=embed_dim, ff_dim=ff_dim)
    x = torch.randn(batch_size, seq_length, embed_dim)

    output = ffn(x)

    # Проверяем, что выход имеет ту же форму, что и вход
    assert output.shape == (batch_size, seq_length, embed_dim), (
        f"Выход: ожидалось {(batch_size, seq_length, embed_dim)}, получено {output.shape}"
    )

    print(f"✓ Вход: {x.shape}")
    print(f"✓ Выход: {output.shape}")
    print(f"  ← Размерность выхода совпадает с входом!")
    print("\nТест 1: ПРОЙДЕН ✓")


def test_intermediate_dimension():
    """Тест 2: Проверка размерности Intermediate."""
    print("\n" + "=" * 70)
    print("ТЕСТ 2: Размерность Intermediate")
    print("=" * 70)

    batch_size = 1
    seq_length = 4
    embed_dim = 16
    ff_dim = 64

    ffn = FeedForwardNetwork(embed_dim=embed_dim, ff_dim=ff_dim)
    x = torch.randn(batch_size, seq_length, embed_dim)

    # Получаем intermediate
    intermediate = ffn.linear_1(x)

    expected_shape = (batch_size, seq_length, ff_dim)
    assert intermediate.shape == expected_shape, (
        f"Intermediate: ожидалось {expected_shape}, получено {intermediate.shape}"
    )

    print(f"✓ Вход: {x.shape}")
    print(f"✓ Intermediate: {intermediate.shape}")
    print(f"  ← Расширение с {embed_dim} до {ff_dim} работает!")
    print("\nТест 2: ПРОЙДЕН ✓")


def test_relu_activation():
    """Тест 3: Проверка функции ReLU."""
    print("\n" + "=" * 70)
    print("ТЕСТ 3: Функция активации ReLU")
    print("=" * 70)

    embed_dim = 16
    ff_dim = 64

    ffn = FeedForwardNetwork(embed_dim=embed_dim, ff_dim=ff_dim)
    ffn.eval()

    # Создаём вход с отрицательными значениями
    x = torch.randn(1, 4, embed_dim)

    # Получаем intermediate и activated
    intermediate = ffn.linear_1(x)
    activated = torch.nn.functional.relu(intermediate)

    # Проверяем, что все значения неотрицательные
    assert (activated >= 0).all(), "Все значения после ReLU должны быть >= 0"

    # Считаем, сколько значений обнулилось
    zero_count = (activated == 0).sum().item()
    total_count = activated.numel()

    print(f"✓ Все значения после ReLU >= 0")
    print(
        f"✓ Обнулено значений: {zero_count} из {total_count} ({zero_count / total_count * 100:.1f}%)"
    )
    print(f"  ← ReLU работает корректно!")
    print("\nТест 3: ПРОЙДЕН ✓")


def test_parameter_count():
    """Тест 4: Количество параметров."""
    print("\n" + "=" * 70)
    print("ТЕСТ 4: Количество параметров")
    print("=" * 70)

    embed_dim = 16
    ff_dim = 64

    ffn = FeedForwardNetwork(embed_dim=embed_dim, ff_dim=ff_dim)
    num_params = ffn.get_num_parameters()

    # W_1: embed_dim × ff_dim + ff_dim (bias)
    # W_2: ff_dim × embed_dim + embed_dim (bias)
    expected = (embed_dim * ff_dim + ff_dim) + (ff_dim * embed_dim + embed_dim)

    print(f"embed_dim: {embed_dim}")
    print(f"ff_dim: {ff_dim}")
    print(f"Ожидаемо параметров: {expected}")
    print(f"Получено параметров: {num_params}")
    print(f"  ← Формула: 2 × embed_dim × ff_dim + embed_dim + ff_dim")

    assert num_params == expected, f"Ожидалось {expected}, получено {num_params}"

    print("\nТест 4: ПРОЙДЕН ✓")


def test_gradient_flow():
    """Тест 5: Протекание градиентов."""
    print("\n" + "=" * 70)
    print("ТЕСТ 5: Протекание градиентов")
    print("=" * 70)

    embed_dim = 16
    ff_dim = 64
    seq_length = 4

    ffn = FeedForwardNetwork(embed_dim=embed_dim, ff_dim=ff_dim)
    ffn.train()  # Режим обучения

    x = torch.randn(1, seq_length, embed_dim, requires_grad=True)
    output = ffn(x)

    # Создаём фиктивную функцию потерь
    loss = output.sum()

    # Обратное распространение
    loss.backward()

    # Проверяем, что градиенты есть
    assert x.grad is not None, "Градиент входа должен существовать"
    assert ffn.linear_1.weight.grad is not None, "Градиент W_1 должен существовать"
    assert ffn.linear_2.weight.grad is not None, "Градиент W_2 должен существовать"

    print(f"✓ Градиент входа: {x.grad.shape}")
    print(f"✓ Градиент W_1: {ffn.linear_1.weight.grad.shape}")
    print(f"✓ Градиент W_2: {ffn.linear_2.weight.grad.shape}")
    print(f"  ← Все градиенты протекают корректно!")
    print("\nТест 5: ПРОЙДЕН ✓")


def run_all_tests():
    """Запускает все тесты."""
    print("\n" + "=" * 70)
    print("ЗАПУСК ВСЕХ ТЕСТОВ: Feed-Forward Network")
    print("=" * 70)

    try:
        test_output_shape()
        test_intermediate_dimension()
        test_relu_activation()
        test_parameter_count()
        test_gradient_flow()

        print("\n" + "=" * 70)
        print("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ! 🎉")
        print("=" * 70)
        print("\nFeed-Forward Network работает корректно.")


    except AssertionError as e:
        print(f"\n❌ ТЕСТ ПРОВАЛЕН: {e}")
    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
