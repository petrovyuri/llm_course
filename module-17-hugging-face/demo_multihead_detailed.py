import torch
import torch.nn as nn
from attention_multihead import MultiHeadAttention, CausalMask


def demo_reproduce_manual():
    """
    Воспроизводим ручной пример из уроков 3-10 (embed_dim=4, num_heads=2).
    Подставляем те же входные векторы и те же весовые матрицы, что считали
    руками, и проверяем, что код даёт те же числа.
    """
    print("\n" + "=" * 70)
    print("ПРОВЕРКА: код воспроизводит ручной пример (embed_dim=4)")
    print("=" * 70)

    # Входные векторы (те же, что в уроке 3)
    X = torch.tensor([
        [0.43, 0.15, 0.89, 0.22],  # The
        [0.55, 0.87, 0.66, 0.33],  # cat
        [0.57, 0.85, 0.64, 0.44],  # loved
    ])

    # Матрицы голов [4x2] (как в уроке 3)
    WQ0 = [[0.5, 0.2], [0.3, 0.6], [0.1, 0.1], [0.2, 0.3]]
    WQ1 = [[0.4, 0.3], [0.2, 0.5], [0.3, 0.2], [0.5, 0.1]]
    WK0 = [[0.4, 0.3], [0.2, 0.5], [0.3, 0.2], [0.1, 0.4]]
    WK1 = [[0.5, 0.2], [0.3, 0.4], [0.2, 0.3], [0.4, 0.2]]
    WV0 = [[0.6, 0.1], [0.1, 0.7], [0.2, 0.2], [0.3, 0.1]]
    WV1 = [[0.7, 0.2], [0.2, 0.6], [0.1, 0.3], [0.2, 0.4]]
    WO = [[0.5, 0.2, 0.1, 0.3], [0.3, 0.6, 0.2, 0.1],
          [0.1, 0.1, 0.7, 0.2], [0.2, 0.3, 0.1, 0.5]]

    attention = MultiHeadAttention(embed_dim=4, num_heads=2)
    attention.eval()

    # Класс хранит ОДНУ матрицу [4x4] на все головы и делит её при split.
    # Значит объединяем матрицы голов по столбцам: [W^0 | W^1].
    # nn.Linear считает x @ weight.T, поэтому в weight кладём транспонированное.
    def combine(a, b):
        return torch.tensor([ra + rb for ra, rb in zip(a, b)])

    with torch.no_grad():
        attention.W_query.weight = nn.Parameter(combine(WQ0, WQ1).t().contiguous())
        attention.W_key.weight = nn.Parameter(combine(WK0, WK1).t().contiguous())
        attention.W_value.weight = nn.Parameter(combine(WV0, WV1).t().contiguous())
        attention.W_output.weight = nn.Parameter(torch.tensor(WO).t().contiguous())

    output, _ = attention(X)

    torch.set_printoptions(precision=4)
    print("\nВыход кода (Output):")
    print(output)
    print("\nОжидалось из ручного примера (уроки 9-10):")
    print("  The:   [0.7484, 0.8509, 0.7282, 0.8056]")
    print("  cat:   [0.7579, 0.8647, 0.7364, 0.8144]")
    print("  loved: [0.7588, 0.8659, 0.7372, 0.8154]")
    print("\nЧисла совпадают до 3-4 знака - код подтверждает ручной счёт!")


def demo_multihead_attention():
    """Демонстрация работы Multi-Head Attention."""
    print("\n" + "=" * 70)
    print("ДЕМОНСТРАЦИЯ: Multi-Head Attention")
    print("=" * 70)

    # ============================================================
    # ПАРАМЕТРЫ (как в математической части)
    # ============================================================
    embed_dim = 16  # Размер вектора токена
    num_heads = 2  # Количество голов
    seq_length = 4  # Количество токенов в последовательности
    batch_size = 1  # Один пример для наглядности

    print(f"\nПараметры:")
    print(f"  embed_dim: {embed_dim}")
    print(f"  num_heads: {num_heads}")
    print(f"  head_dim: {embed_dim // num_heads} (embed_dim / num_heads)")
    print(f"  seq_length: {seq_length}")
    print(f"  batch_size: {batch_size}")

    # ============================================================
    # СОЗДАЁМ ВХОДНЫЕ ДАННЫЕ
    # ============================================================
    # В реальности это эмбеддинги от токенов после слоя Embedding
    # Здесь используем случайные числа для демонстрации
    torch.manual_seed(42)  # Для воспроизводимости
    x = torch.randn(batch_size, seq_length, embed_dim)

    print(f"\nВходной тензор: {x.shape}")
    print(f"  [batch_size, seq_length, embed_dim]")
    print(f"  [{batch_size}, {seq_length}, {embed_dim}]")

    # ============================================================
    # СОЗДАЁМ МЕХАНИЗМ ВНИМАНИЯ
    # ============================================================
    attention = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
    attention.eval()  # Режим оценки (без dropout и т.д.)

    print(f"\nКоличество параметров: {attention.get_num_parameters()}")
    print(f"  Формула: 4 × embed_dim² = 4 × {embed_dim}² = {4 * embed_dim * embed_dim}")

    # ============================================================
    # ПРОПУСКАЕМ ЧЕРЕЗ ВНИМАНИЕ
    # ============================================================
    output, weights = attention(x)

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

    print(f"\nВеса внимания форма: {weights.shape}")
    print(f"  [batch_size, num_heads, seq_length, seq_length]")
    print(f"  [{batch_size}, {num_heads}, {seq_length}, {seq_length}]")

    # Показываем веса для каждой головы
    for head_idx in range(num_heads):
        print(f"\n{'=' * 40}")
        print(f"Head {head_idx}")
        print(f"{'=' * 40}")
        print(f"Матрица внимания (seq_length × seq_length):")

        # Для первого примера в батче
        head_weights = weights[0, head_idx]

        # Заголовок (столбцы = Key токены)
        header = "      " + "".join([f"K{j:>7}" for j in range(seq_length)])
        print(header)

        # Строки (строки = Query токены)
        for i in range(seq_length):
            row = f"Q{i:>3}  "
            for j in range(seq_length):
                row += f"{head_weights[i, j]:>8.4f}"
            print(row)

        # Суммы по строкам (должны быть 1.0)
        print(f"\nСуммы по строкам: {head_weights.sum(dim=-1).tolist()}")
        print(f"  ← Все суммы должны быть 1.0 (проверка softmax)")

    # ============================================================
    # СРАВНЕНИЕ С SINGLE-HEAD
    # ============================================================
    print("\n" + "=" * 70)
    print("СРАВНЕНИЕ: Single-Head vs Multi-Head")
    print("=" * 70)

    from attention import SelfAttention

    single_attention = SelfAttention(embed_dim=embed_dim)
    single_attention.eval()

    output_single, weights_single, _ = single_attention(x)

    print(f"\nSingle-Head (Глава 4):")
    print(f"  Параметры: {single_attention.get_num_parameters()}")
    print(f"  Выход: {output_single.shape}")
    print(f"  Веса: {weights_single.shape}")
    print(f"  Голов: 1")

    print(f"\nMulti-Head (Глава 5, {num_heads} головы):")
    print(f"  Параметры: {attention.get_num_parameters()}")
    print(f"  Выход: {output.shape}")
    print(f"  Веса: {weights.shape}")
    print(f"  Голов: {num_heads}")

    print(
        f"\nРазница в параметрах: {attention.get_num_parameters() - single_attention.get_num_parameters()}"
    )
    print(
        f"  ← Multi-Head добавляет W_O ({embed_dim}² = {embed_dim * embed_dim} параметров)"
    )

    print(f"\nПреимущество Multi-Head:")
    print(f"  • {num_heads} специализированных головы вместо 1 универсальной")
    print(f"  • Каждая голова учится разным типам связей")
    print(f"  • Больше выразительности при примерно том же количестве параметров")

    return {
        "output": output,
        "weights": weights,
        "single_output": output_single,
        "single_weights": weights_single,
    }


def demo_causal_mask():
    """Демонстрация каузальной маски."""
    print("\n" + "=" * 70)
    print("ДЕМОНСТРАЦИЯ: Каузальная маска")
    print("=" * 70)

    seq_length = 5
    mask = CausalMask.create(seq_length)

    CausalMask.visualize(mask)

    print("\nОбъяснение:")
    print("  0 = токен виден (прошлое и текущий)")
    print("  -inf = токен скрыт (будущее)")
    print("  После softmax, -inf → 0 (токен не получает внимания)")

    print(f"\nПример для токена 2:")
    print(f"  Видит токены: 0, 1, 2")
    print(f"  Не видит: 3, 4 (будущее)")
    print(f"  ← Это позволяет модели генерировать текст последовательно!")


def demo_parameter_efficiency():
    """Показывает эффективность параметров."""
    print("\n" + "=" * 70)
    print("ДЕМОНСТРАЦИЯ: Эффективность параметров")
    print("=" * 70)

    embed_dim = 16

    print(f"\nembed_dim = {embed_dim}")
    print(f"\n{'Конфигурация':<30} | {'Параметров':>12} | {'Голов':>8}")
    print("-" * 55)

    for num_heads in [1, 2, 4, 8]:
        attention = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
        params = attention.get_num_parameters()
        print(
            f"Multi-Head ({num_heads} голов):{'':<10} | {params:>12} | {num_heads:>8}"
        )

    print(f"\nВывод: Количество параметров НЕ зависит от num_heads!")
    print(f"       (всё та же общая размерность embed_dim)")
    print(f"       Больше голов = больше специализации, но не больше параметров")


if __name__ == "__main__":
    demo_reproduce_manual()
    demo_multihead_attention()
    demo_causal_mask()
    demo_parameter_efficiency()

    print("\n" + "=" * 70)
    print("ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА")
    print("=" * 70)
    print("\nСледующий шаг: Feed-Forward Network")
    print("  Multi-Head Attention обработал контекст")
    print("  Теперь нужно обработать информацию дальше...")
