import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention механизм.

    Архитектура (как в математической части):
        1. Для каждой головы: Q = X @ W_Q, K = X @ W_K, V = X @ W_V
        2. Attention = softmax(Q @ K^T / sqrt(d_k)) @ V
        3. Concat всех голов
        4. Output = Concat @ W_O

    Args:
        embed_dim: Размерность входных векторов (d_model)
                   В нашем примере: 16
        num_heads: Количество голов внимания
                   В нашем примере: 2
    """

    def __init__(self, embed_dim, num_heads=2):
        super().__init__()

        # ============================================================
        # СОХРАНЯЕМ ПАРАМЕТРЫ
        # ============================================================
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # ============================================================
        # ПРОВЕРКА: embed_dim должен делиться на num_heads
        # ============================================================
        # Это критически важно! Мы физически разделяем вектор на равные части.
        # Если embed_dim=16 и num_heads=2, то head_dim=8 (делится без остатка)
        # Если embed_dim=17 и num_heads=2, то head_dim=8.5 (ошибка!)
        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) должен делиться на num_heads ({num_heads}) без остатка"
        )

        # Вычисляем размерность каждой головы
        # В нашем примере: 16 / 2 = 8
        self.head_dim = embed_dim // num_heads

        # ============================================================
        # ВЕСОВЫЕ МАТРИЦЫ ДЛЯ ВСЕХ ГОЛОВ
        # ============================================================
        # Важное замечание по реализации:
        # Вместо того чтобы создавать отдельные матрицы для каждой головы
        # (W_Q^0, W_Q^1, W_Q^2...), мы создаём ОДНУ большую матрицу и
        # разделяем её при вычислении. Это более эффективно для GPU.
        #
        # Математически это эквивалентно:
        #   Было бы: num_heads отдельных матриц [embed_dim, head_dim]
        #   Стало:   одна матрица [embed_dim, embed_dim]
        #
        # Почему? Потому что:
        #   num_heads × head_dim = num_heads × (embed_dim / num_heads) = embed_dim

        # W_Q: [embed_dim, embed_dim]
        # После разделения: num_heads матриц [embed_dim, head_dim]
        self.W_query = nn.Linear(embed_dim, embed_dim, bias=False)

        # W_K: [embed_dim, embed_dim]
        self.W_key = nn.Linear(embed_dim, embed_dim, bias=False)

        # W_V: [embed_dim, embed_dim]
        self.W_value = nn.Linear(embed_dim, embed_dim, bias=False)

        # ============================================================
        # W_O: ВЫХОДНАЯ ПРОЕКЦИЯ (НОВОЕ по сравнению с Single-Head!)
        # ============================================================
        # В главе 4 (Single-Head) у нас не было этого слоя.
        # В Multi-Head он обязателен, потому что:
        #   1. Нужно смешать информацию от всех голов
        #   2. Нужно спроецировать результат обратно в embed_dim
        #   3. Модель учится оптимально комбинировать результаты голов
        self.W_output = nn.Linear(embed_dim, embed_dim, bias=False)

        # Инициализация весов (как в главе 4)
        self._init_weights()

    def _init_weights(self):
        """
        Инициализация весов нормальным распределением.

        Это важно для стабильности обучения. Если инициализировать
        веса нулями или слишком большими значениями, градиенты могут
        исчезнуть или «взорваться».
        """
        for module in [self.W_query, self.W_key, self.W_value, self.W_output]:
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _split_heads(self, x, batch_size):
        """
        Разделяет последние измерение на num_heads × head_dim.

        Это ключевая функция для Multi-Head Attention!

        Вход:  [batch_size, seq_length, embed_dim]
               Пример: [2, 10, 16] (2 примера, 10 токенов, 16 измерений)

        Выход: [batch_size, num_heads, seq_length, head_dim]
               Пример: [2, 2, 10, 8] (2 головы, по 8 измерений каждая)

        Что происходит:
          1. Разбиваем embed_dim на num_heads × head_dim
          2. Перемещаем num_heads в новое измерение
          3. Теперь каждая голова работает независимо!

        Аналогия:
          Было: один вектор из 16 чисел
          Стало: два вектора по 8 чисел (для 2 голов)
        """
        # x.shape: [batch, seq_len, embed_dim]
        # Пример: [2, 10, 16]

        # Шаг 1: Разделяем последнее измерение
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        # x.shape: [batch, seq_len, num_heads, head_dim]
        # Пример: [2, 10, 2, 8]

        # Шаг 2: Перемещаем num_heads на второе место
        # Это нужно для удобного матричного умножения дальше
        x = x.transpose(1, 2)
        # x.shape: [batch, num_heads, seq_len, head_dim]
        # Пример: [2, 2, 10, 8]

        return x

    def _combine_heads(self, x, batch_size):
        """
        Объединяет головы обратно в embed_dim.

        Это обратная операция к _split_heads.

        Вход:  [batch_size, num_heads, seq_length, head_dim]
               Пример: [2, 2, 10, 8]

        Выход: [batch_size, seq_length, embed_dim]
               Пример: [2, 10, 16]

        Что происходит:
          1. Возвращаем num_heads на своё место
          2. Объединяем head_dim × num_heads = embed_dim
        """
        # x.shape: [batch, num_heads, seq_len, head_dim]
        # Пример: [2, 2, 10, 8]

        # Шаг 1: Возвращаем num_heads на третье место
        x = x.transpose(1, 2)
        # x.shape: [batch, seq_len, num_heads, head_dim]
        # Пример: [2, 10, 2, 8]

        # Шаг 2: Объединяем последние два измерения
        # num_heads × head_dim = 2 × 8 = 16 = embed_dim
        x = x.contiguous().view(batch_size, -1, self.embed_dim)
        # x.shape: [batch, seq_len, embed_dim]
        # Пример: [2, 10, 16]

        return x

    def forward(self, x, mask=None):
        """
        Вычисляет Multi-Head Attention.

        Это главный метод, который реализует всю математику из главы 5.

        Args:
            x: входной тензор формы [batch_size, seq_length, embed_dim]
               или [seq_length, embed_dim] если нет батча
               Пример: [2, 10, 16] (2 примера, 10 токенов, 16 измерений)

            mask: опциональная каузальная маска формы [seq_length, seq_length]
                  Используется для генерации текста, чтобы скрыть будущие токены
                  Пример: [10, 10] (маска для 10 токенов)

        Returns:
            output: выходной тензор формы [batch_size, seq_length, embed_dim]
                    Пример: [2, 10, 16] (такая же форма, как на входе!)

            attention_weights: веса внимания для визуализации
                               формы [batch_size, num_heads, seq_length, seq_length]
                               Пример: [2, 2, 10, 10]

        Шаги (как в математической части главы 5):
          1. Q = X @ W_Q, K = X @ W_K, V = X @ W_V
          2. Разделить на головы
          3. Scores = Q @ K^T / sqrt(head_dim)
          4. Weights = softmax(Scores)
          5. Context = Weights @ V
          6. Объединить головы
          7. Output = Concat @ W_O
        """
        # ============================================================
        # ПОДГОТОВКА: Обрабатываем вход без батча
        # ============================================================
        # Если вход без батча (например, [10, 16]), добавляем измерение батча
        # Это нужно для единообразия вычислений
        if x.dim() == 2:
            x = x.unsqueeze(0)  # [seq_len, embed_dim] → [1, seq_len, embed_dim]
            squeeze_output = True  # Запоминаем, что нужно убрать батч в конце
        else:
            squeeze_output = False

        batch_size, seq_length, embed_dim = x.shape
        # Пример: batch_size=2, seq_length=10, embed_dim=16

        # ============================================================
        # ШАГ 1: Вычисляем Q, K, V через линейные слои
        # ============================================================
        # Формула: Q = X @ W_Q
        # В коде: nn.Linear делает это автоматически
        #
        # x: [batch, seq_len, embed_dim] = [2, 10, 16]
        # W_Q: [embed_dim, embed_dim] = [16, 16]
        # Q: [batch, seq_len, embed_dim] = [2, 10, 16]

        Q = self.W_query(x)  # Query
        K = self.W_key(x)  # Key
        V = self.W_value(x)  # Value

        # ============================================================
        # ШАГ 2: Разделяем на головы
        # ============================================================
        # Это ключевое отличие от Single-Head!
        #
        # Q: [batch, seq_len, embed_dim] = [2, 10, 16]
        # После split: [batch, num_heads, seq_len, head_dim] = [2, 2, 10, 8]

        Q_heads = self._split_heads(Q, batch_size)
        K_heads = self._split_heads(K, batch_size)
        V_heads = self._split_heads(V, batch_size)

        # ============================================================
        # ШАГ 3: Attention для каждой головы (векторизовано)
        # ============================================================
        # Формула: Scores = Q @ K^T / sqrt(head_dim)
        #
        # Q_heads: [batch, num_heads, seq_len, head_dim] = [2, 2, 10, 8]
        # K_heads^T: [batch, num_heads, head_dim, seq_len] = [2, 2, 8, 10]
        # Scores: [batch, num_heads, seq_len, seq_len] = [2, 2, 10, 10]
        #
        # Важно: Все головы вычисляются ПАРАЛЛЕЛЬНО в одной операции!

        attention_scores = torch.matmul(Q_heads, K_heads.transpose(-2, -1))

        # ============================================================
        # ШАГ 3.1: Масштабирование
        # ============================================================
        # Делим на sqrt(head_dim) для стабильности градиентов
        #
        # В нашем примере: sqrt(8) = 2.828
        # В GPT-3 (175B): sqrt(128) = 11.31
        #
        # Без этого softmax будет иметь очень маленькие градиенты
        # при большой размерности (проблема «исчезающих градиентов»)

        scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attention_scores = attention_scores / scale

        # ============================================================
        # ШАГ 3.2: Маскирование (если есть маска)
        # ============================================================
        # Для генерации текста скрываем будущие токены
        # Устанавливаем очень отрицательные значения для маскированных позиций
        #
        # После softmax, -inf превратится в 0 (токен не получит внимания)

        if mask is not None:
            # mask: [seq_len, seq_len]
            # Добавляем измерения для broadcast: [1, 1, seq_len, seq_len]
            attention_scores = attention_scores + mask

        # ============================================================
        # ШАГ 4: Attention Weights = softmax(scores)
        # ============================================================
        # Нормализуем по последней размерности (по строкам)
        #
        # attention_scores: [2, 2, 10, 10]
        # attention_weights: [2, 2, 10, 10]
        #
        # Сумма по каждой строке = 1.0 (проверка в тестах)

        attention_weights = F.softmax(attention_scores, dim=-1)

        # ============================================================
        # ШАГ 5: Context Vectors = weights @ V
        # ============================================================
        # Формула: Context = Weights @ V
        #
        # attention_weights: [batch, num_heads, seq_len, seq_len] = [2, 2, 10, 10]
        # V_heads: [batch, num_heads, seq_len, head_dim] = [2, 2, 10, 8]
        # context_heads: [batch, num_heads, seq_len, head_dim] = [2, 2, 10, 8]

        context_heads = torch.matmul(attention_weights, V_heads)

        # ============================================================
        # ШАГ 6: Объединяем головы
        # ============================================================
        # Это обратная операция к разделению
        #
        # context_heads: [2, 2, 10, 8]
        # context_combined: [2, 10, 16]

        context_combined = self._combine_heads(context_heads, batch_size)

        # ============================================================
        # ШАГ 7: Выходная проекция (W_O)
        # ============================================================
        # НОВОЕ по сравнению с Single-Head!
        #
        # context_combined: [batch, seq_len, embed_dim] = [2, 10, 16]
        # W_output: [embed_dim, embed_dim] = [16, 16]
        # output: [batch, seq_len, embed_dim] = [2, 10, 16]

        output = self.W_output(context_combined)

        # ============================================================
        # ЗАВЕРШЕНИЕ: Убираем измерение батча, если добавляли
        # ============================================================
        if squeeze_output:
            output = output.squeeze(0)
            attention_weights = attention_weights.squeeze(0)

        return output, attention_weights

    def get_num_parameters(self):
        """
        Возвращает количество обучаемых параметров в этом слое.

        Полезно для понимания масштаба модели.

        Формула:
          4 линейных слоя (W_Q, W_K, W_V, W_O)
          Каждый: embed_dim × embed_dim
          Итого: 4 × embed_dim²

        В нашем примере (embed_dim=16):
          4 × 16² = 4 × 256 = 1024 параметра

        В GPT-3 (175B, embed_dim=12288):
          4 × 12288² = 604 миллиона параметров (только в одном блоке!)
        """
        return sum(p.numel() for p in self.parameters())


class CausalMask:
    """
    Утилита для создания каузальной маски (маска будущего).

    Используется для генерации текста, чтобы модель не видела
    будущие токены. Это критически важно для авторегрессивной генерации!

    Без маски:
      Токен 2 видит токены: 0, 1, 2, 3, 4 (включая будущие!)

    С маской:
      Токен 2 видит токены: 0, 1, 2 (только прошлые и текущий)
    """

    @staticmethod
    def create(seq_length, device="cpu"):
        """
        Создаёт маску размера [seq_length, seq_length].

        Возвращает:
            mask: тензор где будущие позиции = -inf, прошлые = 0

        Пример для seq_length=4:
            [[0, -inf, -inf, -inf],
             [0,    0, -inf, -inf],
             [0,    0,    0, -inf],
             [0,    0,    0,    0]]
        """
        # Создаём матрицу где верхний треугольник = 1, остальное = 0
        # torch.triu: upper triangular (верхний треугольник)
        # diagonal=1: начиная со следующей диагонали (исключаем главную)
        mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1)

        # Инвертируем: прошлые = 0, будущие = 1
        mask = mask.masked_fill(mask == 1, float("-inf"))
        mask = mask.masked_fill(mask == 0, 0.0)

        return mask.to(device)

    @staticmethod
    def visualize(mask):
        """
        Визуализирует маску в читаемом виде.
        """
        print("\nКаузальная маска:")
        seq_length = mask.shape[0]

        # Заголовок
        header = "         " + "".join([f"{j:>8}" for j in range(seq_length)])
        print(header)

        # Строки
        for i in range(seq_length):
            row = f"{i:>8}"
            for j in range(seq_length):
                if mask[i, j] == 0:
                    row += f"{'0':>8}"
                else:
                    row += f"{'-inf':>8}"
            print(row)
