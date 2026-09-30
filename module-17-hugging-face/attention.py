"""
Модуль механизма самовнимания с обучаемыми весами.
Это полноценная реализация, используемая в трансформерах.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """
    Механизм самовнимания с обучаемыми весами W_Q, W_K, W_V.

    Архитектура:
        Q = x @ W_Q
        K = x @ W_K
        V = x @ W_V
        Attention = softmax(Q @ K^T / sqrt(d_k)) @ V

    Args:
        embed_dim: Размерность входных векторов (d_model)
    """

    def __init__(self, embed_dim):
        super().__init__()

        # Сохраняем размерность
        self.embed_dim = embed_dim

        # ============================================================
        # ОБУЧАЕМЫЕ ВЕСОВЫЕ МАТРИЦЫ
        # ============================================================
        # W_Q, W_K, W_V - матрицы размера [embed_dim, embed_dim]
        # Инициализируем их случайными значениями (будут обучаться)

        self.W_query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_value = nn.Linear(embed_dim, embed_dim, bias=False)

        # Инициализация весов (важно для стабильности обучения)
        self._init_weights()

    def _init_weights(self):
        """
        Инициализация весов матриц.
        Используем нормальное распределение с малым стандартным отклонением.
        """
        for module in [self.W_query, self.W_key, self.W_value]:
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, mask=None, scale=True):
        """
        Вычисляет механизм самовнимания.

        Args:
            x: входной тензор формы [batch_size, seq_length, embed_dim]
               или [seq_length, embed_dim] если нет батча
            mask: опциональная маска для скрытия будущих токенов
                  формы [seq_length, seq_length]

        Returns:
            context_vectors: контекстные векторы той же формы, что и вход
            attention_weights: матрица весов внимания [seq_length, seq_length]
            attention_scores: матрица scores до softmax (для отладки)
        """
        # Если вход без батча, добавляем измерение батча
        if x.dim() == 2:
            x = x.unsqueeze(0)  # [seq_len, embed_dim] → [1, seq_len, embed_dim]
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size, seq_length, embed_dim = x.shape

        # ============================================================
        # ШАГ 1: Вычисляем Query, Key, Value через обучаемые матрицы
        # ============================================================
        # x: [batch, seq_len, embed_dim]
        # W: [embed_dim, embed_dim]
        # Результат: [batch, seq_len, embed_dim]

        Q = self.W_query(x)  # Query
        K = self.W_key(x)  # Key
        V = self.W_value(x)  # Value

        # ============================================================
        # ШАГ 2: Attention Scores = Q @ K^T
        # ============================================================
        # Q: [batch, seq_len, embed_dim]
        # K^T: [batch, embed_dim, seq_len]
        # Результат: [batch, seq_len, seq_len]

        attention_scores = torch.matmul(Q, K.transpose(-2, -1))

        # ============================================================
        # ШАГ 3: Масштабирование (Scaling)
        # ============================================================
        # Делим на sqrt(embed_dim) для стабильности градиентов
        # Это предотвращает слишком большие значения после умножения.
        # scale=False позволяет воспроизвести ручной пример из уроков 4-8,
        # где для наглядности деление на sqrt(d_k) мы опускали.

        if scale:
            scale_factor = torch.sqrt(torch.tensor(embed_dim, dtype=torch.float32))
            attention_scores = attention_scores / scale_factor

        # ============================================================
        # ШАГ 4: Маскирование (опционально)
        # ============================================================
        # Для генерации текста скрываем будущие токены
        # Устанавливаем очень отрицательные значения для маскированных позиций

        if mask is not None:
            # mask: [seq_len, seq_len] с 0 и -inf
            attention_scores = attention_scores + mask

        # ============================================================
        # ШАГ 5: Attention Weights = softmax(scores)
        # ============================================================
        # Нормализуем по последней размерности (по строкам)

        attention_weights = F.softmax(attention_scores, dim=-1)

        # ============================================================
        # ШАГ 6: Context Vectors = weights @ V
        # ============================================================
        # attention_weights: [batch, seq_len, seq_len]
        # V: [batch, seq_len, embed_dim]
        # Результат: [batch, seq_len, embed_dim]

        context_vectors = torch.matmul(attention_weights, V)

        # Убираем измерение батча, если добавляли
        if squeeze_output:
            context_vectors = context_vectors.squeeze(0)
            attention_weights = attention_weights.squeeze(0)
            attention_scores = attention_scores.squeeze(0)

        return context_vectors, attention_weights, attention_scores

    def get_num_parameters(self):
        """
        Возвращает количество обучаемых параметров в этом слое.
        """
        return sum(p.numel() for p in self.parameters())


class CausalMask:
    """
    Утилита для создания каузальной маски (маска будущего).
    Используется для генерации текста, чтобы модель не видела будущие токены.
    """

    @staticmethod
    def create(seq_length, device="cpu"):
        """
        Создаёт маску размера [seq_length, seq_length].

        Возвращает:
            mask: тензор где будущие позиции = -inf, прошлые = 0
        """
        # Создаём матрицу где верхний треугольник = 1, остальное = 0
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
