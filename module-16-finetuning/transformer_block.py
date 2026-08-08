import torch
import torch.nn as nn

from attention_multihead import MultiHeadAttention
from feed_forward import FeedForwardNetwork


class TransformerBlock(nn.Module):
    """
    Один Transformer block в варианте Pre-LN.

    Схема следующая:
        x
        LayerNorm
        Multi-Head Self-Attention
        Residual Add
        LayerNorm
        Feed-Forward Network
        Residual Add

    Вход:
        x: [batch_size, seq_length, embed_dim]
        mask: [seq_length, seq_length] или None

    Выход:
        out: [batch_size, seq_length, embed_dim]
    """

    def __init__(self, embed_dim, num_heads=2, ff_dim=None, dropout=0.0):
        super().__init__()

        if ff_dim is None:
            ff_dim = 4 * embed_dim

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim

        # Нормализация перед attention
        self.norm_1 = nn.LayerNorm(embed_dim)

        # Механизм внимания из предыдущей главы
        self.attention = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads
        )

        # Нормализация перед FFN
        self.norm_2 = nn.LayerNorm(embed_dim)

        # FFN из предыдущей главы
        self.ffn = FeedForwardNetwork(
            embed_dim=embed_dim,
            ff_dim=ff_dim
        )

        # Dropout пока можно оставить нулевым,
        # но архитектурно лучше сразу заложить место под него
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, return_attention=False):
        """
        Прямой проход через Transformer block.

        Args:
            x: [batch_size, seq_length, embed_dim]
            mask: [seq_length, seq_length] или None
            return_attention: если True, дополнительно вернуть attention weights

        Returns:
            out: [batch_size, seq_length, embed_dim]
            attention_weights (опционально)
        """

        # ============================================================
        # 1. PRE-LN + SELF-ATTENTION + RESIDUAL
        # ============================================================
        attn_input = self.norm_1(x)

        attn_output, attention_weights = self.attention(attn_input, mask=mask)

        x = x + self.dropout(attn_output)

        # ============================================================
        # 2. PRE-LN + FFN + RESIDUAL
        # ============================================================
        ffn_input = self.norm_2(x)

        ffn_output = self.ffn(ffn_input)

        out = x + self.dropout(ffn_output)

        if return_attention:
            return out, attention_weights

        return out

    def get_num_parameters(self):
        """Возвращает количество обучаемых параметров в блоке."""
        return sum(p.numel() for p in self.parameters())