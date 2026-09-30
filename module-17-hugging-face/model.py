import torch
import torch.nn as nn

from attention_multihead import CausalMask
from transformer_block import TransformerBlock


class SimpleLLM(nn.Module):
    """
    Простейшая decoder-only LLM.

    Архитектура:
        token_embedding
        + position_embedding
        -> stack of TransformerBlock
        -> final LayerNorm
        -> lm_head

    Вход:
        token_ids: [batch_size, seq_length]

    Выход:
        logits: [batch_size, seq_length, vocab_size]
    """

    def __init__(
        self,
        vocab_size,
        max_seq_length,
        embed_dim=16,
        num_heads=2,
        num_layers=2,
        ff_dim=None,
        dropout=0.0,
    ):
        super().__init__()

        if ff_dim is None:
            ff_dim = 4 * embed_dim

        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_dim = ff_dim

        # ============================================================
        # ЭМБЕДДИНГИ
        # ============================================================
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_length, embed_dim)

        # ============================================================
        # СТЕК TRANSFORMER BLOCK
        # ============================================================
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Финальная нормализация перед выходом
        self.final_norm = nn.LayerNorm(embed_dim)

        # ============================================================
        # OUTPUT LAYER
        # ============================================================
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        """
        Простая инициализация весов.
        """
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    def forward(self, token_ids):
        """
        Args:
            token_ids: [batch_size, seq_length]

        Returns:
            logits: [batch_size, seq_length, vocab_size]
        """
        batch_size, seq_length = token_ids.shape
        device = token_ids.device

        if seq_length > self.max_seq_length:
            raise ValueError(
                f"seq_length ({seq_length}) больше max_seq_length ({self.max_seq_length})"
            )

        # ============================================================
        # 1. ТОКЕНЫ + ПОЗИЦИИ
        # ============================================================
        positions = torch.arange(seq_length, device=device).unsqueeze(0)

        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        # ============================================================
        # 2. КАУЗАЛЬНАЯ МАСКА
        # ============================================================
        mask = CausalMask.create(seq_length, device=device)

        # ============================================================
        # 3. ПРОХОД ЧЕРЕЗ ВСЕ TRANSFORMER BLOCK
        # ============================================================
        for block in self.blocks:
            x = block(x, mask=mask)

        # ============================================================
        # 4. ФИНАЛЬНАЯ НОРМАЛИЗАЦИЯ
        # ============================================================
        x = self.final_norm(x)

        # ============================================================
        # 5. ПРОЕКЦИЯ В ПРОСТРАНСТВО СЛОВАРЯ
        # ============================================================
        logits = self.lm_head(x)

        return logits

    def get_num_parameters(self):
        """Количество параметров модели."""
        return sum(p.numel() for p in self.parameters())