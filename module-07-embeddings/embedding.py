import torch
import torch.nn as nn
from config import settings

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_dim):
        super().__init__(vocab_size, embed_dim)

class PositionalEmbedding(nn.Module):
    def __init__(self, max_length, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(max_length, embed_dim)

    def forward(self, x):
        batch_size, seq_length = x.shape
        positions = torch.arange(0, seq_length, device=x.device).unsqueeze(0)
        return self.embedding(positions)

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters())

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, max_length, embed_dim):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, embed_dim)
        self.position_embedding = PositionalEmbedding(max_length, embed_dim)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(x)
        return token_emb + pos_emb

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters())
