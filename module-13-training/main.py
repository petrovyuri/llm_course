from config import settings
from loader import create_dataloader
from tokenizer_utils import get_tokenizer
from embedding import EmbeddingLayer
import torch

def main():
    print(f"Запуск проекта: {settings.text_file}")
    with open(settings.text_file, "r", encoding="utf-8") as f:
        raw_text = f.read()
    dataloader = create_dataloader(raw_text, shuffle=False)
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.vocab_size
    embedding_layer = EmbeddingLayer(vocab_size=vocab_size, max_length=settings.max_length, embed_dim=settings.embed_dim)
    print(f"Размер словаря: {vocab_size}")
    print(f"Размер эмбеддинга: {settings.embed_dim}")
    print("\n=== Проверка слоя Embedding ===")
    for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
        print(f"Входные IDs (X): {batch_x.shape}")
        embedded_x = embedding_layer(batch_x)
        print(f"Векторные представления: {embedded_x.shape}")
        first_vector = embedded_x[0, 0, :]
        print(f"Первый вектор (пример): {first_vector[:5]}...")
        print(f"Требует градиентов: {embedded_x.requires_grad}")
        break

if __name__ == "__main__":
    main()
