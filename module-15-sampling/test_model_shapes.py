import torch
from model import SimpleLLM


def test_model_output_shape():
    print("\n" + "=" * 70)
    print("ТЕСТ: Проверка формы выхода всей модели")
    print("=" * 70)

    vocab_size = 20
    max_seq_length = 8
    batch_size = 2
    seq_length = 5

    model = SimpleLLM(
        vocab_size=vocab_size,
        max_seq_length=max_seq_length,
        embed_dim=16,
        num_heads=2,
        num_layers=2,
        ff_dim=64,
        dropout=0.0,
    )

    token_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    logits = model(token_ids)

    print(f"Вход token_ids: {token_ids.shape}")
    print(f"Выход logits:   {logits.shape}")

    assert logits.shape == (batch_size, seq_length, vocab_size), (
        f"Ожидалось {(batch_size, seq_length, vocab_size)}, "
        f"получено {logits.shape}"
    )

    print("Тест пройден ✓")


if __name__ == "__main__":
    test_model_output_shape()