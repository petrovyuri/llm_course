import torch
from model import SimpleLLM
from config import settings
from tokenizer_utils import get_tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Загружаем токенизатор ===
tokenizer = get_tokenizer()
vocab_size = tokenizer.vocab_size

# === Создаём модель ===
model = SimpleLLM(
    vocab_size=vocab_size,
    max_seq_length=settings.max_length,
    embed_dim=64,
    num_heads=2,
    num_layers=2,
    ff_dim=256,
).to(device)

# === Загружаем обученные веса ===
model.load_state_dict(torch.load("best_model.pth", map_location=device))

model.eval()


def generate(model, tokenizer, prompt, max_new_tokens=20):
    model.eval()

    # Кодируем начальный текст
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_ids], device=device)

    for _ in range(max_new_tokens):

        # Ограничиваем длину (если длиннее max_seq_length)
        input_ids = input_ids[:, -settings.max_length:]

        with torch.no_grad():
            logits = model(input_ids)

        # Берём логиты последнего токена
        last_logits = logits[:, -1, :]

        # Выбираем следующий токен
        next_token = torch.argmax(last_logits, dim=-1, keepdim=True)

        # Добавляем его к последовательности
        input_ids = torch.cat([input_ids, next_token], dim=1)

    # Декодируем обратно в текст
    output_text = tokenizer.decode(input_ids[0].tolist())

    return output_text


if __name__ == "__main__":
    # Запрашиваем промт у пользователя
    prompt = input("Введите начальный текст (промт): ").strip()

    # Если пользователь ничего не ввел, используем текст по умолчанию
    if not prompt:
        prompt = "кот"
        print(f"Используется промт по умолчанию: '{prompt}'")

    print("\nГенерация текста...")
    print(f"Промт: '{prompt}'")
    result = generate(model, tokenizer, prompt, max_new_tokens=20)

    print("\n=== Результат ===")
    print(result)
