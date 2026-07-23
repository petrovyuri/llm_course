import torch
from model import SimpleLLM
from config import settings
from tokenizer_utils import get_tokenizer
from sampling import sample_next_token

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
# weights_only=True - безопасная загрузка: читаем только тензоры весов
model.load_state_dict(
    torch.load("best_model.pth", map_location=device, weights_only=True)
)

model.eval()


def generate(model, tokenizer, prompt, max_new_tokens=20,
             do_sample=False, temperature=1.0, top_k=None, top_p=None):
    model.eval()

    # Кодируем начальный текст
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_ids], device=device)

    for _ in range(max_new_tokens):

        # Модель видит только последние max_length токенов,
        # но полную последовательность мы сохраняем
        context = input_ids[:, -settings.max_length:]

        with torch.no_grad():
            logits = model(context)

        # Берём логиты последнего токена
        last_logits = logits[:, -1, :]

        # Выбираем следующий токен: жадно или сэмплированием
        next_token = sample_next_token(
            last_logits,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

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

    greedy_result = generate(model, tokenizer, prompt, max_new_tokens=20)
    sampled_result = generate(
        model, tokenizer, prompt, max_new_tokens=20,
        do_sample=True, temperature=1.0, top_p=0.9,
    )

    print("\n=== Greedy (как в модуле 14) ===")
    print(greedy_result)

    print("\n=== Сэмплирование (T=1.0, top-p=0.9) ===")
    print(sampled_result)
