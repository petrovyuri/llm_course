# -*- coding: utf-8 -*-
"""
Тесты модуля 15: сэмплирование, температура, top-k, top-p.

Работают на маленьких логитах из ручного примера модуля 12,
модель не нужна. Запуск: python test_sampling.py
"""
import sys
import torch

from sampling import filter_top_k, filter_top_p, sample_next_token

sys.stdout.reconfigure(encoding="utf-8")

# Логиты из ручного примера: ["кот", "любил", "сметану", "он", "был"]
LOGITS = torch.tensor([[0.085, 0.201, 0.944, 0.334, -0.155]])


def test_greedy_equals_argmax():
    """do_sample=False всегда возвращает argmax (индекс 2, «сметану»)."""
    for _ in range(5):
        token = sample_next_token(LOGITS, do_sample=False)
        assert token.item() == 2, f"ожидали 2, получили {token.item()}"
    print("Тест 1 (greedy = argmax): OK")


def test_low_temperature_is_greedy():
    """При очень низкой температуре сэмплирование сходится к greedy."""
    for _ in range(20):
        token = sample_next_token(LOGITS, do_sample=True, temperature=0.001)
        assert token.item() == 2, f"ожидали 2, получили {token.item()}"
    print("Тест 2 (T->0 = greedy): OK")


def test_top_k_keeps_k_tokens():
    """После top-k=2 остаются ровно 2 конечных логита - два максимума."""
    filtered = filter_top_k(LOGITS, k=2)
    finite = torch.isfinite(filtered[0])
    assert finite.sum().item() == 2, f"ожидали 2 токена, осталось {finite.sum().item()}"
    # Выжить должны «сметану» (индекс 2) и «он» (индекс 3)
    assert finite[2] and finite[3], "top-k=2 должен оставить индексы 2 и 3"
    print("Тест 3 (top-k оставляет k токенов): OK")


def test_top_p_nucleus():
    """Top-p=0.7 оставляет минимальное ядро с массой >= 0.7 (здесь 3 токена)."""
    filtered = filter_top_p(LOGITS, p=0.7)
    finite = torch.isfinite(filtered[0])
    assert finite.sum().item() == 3, f"ожидали ядро из 3 токенов, вышло {finite.sum().item()}"
    # Ядро: «сметану», «он», «любил» (индексы 2, 3, 1)
    assert finite[2] and finite[3] and finite[1], "в ядре должны быть индексы 1, 2, 3"
    # Масса ядра в исходном распределении не меньше p
    probs = torch.softmax(LOGITS, dim=-1)
    mass = probs[0][finite].sum().item()
    assert mass >= 0.7, f"масса ядра {mass:.3f} < 0.7"
    print("Тест 4 (top-p собирает ядро >= p): OK")


def test_sampling_respects_filter():
    """Сэмплирование никогда не выбирает токен, отброшенный фильтром."""
    torch.manual_seed(42)
    allowed = {2, 3}  # top-k=2 оставляет только их
    for _ in range(200):
        token = sample_next_token(LOGITS, do_sample=True, top_k=2)
        assert token.item() in allowed, f"выбран запрещённый токен {token.item()}"
    print("Тест 5 (фильтр непробиваем): OK")


if __name__ == "__main__":
    test_greedy_equals_argmax()
    test_low_temperature_is_greedy()
    test_top_k_keeps_k_tokens()
    test_top_p_nucleus()
    test_sampling_respects_filter()
    print("\nВсе тесты пройдены: 5/5")
