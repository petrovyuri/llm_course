# -*- coding: utf-8 -*-
"""
Тесты модуля 16: формат данных и маскирование loss.

Модель для них не нужна: проверяем, как устроен датасет.
Запуск: python test_finetune.py
"""
import sys

import torch

from config import settings
from qa_data import QA_PAIRS, build_prompt
from qa_dataset import QADataset, IGNORE_INDEX
from tokenizer_utils import get_tokenizer

sys.stdout.reconfigure(encoding="utf-8")

tokenizer = get_tokenizer()
dataset = QADataset(QA_PAIRS, tokenizer, settings.max_length)


def test_prompt_format():
    """Промпт собирается по шаблону Question/Answer."""
    prompt = build_prompt("What is the cat name?")
    assert prompt == "Question: What is the cat name?\nAnswer:", prompt
    print("Тест 1 (формат промпта): OK")


def test_dataset_size():
    """В датасете столько примеров, сколько пар вопрос-ответ."""
    assert len(dataset) == len(QA_PAIRS), f"ожидали {len(QA_PAIRS)}, вышло {len(dataset)}"
    inputs, labels = dataset[0]
    expected = settings.max_length - 1
    assert inputs.shape == (expected,), inputs.shape
    assert labels.shape == (expected,), labels.shape
    print("Тест 2 (размер датасета и формы): OK")


def test_prompt_is_masked():
    """Токены вопроса закрыты меткой -100, токены ответа - нет."""
    inputs, labels = dataset[0]

    # Первый токен ответа стоит сразу после промпта
    prompt_len = len(tokenizer.encode(build_prompt(QA_PAIRS[0][0])))

    masked = labels[: prompt_len - 1]
    assert (masked == IGNORE_INDEX).all(), "промпт должен быть замаскирован"

    answer_part = labels[prompt_len - 1:]
    assert (answer_part != IGNORE_INDEX).any(), "ответ маскировать нельзя"
    print("Тест 3 (промпт замаскирован, ответ нет): OK")


def test_answer_ends_with_eos():
    """Каждый ответ заканчивается токеном EOS: модель учится молчать."""
    for idx in range(len(dataset)):
        _, labels = dataset[idx]
        real = labels[labels != IGNORE_INDEX]
        assert real[-1].item() == tokenizer.eos_token_id, f"пример {idx} без EOS"
    print("Тест 4 (ответ заканчивается EOS): OK")


def test_loss_ignores_prompt():
    """cross_entropy с ignore_index не считает потери на промпте."""
    import torch.nn.functional as F

    vocab = tokenizer.vocab_size
    _, labels = dataset[0]

    # Случайные логиты: важно не значение loss, а то, что он конечный
    torch.manual_seed(0)
    logits = torch.randn(len(labels), vocab)

    loss = F.cross_entropy(logits, labels, ignore_index=IGNORE_INDEX)
    assert torch.isfinite(loss), "loss должен быть конечным"

    # Если бы маску не пропускали, метка -100 сломала бы расчёт
    all_masked = torch.full_like(labels, IGNORE_INDEX)
    empty = F.cross_entropy(logits, all_masked, ignore_index=IGNORE_INDEX)
    assert torch.isnan(empty), "при полностью замаскированном примере loss не определён"
    print("Тест 5 (loss считается только по ответу): OK")


if __name__ == "__main__":
    test_prompt_format()
    test_dataset_size()
    test_prompt_is_masked()
    test_answer_ends_with_eos()
    test_loss_ignores_prompt()
    print("\nВсе тесты пройдены: 5/5")
