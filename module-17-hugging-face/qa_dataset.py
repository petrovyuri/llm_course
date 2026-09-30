# -*- coding: utf-8 -*-
"""
Датасет для дообучения (SFT, supervised fine-tuning).

Отличие от SimpleDataset модуля 6: там мы резали сплошной текст скользящим
окном и учили модель предсказывать каждый следующий токен. Здесь один
пример - это одна пара «вопрос - ответ», и loss мы считаем только по ответу.

Токены промпта закрываем меткой -100. Функция cross_entropy пропускает такие
позиции (ignore_index=-100), поэтому за вопрос модель не получает ни награды,
ни штрафа. Учим её только тому, что отвечать.
"""
import torch
from torch.utils.data import Dataset

from qa_data import build_prompt

IGNORE_INDEX = -100


class QADataset(Dataset):
    """
    Пары «вопрос - ответ» для дообучения.

    Один пример:
        вход:  Question: What is the cat name?\nAnswer: Barsik.<|endoftext|>
        loss:  считается только по части « Barsik.<|endoftext|>»
    """

    def __init__(self, qa_pairs, tokenizer, max_length):
        self.inputs = []
        self.labels = []

        eos_id = tokenizer.eos_token_id

        for question, answer in qa_pairs:
            # 1. Промпт (вопрос) и ответ кодируем по отдельности:
            #    так мы знаем, где заканчивается одно и начинается другое.
            prompt_ids = tokenizer.encode(build_prompt(question))

            # Пробел перед ответом нужен: токенизатор GPT-2 считает
            # « Barsik» и «Barsik» разными токенами.
            # EOS в конце - сигнал «ответ закончен, дальше молчим».
            answer_ids = tokenizer.encode(" " + answer) + [eos_id]

            # 2. Модель видит вопрос и ответ подряд, одной строкой
            input_ids = prompt_ids + answer_ids

            # 3. А учится только на ответе: промпт закрываем -100
            labels = [IGNORE_INDEX] * len(prompt_ids) + answer_ids

            # 4. Обрезаем по контекстному окну и добиваем до одной длины
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]

            padding = max_length - len(input_ids)
            input_ids = input_ids + [eos_id] * padding
            labels = labels + [IGNORE_INDEX] * padding

            # 5. Сдвиг на один токен: по input_ids[i] предсказываем labels[i]
            self.inputs.append(torch.tensor(input_ids[:-1], dtype=torch.long))
            self.labels.append(torch.tensor(labels[1:], dtype=torch.long))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")

    from config import settings
    from qa_data import QA_PAIRS
    from tokenizer_utils import get_tokenizer

    tokenizer = get_tokenizer()
    dataset = QADataset(QA_PAIRS, tokenizer, settings.max_length)

    print(f"Примеров в датасете: {len(dataset)}")

    inputs, labels = dataset[0]
    print(f"Форма входа: {tuple(inputs.shape)}, форма меток: {tuple(labels.shape)}")
    print()
    print("Первый пример по токенам (метка -100 = не учимся на этом токене):")
    print(f"{'токен':>16} | {'вход':>6} | {'метка':>6}")
    for token_id, label in zip(inputs.tolist(), labels.tolist()):
        text = tokenizer.decode([token_id])
        mark = "-100" if label == IGNORE_INDEX else str(label)
        print(f"{text!r:>16} | {token_id:>6} | {mark:>6}")
