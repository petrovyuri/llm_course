# -*- coding: utf-8 -*-
"""
Данные для дообучения: пары «вопрос - ответ» по рассказу про Барсика.

Все ответы взяты из cat_story.txt почти дословно. Это важно: модель
уже читала этот текст на этапе обучения, значит знания у неё есть.
Дообучением мы не добавляем ей новых фактов, а показываем формат:
после вопроса идёт короткий ответ, а потом нужно замолчать.
"""

# Шаблон промпта. Ровно в таком виде модель будет видеть вопрос
# и на обучении, и потом в чате. Формат менять нельзя: модель
# привыкает именно к этим словам и переносу строки.
PROMPT_TEMPLATE = "Question: {question}\nAnswer:"

QA_PAIRS = [
    ("What is the cat name?", "Barsik."),
    ("What did Barsik like?", "Barsik liked to sleep."),
    ("What did Barsik love?", "Barsik loved to eat."),
    ("Where did the cat live?", "He lived in the house."),
    ("What was the cat catching?", "The cat was catching mice."),
    ("Who was afraid of the cat?", "The mice were afraid of the cat."),
    ("What was the cat drinking?", "He was drinking milk."),
    ("Where was the cat walking?", "The cat was walking in the garden."),
    ("Who loved the cat?", "The mistress loved the cat."),
    ("What was the cat doing in the evening?", "He was going to bed."),
    ("What was the cat dreaming about?", "The cat was flying."),
    ("Where was the cat basking?", "The cat was basking in the window."),
]


def build_prompt(question):
    """Собирает промпт из вопроса: то, что видит модель перед ответом."""
    return PROMPT_TEMPLATE.format(question=question)


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")

    print(f"Пар вопрос-ответ: {len(QA_PAIRS)}")
    print()
    print("Так выглядит один обучающий пример целиком:")
    question, answer = QA_PAIRS[0]
    print(repr(build_prompt(question) + " " + answer))
