import torch


def filter_top_k(logits, k):
    """
    Top-k: оставляет k самых больших логитов, остальным ставит -inf.

    После softmax токены с логитом -inf получают вероятность 0,
    то есть полностью выбывают из выбора.

    Args:
        logits: [batch_size, vocab_size]
        k: сколько кандидатов оставить

    Returns:
        logits той же формы, где всё, кроме топ-k, равно -inf
    """
    top_values, _ = torch.topk(logits, k)

    # Порог - самый маленький логит из топ-k
    threshold = top_values[:, -1].unsqueeze(-1)

    # Всё, что ниже порога, выключаем
    return logits.masked_fill(logits < threshold, float("-inf"))


def filter_top_p(logits, p):
    """
    Top-p (nucleus): оставляет минимальное ядро самых вероятных токенов,
    чья суммарная вероятность достигает p. Остальным ставит -inf.

    Args:
        logits: [batch_size, vocab_size]
        p: порог накопленной вероятности, число от 0 до 1

    Returns:
        logits той же формы, где все токены вне ядра равны -inf
    """
    # 1. Сортируем логиты по убыванию
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)

    # 2. Вероятности отсортированных токенов и накопленная сумма
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)

    # 3. Токен выбывает, если сумма ДО него уже достигла p.
    #    Токен, на котором порог пересекли, остаётся в ядре -
    #    поэтому при любом p > 0 хотя бы один токен выживает.
    remove = (cumulative - sorted_probs) >= p
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))

    # 4. Возвращаем логиты на их исходные места в словаре
    filtered = torch.full_like(logits, float("-inf"))
    filtered.scatter_(-1, sorted_indices, sorted_logits)

    return filtered


def sample_next_token(last_logits, do_sample=False, temperature=1.0,
                      top_k=None, top_p=None):
    """
    Выбирает следующий токен по логитам последней позиции.

    Args:
        last_logits: [batch_size, vocab_size] - логиты последнего токена
        do_sample: False - жадный выбор (argmax, как в модуле 14),
                   True - случайный выбор по вероятностям
        temperature: температура softmax (используется при do_sample=True)
        top_k: если задано - оставить только k самых вероятных токенов
        top_p: если задано - оставить ядро с суммарной вероятностью >= p

    Returns:
        next_token: [batch_size, 1]
    """
    # Жадный режим: всегда самый вероятный токен
    if not do_sample:
        return torch.argmax(last_logits, dim=-1, keepdim=True)

    # 1. Температура: меняем резкость распределения
    logits = last_logits / temperature

    # 2. Отсекаем хвост распределения
    if top_k is not None:
        logits = filter_top_k(logits, top_k)
    if top_p is not None:
        logits = filter_top_p(logits, top_p)

    # 3. Логиты -> вероятности
    probs = torch.softmax(logits, dim=-1)

    # 4. Тянем случайный токен пропорционально вероятностям
    next_token = torch.multinomial(probs, num_samples=1)

    return next_token
