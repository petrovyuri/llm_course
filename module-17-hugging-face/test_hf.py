# -*- coding: utf-8 -*-
"""
Тесты модуля 17: работа с форматами и настоящими моделями.

Запуск: py -3.11 -m pytest test_hf.py -v
Часть тестов скачивает модели с Hugging Face - нужен интернет.
"""
import json
import os

import pytest
import torch

from save_safetensors import load_model_from_safetensors, save_model_to_safetensors


@pytest.fixture(scope="session", autouse=True)
def artifacts():
    """
    Тестам нужны файлы, которые создают save_safetensors.py и quant_manual.py
    при запуске. Создаём их здесь, вызовом тех же функций, которыми пользуются
    сами скрипты, - чтобы тесты не зависели от того, запускали их или нет.
    """
    if not (os.path.exists("chat_model.safetensors")
            and os.path.exists("simple_llm_config.json")):
        save_model_to_safetensors(
            "chat_model.pth", "chat_model.safetensors", "simple_llm_config.json"
        )

    if not os.path.exists("chat_model_int8.safetensors"):
        from quant_manual import build_int8_weights_file

        build_int8_weights_file()


def test_safetensors_roundtrip(tmp_path):
    """Веса, записанные в safetensors, читаются обратно без потерь."""
    weights = str(tmp_path / "m.safetensors")
    config = str(tmp_path / "m.json")

    save_model_to_safetensors("chat_model.pth", weights, config)

    original = torch.load("chat_model.pth", map_location="cpu", weights_only=True)
    restored = load_model_from_safetensors(weights, config).state_dict()

    assert set(original) == set(restored)
    for name, tensor in original.items():
        assert torch.equal(tensor, restored[name]), name


def test_safetensors_config_matches_model(tmp_path):
    """Рядом с весами лежит конфигурация, по которой модель собирается."""
    weights = str(tmp_path / "m.safetensors")
    config_path = str(tmp_path / "m.json")

    save_model_to_safetensors("chat_model.pth", weights, config_path)

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    assert config["vocab_size"] == 50257
    assert config["embed_dim"] == 64
    assert config["num_heads"] == 2
    assert config["num_layers"] == 2
    assert config["ff_dim"] == 256
    assert config["max_seq_length"] == 32


def test_safetensors_file_is_not_bigger_than_pth():
    """Формат не раздувает файл: те же веса, а размер не больше - обычно даже чуть меньше."""
    assert os.path.getsize("chat_model.safetensors") <= os.path.getsize("chat_model.pth")


def test_gpt2_shapes_match_its_config():
    """
    Формы весов GPT-2 выводятся из её config.json - как у нашей модели.

    Таблица сопоставления в уроке 4 показывает 12 тензоров - все 12 и
    проверены здесь (ln_1, ln_2, ln_f, attn.c_proj и mlp.c_proj раньше были
    не покрыты; lm_head проверен отдельно, в test_gpt2_ties_lm_head_to_embedding).
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    state = model.state_dict()

    assert tuple(state["transformer.wte.weight"].shape) == (config.vocab_size, config.n_embd)
    assert tuple(state["transformer.wpe.weight"].shape) == (config.n_positions, config.n_embd)
    assert tuple(state["transformer.h.0.ln_1.weight"].shape) == (config.n_embd,)
    # Q, K и V склеены в одну матрицу, отсюда тройка
    assert tuple(state["transformer.h.0.attn.c_attn.weight"].shape) == (config.n_embd, 3 * config.n_embd)
    assert tuple(state["transformer.h.0.attn.c_proj.weight"].shape) == (config.n_embd, config.n_embd)
    assert tuple(state["transformer.h.0.ln_2.weight"].shape) == (config.n_embd,)
    assert tuple(state["transformer.h.0.mlp.c_fc.weight"].shape) == (config.n_embd, 4 * config.n_embd)
    assert tuple(state["transformer.h.0.mlp.c_proj.weight"].shape) == (4 * config.n_embd, config.n_embd)
    assert tuple(state["transformer.ln_f.weight"].shape) == (config.n_embd,)
    assert len(model.transformer.h) == config.n_layer


def test_gpt2_ties_lm_head_to_embedding():
    """Выходной слой GPT-2 - это та же таблица эмбеддингов (weight tying)."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    assert model.lm_head.weight.data_ptr() == model.transformer.wte.weight.data_ptr()


def test_our_model_and_gpt2_share_vocabulary():
    """Словарь один и тот же: мы с модуля 6 живём на токенизаторе gpt2."""
    import json

    from transformers import AutoConfig

    with open("simple_llm_config.json", encoding="utf-8") as f:
        ours = json.load(f)

    assert ours["vocab_size"] == AutoConfig.from_pretrained("gpt2").vocab_size


def test_our_greedy_sampler_matches_generate():
    """Наш sample_next_token из модуля 15 повторяет жадный выбор generate."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from sampling import sample_next_token

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.eval()

    input_ids = tokenizer("The cat loved sour cream", return_tensors="pt")["input_ids"]

    # Что выберет библиотека
    with torch.no_grad():
        library = model.generate(
            input_ids, max_new_tokens=5, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Что выберем мы, своим кодом, по одному токену за шаг
    ours = input_ids
    for _ in range(5):
        with torch.no_grad():
            logits = model(ours).logits

        next_token = sample_next_token(logits[:, -1, :], do_sample=False)
        ours = torch.cat([ours, next_token], dim=1)

    assert torch.equal(library, ours), (
        tokenizer.decode(library[0]), tokenizer.decode(ours[0])
    )


def test_greedy_picks_largest_logit():
    """Жадный выбор берёт токен с самым большим логитом, температура не участвует."""
    import torch

    from sampling import sample_next_token

    logits = torch.tensor([[1.0, 2.0, 3.0]])

    token = sample_next_token(logits, do_sample=False)
    assert token.item() == 2


def test_smollm2_config_matches_lesson_6():
    """Конфигурация SmolLM2 даёт ровно те числа, что напечатаны в уроке 6."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")

    assert config.architectures[0] == "LlamaForCausalLM"
    assert config.hidden_size == 576
    assert config.num_hidden_layers == 30
    assert config.num_attention_heads == 9
    assert config.num_key_value_heads == 3  # grouped-query attention
    assert config.intermediate_size == 1536
    assert config.vocab_size == 49152
    assert config.max_position_embeddings == 8192
    assert config.hidden_act == "silu"
    assert config.tie_word_embeddings is True


def test_smollm2_chat_template_wraps_roles_in_im_tags():
    """apply_chat_template превращает роли в строку со спецтокенами <|im_start|>/<|im_end|>."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France? Answer in one sentence."},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    assert text.startswith("<|im_start|>system")
    assert text.endswith("<|im_start|>assistant\n")
    assert tokenizer.eos_token == "<|im_end|>"


def test_smollm2_answers_and_stops_on_eos():
    """Модель отвечает на вопрос и сама останавливается на EOS, не доходя до лимита токенов."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    model.eval()

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France? Answer in one sentence."},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )

    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=30, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    answer_ids = output[0, inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(answer_ids, skip_special_tokens=True)

    assert len(answer_ids) < 30  # остановилась сама, не упёршись в лимит
    assert answer_ids[-1].item() == tokenizer.eos_token_id
    assert "Paris" in answer


def test_int8_error_is_within_half_step():
    """
    Ошибка округления в каждой строке не превышает половину шага квантования
    ЭТОЙ строки.

    Множитель считается по строкам (урок 7), а не один на всю матрицу -
    поэтому сравниваем построчно, а не с глобальным scale.max(). Старая
    версия теста сравнивала с глобальным максимумом и остаётся зелёной даже
    если quantize_int8 откатить на один множитель на всю матрицу - тогда
    сжатие 3,76 из урока 8 перестало бы быть верным, а тест бы этого не заметил.
    """
    import torch

    from quant_manual import dequantize, quantize_int8

    torch.manual_seed(0)
    weight = torch.randn(64, 128) * 0.1

    q, scale = quantize_int8(weight)
    restored = dequantize(q, scale)

    assert q.dtype == torch.int8
    assert scale.shape == (64, 1)  # множитель свой у каждой строки, а не один на всю матрицу

    per_row_error = (restored - weight).abs().amax(dim=1, keepdim=True)
    assert (per_row_error <= (scale / 2) * 1.001).all()


def test_int8_model_is_about_four_times_smaller():
    """Один байт на вес вместо четырёх - файл худеет примерно вчетверо."""
    import os

    assert os.path.getsize("chat_model_int8.safetensors") * 3.4 < os.path.getsize("chat_model.safetensors")


def test_quantized_model_answers_the_same():
    """После квантизации модель отвечает на все 12 вопросов теми же словами."""
    from quant_manual import answer_all, build_int8_model, build_fp32_model

    fp32 = answer_all(build_fp32_model())
    int8 = answer_all(build_int8_model())

    assert fp32 == int8, [p for p in zip(fp32, int8) if p[0] != p[1]]


def test_quanto_int8_produces_weightqbytestensor_with_honest_memory_saving():
    """
    quant_hf.py (урок 8, библиотечная квантизация) не был покрыт тестами вовсе,
    хотя в нём два несущих факта, зависящих от версии optimum-quanto/transformers:

    1. после int8-квантизации вес - это WeightQBytesTensor типа quanto.qint8,
       а не обычный тензор;
    2. честный подсчёт памяти (через приватные _data/_scale) заметно меньше
       наивного подсчёта через element_size() - тот квантизацию не видит и
       "врёт", будто памяти столько же, сколько в bfloat16.

    Модель тяжёлая, поэтому обе проверки - на одной загрузке, в одном тесте.
    """
    from transformers import AutoModelForCausalLM, QuantoConfig

    model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM2-135M-Instruct",
        quantization_config=QuantoConfig(weights="int8"),
    )

    weight = model.model.layers[0].self_attn.q_proj.weight
    assert type(weight).__name__ == "WeightQBytesTensor"
    assert f"{weight.qtype}" == "quanto.qint8"

    naive_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    real_bytes = 0
    for p in model.parameters():
        if hasattr(p, "_data"):
            real_bytes += p._data.numel() * p._data.element_size()
            real_bytes += p._scale.numel() * p._scale.element_size()
        else:
            real_bytes += p.numel() * p.element_size()

    # честно посчитанная память заметно (не на проценты) меньше наивной
    assert real_bytes < naive_bytes * 0.8


def test_publish_and_load_from_hub_exit_before_network_without_real_repo(monkeypatch):
    """
    Свойство, на котором держится урок 9: без настоящего REPO_ID (плейсхолдер
    "ВАШ_НИК/...") оба скрипта публикации останавливаются сами, до единого
    обращения к сети - ни на чтение, ни на запись. Раньше это не проверялось
    ничем; monkeypatch делает сетевые функции "взрывными", чтобы тест падал,
    если код когда-нибудь дотянется до них раньше проверки плейсхолдера.
    """
    import runpy
    import sys

    import huggingface_hub

    def _forbidden(*_args, **_kwargs):
        raise AssertionError("сетевой вызов не должен происходить без настоящего REPO_ID")

    monkeypatch.setattr(huggingface_hub, "HfApi", _forbidden)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _forbidden)

    # publish.py: проверка плейсхолдера стоит внутри `if __name__ == "__main__":`,
    # поэтому обычный import её не исполнит - нужно запустить файл как скрипт.
    with pytest.raises(SystemExit) as exc_publish:
        runpy.run_path("publish.py", run_name="__main__")
    assert exc_publish.value.code == 0

    # load_from_hub.py: та же проверка стоит на верхнем уровне модуля.
    sys.modules.pop("load_from_hub", None)
    try:
        with pytest.raises(SystemExit) as exc_load:
            import load_from_hub  # noqa: F401
        assert exc_load.value.code == 0
    finally:
        sys.modules.pop("load_from_hub", None)
