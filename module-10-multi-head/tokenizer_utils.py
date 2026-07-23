from transformers import AutoTokenizer
from config import settings

def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(settings.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def encode_text(tokenizer, text):
    return tokenizer.encode(text)
