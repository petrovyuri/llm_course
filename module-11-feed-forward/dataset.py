import torch
from torch.utils.data import Dataset
from tokenizer_utils import get_tokenizer, encode_text
from config import settings

class SimpleDataset(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.input_ids = encode_text(tokenizer, text)

    def __len__(self):
        return (len(self.input_ids) - self.max_length - 1) // self.stride + 1

    def __getitem__(self, idx):
        start_idx = idx * self.stride
        end_idx = start_idx + self.max_length
        chunk = self.input_ids[start_idx:end_idx]
        targets = self.input_ids[start_idx + 1 : end_idx + 1]
        return torch.tensor(chunk, dtype=torch.long), torch.tensor(targets, dtype=torch.long)

def create_dataset(text):
    tokenizer = get_tokenizer()
    return SimpleDataset(text=text, tokenizer=tokenizer, max_length=settings.max_length, stride=settings.stride)
