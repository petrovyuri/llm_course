from dataclasses import dataclass

@dataclass
class Config:
    model_name: str = "gpt2"
    text_file: str = "cat_story.txt"
    max_length: int = 32               # Контекстное окно: было 4, стало 32
    stride: int = 2
    batch_size: int = 4
    seed: int = 42
    embed_dim: int = 16
    num_epochs: int = 200              # Количество эпох обучения

settings = Config()
