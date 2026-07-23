from dataclasses import dataclass

@dataclass
class Config:
    model_name: str = "gpt2"
    text_file: str = "cat_story.txt"
    max_length: int = 4
    stride: int = 1
    batch_size: int = 2
    seed: int = 42

settings = Config()
