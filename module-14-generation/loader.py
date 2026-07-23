from torch.utils.data import DataLoader
from dataset import create_dataset
from config import settings

def create_dataloader(text, batch_size=None, shuffle=False, drop_last=True):
    if batch_size is None:
        batch_size = settings.batch_size
    dataset = create_dataset(text)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return dataloader
