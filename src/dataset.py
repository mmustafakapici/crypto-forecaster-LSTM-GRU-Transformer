import torch
from torch.utils.data import Dataset
from .data import create_sequences

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, seq_len: int):
        Xs, ys = create_sequences(X, y, seq_len)
        self.X = torch.tensor(Xs, dtype=torch.float32)
        self.y = torch.tensor(ys, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
