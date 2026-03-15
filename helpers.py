import torch
from torch.utils.data import TensorDataset


def build_dummy_dataset(num_samples: int = 200, num_classes: int = 10):
    x = torch.randn(num_samples, 3, 32, 32)
    y = torch.randint(0, num_classes, (num_samples,))
    return TensorDataset(x, y)
