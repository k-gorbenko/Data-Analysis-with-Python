import numpy as np
import torch
from torch import nn

def create_model():
    # Linear layer mapping from 784 features, so it should be 784->256->16->10

    model = nn.Sequential(
        nn.Linear(784, 256),  # Первый линейный слой (784 -> 256)
        nn.ReLU(),            # Функция активации ReLU
        nn.Linear(256, 16),   # Второй линейный слой (256 -> 16)
        nn.ReLU(),            # Функция активации ReLU
        nn.Linear(16, 10)     # Третий линейный слой (16 -> 10)
        # Последний слой без функции активации
    )


    # return model instance (None is just a placeholder)

    return model

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params
