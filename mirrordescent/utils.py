import warnings

import torch

def disable_warnings():
    warnings.filterwarnings('ignore')

def gamma(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(torch.lgamma(x))