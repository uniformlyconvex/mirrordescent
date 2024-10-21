import warnings

import torch

def disable_warnings():
    warnings.filterwarnings('ignore')

def rd_to_rdp1(x: torch.Tensor) -> torch.Tensor:
    """
    Pad the d-dimensional input x, such that sum_{i=1}^d x_i <=1, to d+1 dimensions
    such that sum_{i=1}^{d+1} x_i = 1.
    """
    if not x.shape:  # x is a scalar
        return torch.cat(
            (x.reshape(1), torch.tensor([1 - x]))
        )
    
    if x.ndim == 1:  # x is a vector
        return torch.cat(
            (x, torch.tensor([1 - torch.sum(x)]))
        )
    
    if x.ndim == 2:  # x is a matrix
        return torch.cat(
            (x, 1 - torch.sum(x, axis=1, keepdim=True)),
            axis=1
        )