from numpy.typing import ArrayLike

import torch

from mirrordescent.mirror_maps import EntropicMirrorMap

def _entropic_slow(x: ArrayLike) -> ArrayLike:
    first_term = 0.0
    for xi in x:
        if xi > 0:
            first_term += xi * torch.log(xi)
    
    second_term = 0.0
    one_minus_sum = 1 - torch.sum(x)
    if one_minus_sum > 0:
        second_term = one_minus_sum * torch.log(one_minus_sum)
    
    return torch.tensor([first_term + second_term])

def _entropic_grad_fenchel_dual(y: ArrayLike) -> ArrayLike:
    sum_exp_y = torch.sum(torch.exp(y))
    return torch.tensor([torch.exp(yj) / (1 + sum_exp_y) for yj in y])

def _entropic_grad(x: ArrayLike) -> ArrayLike:
    return torch.tensor([1 + torch.log(xi) for xi in x])

def test_evaluation():
    x = torch.tensor([0.1, 0.2, 0.3])
    assert torch.isclose(EntropicMirrorMap()(x), _entropic_slow(x)).all()

def test_grad():
    x = torch.tensor([0.1, 0.2, 0.3])
    assert torch.isclose(EntropicMirrorMap().grad(x), _entropic_grad(x)).all()

def test_grad_fenchel_dual():
    y = torch.tensor([0.1, 0.2, 0.3])
    
    assert torch.isclose(EntropicMirrorMap().grad_fenchel_dual(y), _entropic_grad_fenchel_dual(y)).all()

def test_invertibility():
    x = torch.tensor([0.1, 0.2, 0.3])
    y = EntropicMirrorMap().grad(x)

    x_reconstructed = EntropicMirrorMap().grad_fenchel_dual(y)
    assert torch.isclose(x, x_reconstructed).all()


if __name__ == "__main__":
    test_evaluation()
    test_grad()
    test_grad_fenchel_dual()
    test_invertibility()
