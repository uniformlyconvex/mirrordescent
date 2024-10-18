import numpy as np
from numpy.typing import ArrayLike

import torch

from mirrordescent.mirror_maps import EntropicMirrorMap

def _entropic_mirror_map_slow(x: ArrayLike) -> ArrayLike:
    first_term = 0.0
    for xi in x:
        if xi > 0:
            first_term += xi * np.log(xi)
    
    second_term = 0.0
    one_minus_sum = 1 - np.sum(x)
    if one_minus_sum > 0:
        second_term = one_minus_sum * np.log(one_minus_sum)
    
    return np.array([first_term + second_term])

def _entropic_mirror_map_grad_fenchel_dual_slow(y: ArrayLike) -> ArrayLike:
    sum_exp_y = np.sum(np.exp(y))
    return np.array([np.exp(yj) / (1 + sum_exp_y) for yj in y])

def test_entropic_mirror_map():
    x = np.array([0.1, 0.2, 0.3, 0.4])
    
    assert np.isclose(EntropicMirrorMap()(x), _entropic_mirror_map_slow(x)).all()

def test_entropic_mirror_map_grad_fenchel_dual():
    y = torch.tensor([0.1, 0.2, 0.3, 0.4])
    
    assert np.isclose(EntropicMirrorMap().grad_fenchel_dual(y), _entropic_mirror_map_grad_fenchel_dual_slow(y)).all()


if __name__ == '__main__':
    # test_entropic_mirror_map()
    test_entropic_mirror_map_grad_fenchel_dual()