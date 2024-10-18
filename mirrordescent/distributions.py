import typing as t
from functools import cached_property

import torch

from mirrordescent.utils import gamma

class DirichletPosterior:
    def __init__(
        self,
        alphas: torch.Tensor,
        observations: torch.Tensor
    ):
        self._alphas = alphas
        self._observations = observations

    @cached_property
    def _normalising_constant(self) -> torch.Tensor:
        numerator = torch.prod(gamma(self._alphas + self._observations))
        denominator = gamma(torch.sum(self._alphas + self._observations))
        return numerator / denominator

    def density(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] != self._alphas.shape[0] - 1:
            raise ValueError("x must have the same dimension as alphas - 1")

        x = torch.cat([x, 1 - torch.sum(x)])

        if any(x < 0):
            return torch.tensor(0.0)
        
        pows = x ** (self._observations + self._alphas - 1) # element-wise power
        return torch.prod(pows) / self._normalising_constant

    def V(self, x: torch.Tensor) -> torch.Tensor:
        return -torch.log(self.density(x))