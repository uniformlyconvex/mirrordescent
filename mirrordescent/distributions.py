from __future__ import annotations

import abc

import torch
import torch.distributions as dist

from mirrordescent.utils import rd_to_rdp1


class BaseDistribution(dist.Distribution, abc.ABC):
    @abc.abstractmethod
    def V(self, x: torch.Tensor) -> torch.Tensor:
        """
        The potential energy function.
        """
    
    def grad_V(self, x: torch.Tensor) -> torch.Tensor:
        return torch.autograd.functional.jacobian(self.V, x)
    
    @property
    @abc.abstractmethod
    def dim(self) -> int:
        """
        The dimension of the distribution.
        """

class DirichletPosterior(BaseDistribution, dist.Dirichlet):
    def add_observations(self, observations: torch.Tensor) -> DirichletPosterior:
        """
        Return the resulting Dirichlet posterior after adding observations.
        """
        if observations.shape != self.concentration.shape:
            raise ValueError(
                f"""
                Observations must have the same shape as the concentration
                (expected {self.concentration.shape}, got {observations.shape})
                """
            )
        return DirichletPosterior(
            concentration=self.concentration + observations,
            validate_args=self._validate_args
        )

    def V(self, x: torch.Tensor) -> torch.Tensor:
        x = rd_to_rdp1(x)
        # Please for god's sake just accept this normalisation
        x = x / torch.sum(x)
        x = torch.clamp(x, min=1e-6, max=1 - 1e-6)
        x = x / torch.sum(x)
        return -self.log_prob(x)
    
    @property
    def dim(self) -> int:
        return self.concentration.shape[0]