import abc
import typing as t

import numpy as np
import numpy.typing as npt
import torch

class MirrorMap(abc.ABC):
    """
    Abstract base class for mirror maps (in case we want to consider other mirror maps).
    """
    @abc.abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the mirror map at x."""

    @abc.abstractmethod
    def fenchel_dual(y: torch.Tensor) -> torch.Tensor:
        """Returns the dual of h evaluated at y"""

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the gradient of h evaluated at x"""
        return torch.autograd.functional.jacobian(self, x)

    def grad_fenchel_dual(self, y: torch.Tensor) -> torch.Tensor:
        """Returns the gradient of the dual of h evaluated at y"""
        return torch.autograd.functional.jacobian(self.fenchel_dual, y)
    
    def hessian(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the Hessian of h evaluated at x"""
        return torch.autograd.functional.hessian(self, x)
    
    def _log_det_hessian(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the log determinant of the Hessian of h evaluated at x"""
        return torch.logdet(self.hessian(x))
    
    def grad_log_det_hessian(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the gradient of the log determinant of the Hessian of h evaluated at x"""
        return torch.autograd.functional.jacobian(self._log_det_hessian, x)


class EntropicMirrorMap(MirrorMap):
    """
    The entropic mirror map; see eqn. (3.4) in "Mirrored Langevin Dynamics".
    """
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        def _sum_x_log_x(arr: torch.tensor) -> torch.tensor:
            # Sum xlogx with the convention that 0log0 = 0 
            return torch.sum(torch.where(arr > 0, arr * torch.log(arr), 0))
        
        first_term = _sum_x_log_x(x)
        second_term = _sum_x_log_x(torch.tensor([1-torch.sum(x)]))
        return first_term + second_term
    
    @staticmethod
    def fenchel_dual(y: torch.Tensor) -> torch.Tensor:
        """Returns the dual of h evaluated at y"""
        sum_exp_y = torch.sum(torch.exp(y))
        return torch.log(1 + sum_exp_y)