import abc

import torch

from mirrordescent.utils import rd_to_rdp1

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
    @staticmethod
    def _sum_x_log_x(x: torch.Tensor) -> torch.Tensor:
        # Sum xlogx with the convention that 0log0 = 0
        return torch.sum(
            torch.where(
                x > 0,
                x * torch.log(x),
                0
            )
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the entropic mirror map to x.
        """
        # x is such that sum_i x_i <= 1, so we need to append 1 - sum_i x_i to x
        x = rd_to_rdp1(x)
        return self._sum_x_log_x(x)
    
    @staticmethod
    def fenchel_dual(y: torch.Tensor) -> torch.Tensor:
        """Returns the dual of h evaluated at y"""
        sum_exp_y = torch.sum(torch.exp(y))
        return torch.log(1 + sum_exp_y)