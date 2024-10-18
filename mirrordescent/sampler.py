import typing as t

import torch

from mirrordescent.mirror_maps import MirrorMap

class MLD:
    def __init__(
        self,
        start_x_point: torch.Tensor,
        mirror_map: MirrorMap,
        step_sizes: t.Callable[[int], float],
        grad_V: t.Callable[[torch.Tensor], torch.Tensor],
    ):
        start_y_point = mirror_map.grad(start_x_point)
        
        self._mirror_map = mirror_map
        self._step_sizes = step_sizes
        self._mirror_iterates: t.List[torch.Tensor] = [start_y_point]
        self._grad_V = grad_V

    def _step(self):
        y_t = self._mirror_iterates[-1]
        x_t = self._mirror_map.grad_fenchel_dual(y_t)
        beta_t = self._step_sizes(len(self._mirror_iterates))

        hessian = self._mirror_map.hessian(x_t)
        hessian_inverse = hessian.inverse() if hessian.shape else hessian.pow(-1)
        grad_V = self._grad_V(x_t)
        grad_log_det_hessian = self._mirror_map.grad_log_det_hessian(x_t)
        
        deterministic_term = -beta_t * hessian_inverse @ (grad_V + grad_log_det_hessian)
        gaussian = torch.distributions.MultivariateNormal(
            loc=torch.zeros_like(deterministic_term),
            covariance_matrix=torch.eye(deterministic_term.shape[0]),
        )
        noise_term = torch.sqrt(torch.tensor(2 * beta_t)) * gaussian.sample()

        y_tp1 = deterministic_term + noise_term
        self._mirror_iterates.append(y_tp1)

    def get_samples(self, no_samples: int) -> list[torch.Tensor]:
        while len(self._mirror_iterates) < no_samples:
            self._step()

        iterates = [self._mirror_map.grad_fenchel_dual(yt) for yt in self._mirror_iterates]
        return iterates








