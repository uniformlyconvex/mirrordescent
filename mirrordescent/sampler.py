import multiprocessing as mp
import typing as t

import torch

from mirrordescent.distributions import BaseDistribution
from mirrordescent.mirror_maps import MirrorMap

class MLD:
    def __init__(
        self,
        mirror_map: MirrorMap,
        step_sizes: t.Callable[[int], float],
        dist: BaseDistribution
    ):
        
        self._mirror_map = mirror_map
        self._step_sizes = step_sizes
        self._dist = dist

    def _sample_once(self, i: int) -> torch.Tensor:
        # Get a random start point in the unit simplex
        uniform = torch.distributions.Dirichlet(torch.ones(self._dist.dim))
        x0 = uniform.sample()[:-1]

        # Run MLD
        iterates = [x0]
        mirror_iterates = [self._mirror_map.grad(x0)]
        for _ in range(50):
            x_t = iterates[-1]
            y_t = mirror_iterates[-1]
            # x_t = self._mirror_map.grad_fenchel_dual(y_t)

            beta_t = self._step_sizes(len(mirror_iterates))

            hessian = self._mirror_map.hessian(x_t)
            hessian_inverse = hessian.inverse() if hessian.shape else hessian.pow(-1)
            grad_V = self._dist.grad_V(x_t)
            grad_log_det_hessian = self._mirror_map.grad_log_det_hessian(x_t)

            deterministic_term = - beta_t * hessian_inverse @ (grad_V + grad_log_det_hessian)
            gaussian = torch.distributions.MultivariateNormal(
                loc=torch.zeros_like(deterministic_term),
                covariance_matrix=torch.eye(deterministic_term.shape[0]),
            )
            noise_term = torch.sqrt(torch.tensor(2 * beta_t)) * gaussian.sample()

            y_tp1 = y_t + deterministic_term + noise_term
            x_tp1 = self._mirror_map.grad_fenchel_dual(y_tp1)
            iterates.append(x_tp1)
            mirror_iterates.append(y_tp1)
        
        print(f'Got sample {i}')
        return iterates[-1]

    def get_samples(self, no_samples: int) -> list[torch.Tensor]:
        ctx = mp.get_context("spawn")
        with ctx.Pool() as pool:
            samples = pool.map(self._sample_once, range(no_samples))
        
        return samples







