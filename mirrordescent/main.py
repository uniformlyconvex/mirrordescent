import torch

import mirrordescent.distributions as dists
import mirrordescent.mirror_maps as mm
import mirrordescent.sampler as sampler


dist = dists.DirichletPosterior(
    alphas=torch.Tensor([0.1,0.1]),
    observations=torch.Tensor([100,1])
)

mld = sampler.MLD(
    start_x_point=torch.tensor([0.1]),
    mirror_map=mm.EntropicMirrorMap(),
    step_sizes=lambda _: 0.01,
    grad_V=lambda x: torch.autograd.functional.jacobian(dist.V, x)
)

mld.get_samples(100)