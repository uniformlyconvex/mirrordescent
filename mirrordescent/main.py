import torch

import mirrordescent.distributions as dists
import mirrordescent.mirror_maps as mm
import mirrordescent.sampler as sampler
import mirrordescent.plotting as plotting
import mirrordescent.utils as utils

def step_sizes(t):
        return t ** -1


def demo_sampling():
    dist = dists.DirichletPosterior(
        concentration=torch.tensor([2.0, 4.0, 4.0])
    )

    mld = sampler.MLD(
        mirror_map=mm.EntropicMirrorMap(),
        step_sizes=step_sizes,
        dist=dist
    )

    samples = mld.get_samples(500)
    samples = [utils.rd_to_rdp1(s) for s in samples]

    fig = plotting.plot_dist(dist, samples)
    fig.show()
    

if __name__ == '__main__':
    plotting.plot_dimension_experiment()