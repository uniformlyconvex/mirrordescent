import plotly.figure_factory as ff
import plotly.graph_objects as go
import numpy as np
import torch

def plot_dist(dist: torch.distributions.distribution, samples: list[torch.Tensor]) -> go.Figure:
    x1s = torch.linspace(0, 1, 100)
    x2s = torch.linspace(0, 1, 100)
    
    X1, X2 = torch.meshgrid(x1s, x2s)
    X1, X2 = X1.ravel(), X2.ravel()
    mask = X1 + X2 <= 1
    X1, X2 = X1[mask], X2[mask]

    X3 = 1 - X1 - X2

    new_mask = (X1 > 0) & (X2 > 0) & (X3 > 0)
    X1, X2, X3 = X1[new_mask], X2[new_mask], X3[new_mask]

    X = torch.stack((X1, X2, X3), axis=1)
    X = X / X.sum(axis=1, keepdim=True)
    X = torch.clamp(X, min=1e-10, max=1-1e-10)

    densities = dist.log_prob(X).exp()

    X1, X2, X3 = X1.numpy(), X2.numpy(), X3.numpy()

    fig = ff.create_ternary_contour(
        np.array([X1, X2, X3]),
        densities.numpy(),
        pole_labels=['$x_1$', '$x_2$', '$x_3$'],
        ncontours=20,
        coloring='lines',
    )

    for sample in samples:
        fig.add_trace(go.Scatterternary({
            'mode': 'markers',
            'a': sample[0].numpy(),
            'b': sample[1].numpy(),
            'c': sample[2].numpy(),
            'marker': {
                'symbol': 100,  # Marker symbol
                'color': 'black',  # Marker color
                'size': 3,  # Marker size
                'line': {'width': 2, 'color': 'black'}  # Marker border
            },
            'legendgroup': 'Samples',
        }))

    fig.update_layout(
        title='Posterior distribution',
        ternary=dict(
            sum=1,
            aaxis=dict(title='$x_1$', ticks='outside'),
            baxis=dict(title='$x_2$', ticks='outside'),
            caxis=dict(title='$x_3$', ticks='outside'),
        )
    )

    return fig