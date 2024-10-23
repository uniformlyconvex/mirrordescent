import json

import plotly.figure_factory as ff
import plotly.graph_objects as go
import numpy as np
import torch

import mirrordescent.experiments as exp

def plot_dist(dist: torch.distributions.distribution, samples: list[torch.Tensor]) -> go.Figure:
    """
    Makes a contour plot of the distribution and scatters the samples.
    """
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


def plot_dimension_experiment() -> None:
    NO_STDS = 2.0
    def color(opacity: float) -> str:
        return f'rgba(0, 100, 250, {opacity})'

    filename = exp.DimensionExperiment.RESULTS_FILE

    with open(filename, 'r') as f:
        results: dict[str, list[float]] = json.load(f)

    results: dict[int, list[float]] = {int(k): v for k, v in results.items()}
    results = {k: v for k, v in sorted(results.items())}
    results.pop(2, None)

    dims = list(results.keys())
    mean = [np.mean(results[dim]) for dim in dims]
    upper_std = [np.mean(results[dim]) + NO_STDS * np.std(results[dim]) for dim in dims]
    lower_std = [np.mean(results[dim]) - NO_STDS * np.std(results[dim]) for dim in dims]

    mean_trace = go.Scatter(
        x=dims,
        y=mean,
        mode='lines',
        name='Mean',
        line=dict(color=color(1.0))
    )
    upper_std_trace = go.Scatter(
        x=dims,
        y=upper_std,
        mode='lines',
        name='Mean + 2 std',
        line=dict(color=color(1.0), dash='dash')
    )
    lower_std_trace = go.Scatter(
        x=dims,
        y=lower_std,
        mode='lines',
        name='Mean - 2 std',
        fill='tonexty',
        fillcolor=color(0.2),
        line=dict(color=color(1.0), dash='dash')
    )

    fig = go.Figure(
        data=[mean_trace, upper_std_trace, lower_std_trace],
        layout=go.Layout(
            title='Time taken to sample from Dirichlet posterior',
            xaxis={'title': 'Dimension'},
            yaxis={'title': 'Time taken (s)'},
        )
    )

    fig.update_layout(
        xaxis_title=dict(font=dict(size=20)),
        yaxis_title=dict(font=dict(size=20)),
        xaxis=dict(tickfont=dict(size=18)),
        yaxis=dict(tickfont=dict(size=18)),
        legend=dict(font=dict(size=18)),
    )

    fig.show()


if __name__ == '__main__':
    plot_dimension_experiment()