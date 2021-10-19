from typing import Any, Sequence, Union, List
import numpy as np
from torch import Tensor
import plotly.graph_objects as go
from plotly.graph_objects import Figure
from plotly.subplots import make_subplots


def make_stacked_feedback_training_figure(
    all_values: List[Union[Tensor, np.ndarray, Sequence[Sequence[Any]]]],
    row_titles: List[str],
    title_text: str,
    layer_names: List[str] = None,
) -> Figure:
    """Creates a stacked plot that shows the evolution of different values during a step of
    feedback training.
    
    `all_values` should contain a sequence of list of lists. (a list of "metric_values").
    Each "metric_values" should contain the value of a metric, for each layer, for each iteration.
    `row_titles` should contain the name associated with each item in `all_values`.
    `title_text` is the name of the overall figure.
    """
    all_values = [
        [
            [v.cpu().numpy() if isinstance(v, Tensor) else v for v in layer_values]
            for layer_values in values
        ]
        for values in all_values
    ]

    n_layers = len(all_values[0])
    n_plots = len(all_values)
    layer_names = layer_names or [f"layer {i}" for i in range(n_layers)]
    assert len(row_titles) == n_plots
    # Each list needs to have the same number of items (i.e. same number of layers)
    assert all(len(values) == n_layers for values in all_values)

    fig: Figure = make_subplots(
        rows=n_plots,
        cols=n_layers,
        x_title="# of feedback training iterations",
        column_titles=layer_names,
        row_titles=[row_title for row_title in row_titles],
    )

    # Add traces
    for plot_id, values in enumerate(all_values):
        for layer_id in range(n_layers):
            layer_values = values[layer_id]
            x = np.arange(len(layer_values))
            trace = go.Scatter(x=x, y=layer_values)
            fig.add_trace(
                trace, row=plot_id + 1, col=layer_id + 1,
            )

    # Add figure title
    fig.update_layout(
        title_text=title_text, showlegend=False,
    )

    for i, row_title in enumerate(row_titles):
        # Set y-axes titles (only for the first column)
        fig.update_yaxes(title_text=row_title, row=i + 1, col=1)
        # Set a fixed range on the y axis for that row:
        if "angle" in row_title.lower():
            fig.update_yaxes(row=i + 1, range=[0, 90], fixedrange=True)

    return fig
