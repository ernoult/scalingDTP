import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.express as px

# Read data
data_path = "./data/data.csv"
save_path = "./data/plot.pdf"
df = pd.read_csv(data_path)

# # Save figure
# sns.set_theme(style="darkgrid")  # "whitgrid" also looks nice
# fig = sns.catplot(
#     data=df,
#     x="init_scheme",
#     y="angle",
#     hue="param",
#     kind="bar",
#     height=5,
#     aspect=1.5,
#     alpha=0.9,
#     legend_out=False,
#     errwidth=2,
#     capsize=0.05,
# )
# fig.set_axis_labels("", "angle")
# fig.legend.set_title("params")
# plt.savefig(save_path, format="pdf", bbox_inches="tight")

indices = []
values = []


DRL_RANDOM = r"$\Large{\text{DRL}_{\text{random}}}$"
DRL = r"$\Large{\text{DRL}}$"
L_DRL_RANDOM = r"$\Large{\text{L-DRL}_{\text{random}}}$"
L_DRL = r"$\Large{\text{L-DRL}}$"
L_DRL_SYM = r"$\Large{\text{L-DRL}_{\text{sym}}}$"


rename_dict = {
    "random init": L_DRL_RANDOM,
    "symmetric init": L_DRL_SYM,
    "l-drl init": L_DRL,
    "drl init": DRL_RANDOM,
    "drl": DRL,
}
for seed, init_scheme, param, angle, distance in df.values:
    # print(seed, init_scheme, param, angle, distance)
    model = rename_dict[init_scheme]
    indices.append((model, seed, param))
    values.append((distance, angle))
df = pd.DataFrame(
    values,
    index=pd.MultiIndex.from_tuples(indices, names=["model", "seed", "parameter"]),
    columns=pd.Index(["distance", "angle"], name="metric"),
)

gdf = df.groupby(level=("model", "parameter"), sort=True)
df = pd.concat(
    [
        gdf.mean().rename(lambda c: f"{c}", axis="columns"),
        gdf.std().rename(lambda c: f"{c}_std", axis="columns"),
        gdf.count().rename(lambda c: f"{c}_count", axis="columns"),
    ],
    axis="columns",
)

# # rename the names of the models:
# df = df.rename(
#     {
#         "DTP_symmetric": r"$\text{L-DRL}_{\text{sym}}$",
#         "DTP_untrained": r"$\text{L-DRL}_{\text{init}}$",
#         "DTP": r"$\text{L-DRL}$",
#         "Meulemans-DTP": r"$\text{DRL}$",
#         "Meulemans-DTP_untrained": r"$\text{DRL}_{\text{init}}$",
#     },
# )
# TODO: Don't include the bias terms.
parameters = df.index.unique("parameter")
bias_params = [p for p in parameters if "bias" in p]
df = df.drop(labels=bias_params, level="parameter")
df = df.reset_index()

from plotly import graph_objects as go

angles_fig: go.Bar = px.bar(
    df,
    x="model",
    y="angle",
    error_y="angle_std",
    barmode="group",
    color="parameter",
    # title="Angle between DTP and Backprop Updates",
    # color_discrete_map={"cold": "blue", "warm": "orange", "hot": "red"},
    # points="all",
    width=1000,
    height=500,
    category_orders={
        "model": [
            DRL_RANDOM,
            L_DRL_RANDOM,
            DRL,
            L_DRL,
            L_DRL_SYM,
        ]
    },
)
print(angles_fig.layout)


distances_fig = px.bar(
    df,
    x="model",
    y="distance",
    error_y="distance_std",
    barmode="group",
    color="parameter",
    # title="Distances between DTP updates and Backprop Updates",
    # color_discrete_map={"cold": "blue", "warm": "orange", "hot": "red"},
    # points="all",
)
for fig in [distances_fig, angles_fig]:
    fig.update_layout(
        showlegend=False,
        font_family="Serif",
        title_font_family="Serif",
        font_size=24,
        xaxis_title_text="",
        xaxis_tickfont_size=40,
        # xaxis_tickangle=4,
        # xaxis_
        # font_color="blue",
        # title_font_family="Serif",
        # title_font_color="red",
        # legend_title_font_color="green",
    )
    # print(fig.layout)
angles_fig.update_layout(
    title_x=0.5,
    title_xanchor="center",
    margin_r=10,
    margin_l=100,
    margin_t=10,
    margin_b=0,
    yaxis_title_text=r"$\Large{\angle(\Delta \theta^n_{\rm BP}, \Delta \theta^n_{\rm DTP})}$",
)
print(angles_fig.layout)
distances_fig.update_layout(
    yaxis_title_text=r"$\Large{d(\Delta \theta^n_{\rm BP}, \Delta \theta^n_{\rm DTP})}$",
)
from pathlib import Path

figures_dir = Path("final_figures")
figures_dir.mkdir(exist_ok=True)
# assert False, angles_fig.layout.width
angles_fig.write_image(
    figures_dir / "figure_4_3-angles.png",
    # width=angles_fig.layout.width,
    # height=angles_fig.layout.height,
    scale=3,
)

distances_fig.write_image(
    figures_dir / "figure_4_3-distances.png",
    width=distances_fig.layout.width,
    height=distances_fig.layout.height,
    scale=3,
)
# angles_fig.write_image(figures_dir / "figure_4_3-angles.png")
# distances_fig.show()
# angles_fig.show()
# return angles_fig, distances_fig
