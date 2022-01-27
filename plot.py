import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Read data
data_path = "./data/angle_data.csv"
save_path = "./data/angle_plot.pdf"
df = pd.read_csv(data_path)

# Save figure
sns.set_theme(style="darkgrid")  # "whitgrid" also looks nice
fig = sns.catplot(
    data=df,
    x="init_scheme",
    y="angle",
    hue="param",
    kind="bar",
    height=5,
    aspect=1.5,
    alpha=0.9,
    legend_out=False,
    errwidth=2,
    capsize=0.05,
)
fig.set_axis_labels("", "angle")
fig.legend.set_title("params")
plt.savefig(save_path, format="pdf", bbox_inches="tight")
