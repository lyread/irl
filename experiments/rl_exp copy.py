# Import necessary modules and classes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from likelihood_mcv import MultiCasesVisualizer
from util.data_process import get_threshold_plot_data_frames
'''''''''
# Data for line plots and box plots
with_pun = "experiments/examples/plot_data/cliff_walking/Q_LEARNING-pTrue_zero_e-greedy.pkl"
without_pun = "experiments/examples/plot_data/cliff_walking/Q_LEARNING-pFalse_zero_e-greedy.pkl"

df1 = pd.read_pickle(with_pun)
df2 = pd.read_pickle(without_pun)
# Data for Parallel Coordinates Plot
smac_output = "experiments/examples/plot_data/cliff_walking/smac_results_pT.pkl"
result = pd.read_pickle(smac_output)

data = [df1, df2]
cases = ["with punishment", "without punishment"]
mv = MultiCasesVisualizer(data, cases)
# Create figure
'''''''''


work_dir = "experiments/examples/plot_data/robot_arm/"
df_99 = pd.read_pickle(f"{work_dir}NAO4.pkl")
df_09 = pd.read_pickle(f"{work_dir}NAO41.pkl")

data = [ df_09, df_99]
cases = [ "0", "1"]


th = [70]
th_plot_data, th_cases = get_threshold_plot_data_frames(th, data, cases, add_best=True)
'''''''''
mv.single_line_plot(
    metric_x="Cumulated Training Steps",
    metric_y="Test Failure Rate",
    x_label="Training Steps",
    y_label="Failure Rate [%]",
    figsize=(16, 8),
    show_min=True,
    show_mean=True,
    show_percentage=True,
    line_width=4,
    test_every=1,
    x_fontsize=30,
    y_fontsize=30,
    legend_title="algorithm",
    legend_ncols=2,
    legend_loc="best",
    xscale="log",
    yscale="linear",
    save_plot=True,
    filename="line_plot",
    file_format="png",
)
'''''''''
nao_th_v = MultiCasesVisualizer(th_plot_data, th_cases)
# Get color map
cmap = plt.get_cmap("Blues")
colors = [cmap((i + 2) / len(th_cases)) for i in range(len(th_cases))]
nao_th_v.single_line_plot(
    metric_x="Ask Likelihood",
    metric_y="Failure Rate",
    colors=colors,
    figsize=(16, 8),
    line_width=4,
    show_mean=True,
    fill_between=True,
    x_fontsize=30,
    y_fontsize=30,
    y_lim=[-5, 100],
    x_ticks=[0.0, 0.1],
    y_ticks=[i for i in range(0, 101, 10)],
    x_label="Ask Likelihood",
    y_label="Failure Rate [%]",
    legend_title="Thresholds",
    legend_fontsize=20,
    legend_ncols=3,
    save_plot=True,
    foldername="plots/",
    filename="NAO41_ask_likelihood_failure_rate",
    file_format="png"
)
