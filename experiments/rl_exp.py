import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from likelihood_mcv import MultiCasesVisualizer
from util.data_process import get_threshold_plot_data_frames


work_dir = "experiments/examples/plot_data/robot_arm/"
df_99 = pd.read_pickle(f"{work_dir}NAO4-ask99.pkl")
df_09 = pd.read_pickle(f"{work_dir}NAO4-ask09.pkl")
df_08 = pd.read_pickle(f"{work_dir}NAO4-ask08.pkl")
df_07 = pd.read_pickle(f"{work_dir}NAO4-ask07.pkl")
df_06 = pd.read_pickle(f"{work_dir}NAO4-ask06.pkl")
df_05 = pd.read_pickle(f"{work_dir}NAO4-ask05.pkl")
df_04 = pd.read_pickle(f"{work_dir}NAO4-ask04.pkl")
df_03 = pd.read_pickle(f"{work_dir}NAO4-ask03.pkl")
df_02 = pd.read_pickle(f"{work_dir}NAO4-ask02.pkl")
df_01 = pd.read_pickle(f"{work_dir}NAO4-ask01.pkl")
df_00 = pd.read_pickle(f"{work_dir}NAO4-ask00.pkl")
data = [df_00, df_01, df_02, df_03, df_04, df_05, df_06, df_07, df_08, df_09, df_99]
cases = ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "0.99"]


# Get color map
# Extract Threshold values from original data
th = [70, 50, 30, 20, 15]
th_plot_data, th_cases = get_threshold_plot_data_frames(th, data, cases, add_best=True)
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
    x_ticks=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99],
    y_ticks=[i for i in range(0, 101, 10)],
    x_label="Ask Likelihood",
    y_label="Failure Rate [%]",
    legend_title="Thresholds",
    legend_fontsize=20,
    legend_ncols=3,
    save_plot=True,
    foldername="plots/",
    filename="NAO4_ask_likelihood_failure_rate",
    file_format="png"
)
