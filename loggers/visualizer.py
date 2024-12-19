# Import necessary modules
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.io as pio
import seaborn as sns
from scipy.signal import savgol_filter

import algorithms.utils.file_manipulation as fm


# Define a class for visualizing data
class Visualizer:
    def __init__(self, data, wd="/tmp/"):
        # Initialize the visualizer with data and available metrics
        self.data = data  # A DataFrame containing the data
        self.metrics = data.columns.to_list()  # List of available metrics in the data
        self.set_working_directory(wd=wd)

    # Define a method to load data for a specific metric
    def load_data(self, metric):
        assert metric in self.metrics, f"{metric} is not a valid name!"
        return self.data[metric].to_list()

    def set_working_directory(self, wd="/tmp/"):
        self.working_dir = fm.standardize_folder(wd)  # Trailing slash
        self.target_dir = fm.standardize_folder(self.working_dir + "plot_data")

    def export_plot(
        self, foldername=None, filename="", file_format="pdf", use_pio=False, fig=None
    ):
        if foldername == None:
            folder = fm.create_folder(self.working_dir)
        folder = fm.create_folder(foldername)
        filename = fm.create_filename(
            folder, filename=filename, file_format=file_format, time_stampt=False
        )
        if use_pio:
            pio.write_image(fig, filename, file_format)
        else:
            plt.savefig(filename, bbox_inches="tight")

    def create_heat_map(
        self,
        metric,
        figsize=(5, 5),
        vmax=-7,
        vmin=0,
        square=True,
        annot=False,
        cmap="bwr",
        linewidths=0.5,
        linecolor="black",
        cbar_kws={"shrink": 1},
        title="",
        save_plot=False,
        foldername="plot/",
        filename="test",
        file_format="pdf",
    ):
        plt.figure(figsize=figsize)
        arr_ls = self.load_data(metric)
        # compute average value of the matrix if more than one are given
        data = sum(arr_ls) / len(arr_ls)
        if data.shape != (4, 12):
            data = np.max(data, axis=-1)
            # data = np.mean(data, axis=-1)
        sns.heatmap(
            data,
            vmax=vmax,
            vmin=vmin,
            square=square,
            annot=annot,
            cmap=cmap,
            linewidths=linewidths,
            linecolor=linecolor,
            cbar_kws=cbar_kws,
        )
        ax = plt.gca()
        # Creat cliff and start and target layout
        start = [0, 3]
        target = [11, 3]
        ax.text(
            start[0] + 0.5,
            start[1] + 0.5,
            "S",
            ha="center",
            va="center",
            color="black",
            size=20,
        )
        ax.text(
            target[0] + 0.5,
            target[1] + 0.5,
            "T",
            ha="center",
            va="center",
            color="black",
            size=20,
        )
        start = patches.Rectangle(
            xy=(start[0], start[1]),
            width=1,
            height=1,
            linewidth=linewidths,
            edgecolor="black",
            facecolor="yellow",
        )
        target = patches.Rectangle(
            xy=(target[0], target[1]),
            width=1,
            height=1,
            linewidth=0.5,
            edgecolor="black",
            facecolor="green",
        )
        cliff = patches.Rectangle(
            xy=(1, 3),
            width=10,
            height=1,
            linewidth=0.5,
            edgecolor="black",
            facecolor="white",
        )
        ax.add_patch(target)
        ax.add_patch(start)
        ax.add_patch(cliff)
        # Make x and y ticks invisible
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        # Show right and bottom edge
        ax.spines["right"].set_visible(True)
        ax.spines["bottom"].set_visible(True)
        # Set title
        ax.set_title(title)
        # Save or show plot
        if save_plot:
            if foldername == None:
                folder = fm.create_folder(self.working_dir)
            folder = fm.create_folder(foldername)
            filename = fm.create_filename(
                folder, filename=filename, file_format=file_format, time_stampt=False
            )
            box_plot = ax.get_figure()
            box_plot.savefig(filename, bbox_inches="tight")
        else:
            plt.show()

    def normalize_score(self, best, worst):
        self.data["normalized_score"] = [
            1 - (value - best) / (worst - best) for value in self.data["score"]
        ]

    def create_parallel_coordinate_plot(
        self,
        best=12 * 8000 * 30,
        worst=129 * 8000 * 30,
        dimensions=None,
        save_plot=False,
        foldername="plot/",
        filename="parallel_coordinate_plot",
        file_format="pdf",
    ):
        self.normalize_score(best, worst)
        fig = px.parallel_coordinates(
            self.data,
            dimensions=dimensions,
            color="normalized_score",
            range_color=[0.5, 1],
            color_continuous_scale=px.colors.sequential.Viridis,
        )
        if save_plot:
            self.export_plot(
                foldername=foldername,
                filename=filename,
                file_format=file_format,
                use_pio=True,
                fig=fig,
            )
        fig.show()

    def create_histogram(
        self,
        metric="distance",
        xlabel="Distance(m)",
        ylabel="Number of Samples",
        color="gray",
        x_range=None,
        save_plot=False,
        foldername="plot/",
        filename="parallel_coordinate_plot",
        file_format="pdf",
    ):
        data = self.load_data(metric)
        ax = plt.subplot()
        plt.minorticks_on()
        ax.tick_params(which="both", top=True, right=True, direction="in", labelsize=15)
        ax.hist(data, bins=25, range=x_range, color=color, edgecolor="black")
        ax.set_xlabel(xlabel, fontsize=20)
        ax.set_ylabel(ylabel, fontsize=20)
        if save_plot:
            self.export_plot(
                foldername=foldername, filename=filename, file_format=file_format
            )
        else:
            plt.show()
