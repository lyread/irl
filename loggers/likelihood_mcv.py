import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import savgol_filter
from sklearn.metrics import auc

import util.file_manipulation as fm


class MultiCasesVisualizer:
    def __init__(self, datas: list[pd.DataFrame], cases: list[str], wd: str = "/tmp/"):
        self.data = datas  # List of dataframes
        self.metrics = datas[0].columns.to_list()  # List of available metrics
        self.cases = cases  # List of case labels (e.g., algorithm names)
        self.set_working_directory(wd=wd)

    def set_working_directory(self, wd="/tmp/"):
        self.working_dir = fm.standardize_folder(wd)  # Trailing slash
        self.target_dir = fm.standardize_folder(self.working_dir + "plot_data")

    def load_data(self, data, metric):
        assert metric in self.metrics, f"{metric} is not a valid name!"
        tmp = data[
            metric
        ].to_list()  # to_numpy() returns an array of list objects instead of a 2D-array
        return np.array(tmp)

    def compute_auc(self, data):
        """
        Compute the Area Under the Curve (AUC) for a given dataset.

        Args:
            data (np.ndarray): Performance metric data.

        Returns:
            float: AUC value.
        """
        length = [index for index in range(len(data))]
        auc_value = auc(length, data)
        return auc_value

    def get_threshold_points(
        self,
        metric_x: str,
        metric_y: str,
        thresholds: list[int | float],
        show_mean: bool = False,
        show_median: bool = False,
        show_percentage: bool = False,
    ):
        """
        Get threshold points for a given metric.

        Args:
            metric_x (str): X-axis metric.
            metric_y (str): Y-axis metric.
            thresholds (list[int | float]): List of thresholds.
            show_mean (bool, optional): Whether to show mean. Defaults to False.
            show_median (bool, optional): Whether to show median. Defaults to False.
            show_percentage (bool, optional): Whether to show values as percentage. Defaults to False.

        Returns:
            tuple: Threshold X and Y points.
        """
        t_x, t_y = [], []
        for case_index in range(len(self.cases)):
            y_values, *_ = self.get_y_values(
                metric_y, case_index, show_mean, show_median, show_percentage
            )
            x_values = self.get_x_values(metric_x, y_values, case_index)
            x, y = [], []
            for i, threshold in enumerate(thresholds):
                for index, value in enumerate(y_values):
                    if value <= threshold:
                        x.append(x_values[index])
                        y.append(value)
                        break  # If the threshold is not reached fill with inf
                if len(x) < i + 1:
                    print(
                        f"Warning: Threshold: {threshold} not reached for case {self.cases[case_index]}"
                    )
                    print(f"Filling with inf")
                    x.append(np.inf)
                    y.append(np.inf)
            t_x.append(x)
            t_y.append(y)
        return t_x, t_y

    def get_x_values(
        self, metric: str, y: np.ndarray, case_index: int, test_every: int = 1
    ):
        """
        Get X-axis values for plotting.

        Args:
            metric (str): Metric name.
            y (np.ndarray): Y-axis values.
            case_index (int): Index of the case.
            test_every (int, optional): Test frequency. Defaults to 1.

        Returns:
            list: X-axis values.
        """
        if metric in self.metrics:
            x = self.load_data(self.data[case_index], metric)
            return x if x.ndim == 1 else np.mean(x, axis=0)
        elif metric == "Training Episodes":
            return [x * test_every for x in range(len(y))]
        else:
            return [x for x in range(len(y))]

    def get_y_values(
        self,
        metric: str,
        case_index: int,
        show_mean: bool,
        show_median: bool,
        show_percentage: bool = False,
    ):
        """
        Get Y-axis values for plotting.

        Args:
            metric (str): Metric name.
            case_index (int): Index of the case.
            show_mean (bool): Whether to show mean.
            show_median (bool): Whether to show median.
            show_percentage (bool, optional): Whether to show values as percentage. Defaults to False.

        Returns:
            tuple: Y-axis values, upper shadow, lower shadow.
        """
        tmp = self.load_data(self.data[case_index], metric)
        if show_percentage:
            tmp *= 100
        if tmp.ndim == 1:
            y = tmp
            shadow_low = None
            shadow_up = None
            return y, shadow_up, shadow_low

        data_mean = np.mean(tmp, axis=0)
        data_median = np.median(tmp, axis=0)
        data_std = np.std(tmp, axis=0)
        if show_mean:
            y = data_mean
            shadow_up = data_mean + data_std
            shadow_low = data_mean - data_std
        elif show_median:
            y = data_median
            shadow_up = np.quantile(tmp, 0.75, axis=0)
            shadow_low = np.quantile(tmp, 0.25, axis=0)
        return y, shadow_up, shadow_low

    def handle_kwargs(self, ax, **kwargs):
        """
        Handle additional keyword arguments for customizing plots.

        Args:
            ax (AxesSubplot): Axes object.
            **kwargs: Additional keyword arguments. See: single_line_plot().
        """
        if kwargs.get("y_lim"):
            ax.set_ylim(kwargs["y_lim"])
        if kwargs.get("x_lim"):
            ax.set_xlim(kwargs["x_lim"])
        if kwargs.get("xscale") == "log":
            plt.xscale("log")
        if kwargs.get("yscale") == "log":
            plt.yscale("log")
        if kwargs.get("hline"):
            try:
                plt.axhline(y=kwargs["hline"], ls="--", c="k", alpha=0.2)
            except:
                for hline in kwargs["hline"]:
                    plt.axhline(y=hline, ls="--", c="k", alpha=0.2)
        if kwargs.get("vline"):
            try:
                plt.axvline(x=kwargs["vline"], ls="--", c="k", alpha=0.2)
            except:
                for vline in kwargs["vline"]:
                    plt.axvline(x=vline, ls="--", c="k", alpha=0.2)
        if kwargs.get("x_ticks"):
            ax.xaxis.set_ticks(kwargs["x_ticks"])
        if kwargs.get("y_ticks"):
            ax.yaxis.set_ticks(kwargs["y_ticks"])
        if kwargs.get("mark_points"):
            mark_points = kwargs.get("mark_points", [])
            markers = kwargs.get("marker", [])
            marker_size = kwargs.get("marker_size", 100)
            marker_colors = kwargs.get("marker_colors", [])
            for i, point in enumerate(mark_points):
                marker = markers[i] if i < len(markers) else "v"
                color = marker_colors[i] if i < len(marker_colors) else "k"
                plt.scatter(*point, marker=marker, s=marker_size, color=color, zorder=2)

    # Define a method to create a line plot for a specific metric
    def single_line_plot(
        self,
        metric_x,
        metric_y,
        colors,
        show_mean=False,
        show_median=False,
        show_max=False,
        show_min=False,
        show_percentage=False,
        fill_between=False,
        test_every=1,
        figsize=None,
        x_label=None,
        y_label=None,
        x_fontsize=None,
        y_fontsize=None,
        legend_fontsize=None,
        legend_title=None,
        line_width=None,
        smooth=False,
        window_length=11,
        poly_order=4,
        save_plot=False,
        foldername="plots/",
        filename="",
        file_format="pdf",
        **kwargs,
    ):
        """
        Create a single line plot for a specific metric.

        Args:
            metric_x (str): X-axis metric.
            metric_y (str): Y-axis metric.
            colors (list): List of colors for different cases.
            show_mean (bool, optional): Whether to show mean. Defaults to False.
            show_median (bool, optional): Whether to show median. Defaults to False.
            show_max (bool, optional): Whether to show maximum value. Defaults to False.
            show_min (bool, optional): Whether to show minimum value. Defaults to False.
            show_percentage (bool, optional): Whether to show values as percentage. Defaults to False.
            fill_between (bool, optional): Whether to fill between curves. Defaults to False.
            test_every (int, optional): Test frequency. If you want show training episode in x axis.
            figsize (tuple, optional): Figure size. Defaults to None.
            x_label (str, optional): X-axis label. Defaults to None.
            y_label (str, optional): Y-axis label. Defaults to None.
            x_fontsize (int, optional): X-axis label font size. Defaults to None.
            y_fontsize (int, optional): Y-axis label font size. Defaults to None.
            legend_fontsize (int, optional): Legend font size. Defaults to None.
            legend_title (str, optional): Legend title. Defaults to None.
            line_width (int, optional): Line width. Defaults to None.
            smooth (bool, optional): Whether to apply smoothing. Defaults to False.
            window_length (int, optional): Smoothing window length. Defaults to 11.
            poly_order (int, optional): Polynomial order for smoothing. Defaults to 4.
            save_plot (bool, optional): Whether to save the plot. Defaults to False.
            foldername(str, optional): Name of the folder to save the plot. Defaults to "plots/".
            filename (str, optional): Name of the file. Defaults to "".
            file_format (str, optional): File format for saving the plot. Defaults to "pdf".
            **kwargs: Additional keyword arguments for customizing the plot:

                y_lim (tuple, optional): Tuple specifying the lower and upper limits of the y-axis.
                x_lim (tuple, optional): Tuple specifying the lower and upper limits of the x-axis.
                xscale (str, optional): Scale for the x-axis, e.g., "linear" or "log". Defaults to None.
                yscale (str, optional): Scale for the y-axis, e.g., "linear" or "log". Defaults to None.
                hline (float or list[float], optional): Horizontal line(s) to be plotted across the axes.
                    Can be a single float or a list of floats. Defaults to None.
                vline (float or list[float], optional): Vertical line(s) to be plotted across the axes.
                    Can be a single float or a list of floats. Defaults to None.
                x_ticks (list, optional): List of tick locations for the x-axis. Defaults to None.
                y_ticks (list, optional): List of tick locations for the y-axis. Defaults to None.
                legend_loc (str, optional): Location of the legend. Defaults to "best".
                legend_ncols (int, optional): Number of columns in the legend. Defaults to None.
                mark_points (list[tuple], optional): List of points to be marked on the plot.
                    Each point is specified as a tuple (x, y). Defaults to None.
                marker (list[str], optional): List of marker styles for marking the points. Defaults to None.
                marker_size (int, optional): Size of the marker. Defaults to 100.
                marker_colors (list[str], optional): List of colors for the markers. Defaults to None.
        """
        assert not (
            show_mean and show_median
        ), "Can't show both mean and median at the same time!"
        fill_between = fill_between
        plt.figure(figsize=figsize)
        ax = plt.subplot()
        xlabel = x_label if x_label else metric_x
        ylabel = y_label if y_label else metric_y
        ax.set_xlabel(xlabel, fontsize=x_fontsize)
        ax.set_ylabel(ylabel, fontsize=y_fontsize)
        ax.tick_params(which="both", top=True, right=True, direction="in", labelsize=25)
        if show_max:
            max_x, max_y = [], []
        if show_min:
            min_x, min_y = [], []
        self.handle_kwargs(ax, **kwargs)
        for case_index, case in enumerate(self.cases):
            y, shadow_up, shadow_low = self.get_y_values(
                metric_y, case_index, show_mean, show_median, show_percentage
            )
            x = self.get_x_values(metric_x, y, case_index, test_every)
            if smooth:
                y = savgol_filter(y, window_length, poly_order)
                shadow_low = savgol_filter(shadow_low, window_length, poly_order)
                shadow_up = savgol_filter(shadow_up, window_length, poly_order)
            if show_max:
                max_x.append[x[np.argmax(y)]]
                max_y.append[np.max(y)]
            if show_min:
                min_x.append(x[np.argmin(y)])
                min_y.append(np.min(y))
            ax.plot(
                x,
                y,
                label=case,
                color=colors[case_index],
                linewidth=line_width,
                alpha=1,
                zorder=0,
            )
            if fill_between:
                ax.fill_between(
                    x,
                    shadow_up,
                    shadow_low,
                    color=colors[case_index],
                    alpha=0.1,
                    zorder=0,
                )
        if show_max:
            ax.scatter(
                max_x,
                max_y,
                marker="o",
                s=300,
                c=colors,
                alpha=1,
                edgecolors="w",
                linewidths=2,
                zorder=1,
            )
        if show_min:
            ax.scatter(
                min_x,
                min_y,
                marker="o",
                s=300,
                c=colors,
                alpha=1,
                edgecolors="w",
                linewidths=2,
                zorder=1,
            )
        legend_loc = kwargs.get("legend_loc", "best")
        legend = plt.legend(
            fontsize=legend_fontsize,
            loc=legend_loc,
            ncols=kwargs.get("legend_ncols", 1),
        )
        legend.set_title(title=legend_title, prop={"size": legend_fontsize})
        if save_plot:
            self.export_plot(
                foldername=foldername, filename=filename, file_format=file_format
            )
        else:
            plt.show()

    def create_box_plot(
        self,
        metric,
        figsize=(10, 10),
        x_label="",
        y_label="Area Under the Step-to-goal Curve",
        colors=["red", "blue"],
        save_plot=False,
        foldername="plot/",
        filename="test",
        file_format="pdf",
    ):
        """
        Create a box plot for the given metric.

        Args:
            metric (str): Metric name.
            figsize (tuple, optional): Figure size. Defaults to (10, 10).
            x_label (str, optional): X-axis label. Defaults to "".
            y_label (str, optional): Y-axis label. Defaults to "Area Under the Step-to-goal Curve".
            colors (list[str], optional): List of colors for different cases. Defaults to ["red", "blue"].
            save_plot (bool, optional): Whether to save the plot. Defaults to False.
            foldername (str, optional): Name of the folder to save the plot. Defaults to "plot/".
            filename (str, optional): Name of the file. Defaults to "test".
            file_format (str, optional): File format for saving the plot. Defaults to "pdf".
        """
        plt.figure(figsize=figsize)
        dict_cases = {}
        for case_index, case in enumerate(self.cases):
            tmp = self.load_data(self.data[case_index], metric)
            auc_ls = []
            for row in tmp:
                auc_value = self.compute_auc(row)
                auc_ls.append(auc_value)
            dict_cases[case] = auc_ls
        df = pd.DataFrame(dict_cases)
        fig = sns.boxplot(data=df, palette=colors, boxprops=dict(alpha=0.8))
        fig.set(xlabel=x_label, ylabel=y_label)
        if save_plot:
            if foldername == None:
                folder = fm.create_folder(self.working_dir)
            folder = fm.create_folder(foldername)
            filename = fm.create_filename(
                folder, filename=filename, file_format=file_format, time_stampt=False
            )
            box_plot = fig.get_figure()
            box_plot.savefig(filename, bbox_inches="tight")
        else:
            plt.show()

    # Define a method to export the plot to a file
    def export_plot(self, foldername=None, filename="", file_format="pdf"):
        if foldername == None:
            folder = fm.create_folder(self.working_dir)
        folder = fm.create_folder(foldername)
        filename = fm.create_filename(
            folder, filename=filename, file_format=file_format, time_stampt=False
        )
        plt.savefig(filename, bbox_inches="tight")
