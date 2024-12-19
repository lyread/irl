import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import savgol_filter
from sklearn.metrics import auc
import os

import algorithms.utils.file_manipulation as fm


METRICS = {
    "training_episodes": "Training Episodes",
    "training_steps": "Training Steps",
    "training_rewards": "Training Rewards",
    "training_successes": "Training Successes",
    "training_failure_rate": "Training Failure Rate",
    "training_undos": "Training Undos",
    "training_pos_rewards": "Positive Training Rewards",
    "training_neg_rewards": "Negative Training Rewards",
    "test_steps": "Test Steps",
    "test_rewards": "Test Rewards",
    "test_successes": "Test Successes",
    "test_failure_rate": "Test Failure Rate",
    "test_undos": "Test Undos",
    "training_cumulated_steps": "Cumulated Training Steps",
    "test_cumulated_steps": "Cumulated Test Steps",
    "test_pos_rewards": "Positive Test Rewards",
    "test_neg_rewards": "Negative Test Rewards",
}


class MultiCasesVisualizer:
    def __init__(
        self,
        case_dirs: list[str],
        cases: list[str],
        env: str,
        results_dir: str = "results",
        color_palette: str = "rocket",
        use_color_fade: bool = True,
        overwrite_colors: dict[int, str] | None = None,
        wd: str = "/tmp/",
        plots_dir: str = "plots/",
        figsize: tuple[int, int] = (16, 10),
        label_fontsize: int = 30,
        legend_fontsize: int = 30,
    ):
        """A visualizer for comparing multiple cases.

        Args:
            case_dirs (list[str]): Dicrectories of the cases.
            cases (list[str]): Names of the cases.
            env (str): Environment name.
            results_dir (str, optional): Results directory. Defaults to "results".
            color_palette (str, optional): Seaborn color palette. Defaults to "rocket".
            use_color_fade (bool, optional): Use evenly distributed colors from the palette. Defaults to True.
            overwrite_colors (dict[int, str] | None, optional): Dictionary containing a mapping from case indeces to colors that are overwritten. E.g. {0: "blue"} can be used to assign a custom color to the baseline. Defaults to None.
            wd (str, optional): Working directory. Defaults to "/tmp/".
            plots_dir (str, optional): Output directory for plots. Defaults to "plots/".
            figsize (tuple[int, int], optional): Matplotlib figure size. Defaults to (16, 10).
            label_fontsize (int, optional): Matplotlib label font size. Defaults to 30.
            legend_fontsize (int, optional): Matplotlib legend font size. Defaults to 30.
        """
        self.case_dirs = case_dirs
        self.cases = cases
        self.env = env
        self.results_dir = results_dir
        self.data = self.load_data()
        self.metrics = list(METRICS.keys())

        if use_color_fade:
            self.colors = self.get_hex_colors_from_colormap(color_palette, len(cases))
        else:
            self.colors = [
                str(sns.color_palette(color_palette)[i]) for i in range(len(cases))
            ]

        # overwrite certain colors in the palette:
        if overwrite_colors is not None:
            for index, color in overwrite_colors.items():
                self.colors[index] = color

        self.set_working_directory(wd=wd)

        self.plots_dir = plots_dir
        self.figsize = figsize
        self.label_fontsize = label_fontsize
        self.legend_fontsize = legend_fontsize

    def load_data(self):
        
        data_frames = []
        
        for case, case_dir in zip(self.cases, self.case_dirs):
            
            train_df = pd.read_csv(
                os.path.join(
                    self.results_dir,
                    case_dir,
                    "logs",
                    "training_logs",
                    "multiple_run",
                    f"{self.env}.csv",
                )
            )
            test_df = pd.read_csv(
                os.path.join(
                    self.results_dir,
                    case_dir,
                    "logs",
                    "test_logs",
                    "multiple_run",
                    f"{self.env}.csv",
                )
            )
            train_df = train_df.rename(columns={"Unnamed: 0": "episodes"})
            train_df.columns = [f"training_{col}" for col in train_df.columns]

            test_df = test_df.drop(columns=["Unnamed: 0"])
            test_df.columns = [f"test_{col}" for col in test_df.columns]

            overall_df = pd.concat([train_df, test_df], axis=1)
            overall_df = overall_df.rename(
                columns={
                    "test_cumulated_test_steps": "test_cumulated_steps",
                    "test_cumulated_training_steps": "training_cumulated_steps",
                }
            )
            overall_df = overall_df.rename(columns=METRICS)
            overall_df["Case"] = case
            data_frames.append(overall_df)

        return pd.concat(data_frames)

    @staticmethod
    def rgb_to_hex(rgb):
        return "#{:02x}{:02x}{:02x}".format(
            int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
        )

    @staticmethod
    def get_hex_colors_from_colormap(cmap_name, num_colors, margin=0.1):
        cmap = sns.color_palette(cmap_name, as_cmap=True)

        adjusted_range = [margin, 1 - margin]

        colors = [
            cmap(
                adjusted_range[0]
                + i * (adjusted_range[1] - adjusted_range[0]) / (num_colors - 1)
            )
            for i in range(num_colors)
        ]

        hex_colors = [MultiCasesVisualizer.rgb_to_hex(color) for color in colors]

        return hex_colors

    def set_working_directory(self, wd="/tmp/"):
        self.working_dir = fm.standardize_folder(wd)  # Trailing slash
        self.target_dir = fm.standardize_folder(self.working_dir + "plot_data")

    def compute_auc(self, data: np.ndarray) -> float:
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
        return self.data[self.data["Case"] == self.cases[case_index]][metric].to_numpy()

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
        tmp = self.data.copy()
        tmp = tmp[tmp["Case"] == self.cases[case_index]]
        tmp = tmp[metric].to_numpy()

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
        show_mean=False,
        show_median=False,
        show_max=False,
        show_min=False,
        max_line=None,
        min_line=None,
        show_percentage=False,
        fill_between=False,
        test_every=1,
        x_label=None,
        y_label=None,
        legend_title=None,
        line_width=None,
        smooth=False,
        title=None,
        window_length=11,
        poly_order=4,
        save_plot=False,
        filename="",
        file_format="pdf",
        **kwargs,
    ):
        """
        Create a single line plot for a specific metric.

        Args:
            metric_x (str): X-axis metric.
            metric_y (str): Y-axis metric.
            show_mean (bool, optional): Whether to show mean. Defaults to False.
            show_median (bool, optional): Whether to show median. Defaults to False.
            show_max (bool, optional): Whether to show maximum value. Defaults to False.
            show_min (bool, optional): Whether to show minimum value. Defaults to False.
            show_percentage (bool, optional): Whether to show values as percentage. Defaults to False.
            fill_between (bool, optional): Whether to fill between curves. Defaults to False.
            test_every (int, optional): Test frequency. If you want show training episode in x axis.
            x_label (str, optional): X-axis label. Defaults to None.
            y_label (str, optional): Y-axis label. Defaults to None.
            legend_title (str, optional): Legend title. Defaults to None.
            line_width (int, optional): Line width. Defaults to None.
            smooth (bool, optional): Whether to apply smoothing. Defaults to False.
            window_length (int, optional): Smoothing window length. Defaults to 11.
            poly_order (int, optional): Polynomial order for smoothing. Defaults to 4.
            save_plot (bool, optional): Whether to save the plot. Defaults to False.
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
        plt.figure(figsize=self.figsize)
        ax = plt.subplot()
        xlabel = x_label if x_label else metric_x
        ylabel = y_label if y_label else metric_y
        ax.set_xlabel(xlabel, fontsize=self.label_fontsize)
        ax.set_ylabel(ylabel, fontsize=self.label_fontsize)
        if title:
            ax.set_title(title, fontsize=30)
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
                max_x.append(x[np.argmax(y)])
                max_y.append(np.max(y))
            if show_min:
                min_x.append(x[np.argmin(y)])
                min_y.append(np.min(y))
            ax.plot(
                x,
                y,
                label=case,
                color=self.colors[case_index],
                linewidth=line_width,
                alpha=1,
                zorder=0,
            )
            if fill_between:
                ax.fill_between(
                    x,
                    shadow_up,
                    shadow_low,
                    color=self.colors[case_index],
                    alpha=0.1,
                    zorder=0,
                )
        if show_max:
            ax.scatter(
                max_x,
                max_y,
                marker="o",
                s=300,
                c=self.colors,
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
                c=self.colors,
                alpha=1,
                edgecolors="w",
                linewidths=2,
                zorder=1,
            )
        if min_line:
            ax.plot(
                x,
                np.array([min_line for _ in range(len(x))]),
                label="Minimum",
                color="grey",
                linewidth=line_width,
                alpha=1,
                zorder=0,
                linestyle="--",
            )
        if max_line:
            ax.plot(
                x,
                np.array([max_line for _ in range(len(x))]),
                label="Maximum",
                color="grey",
                linewidth=line_width,
                alpha=1,
                zorder=0,
                linestyle="--",
            )
        legend_loc = kwargs.get("legend_loc", "best")
        legend = plt.legend(
            fontsize=self.legend_fontsize,
            loc=legend_loc,
            ncols=kwargs.get("legend_ncols", 1),
        )
        legend.set_title(title=legend_title, prop={"size": self.legend_fontsize})
        if save_plot:
            self.export_plot(
                foldername=self.plots_dir, filename=filename, file_format=file_format
            )
        else:
            plt.show()

    def create_box_plot(
        self,
        metric,
        x_label="",
        y_label="Area Under the Step-to-goal Curve",
        save_plot=False,
        filename="test",
        file_format="pdf",
    ):
        """
        Create a box plot for the given metric.

        Args:
            metric (str): Metric name.
            x_label (str, optional): X-axis label. Defaults to "".
            y_label (str, optional): Y-axis label. Defaults to "Area Under the Step-to-goal Curve".
            save_plot (bool, optional): Whether to save the plot. Defaults to False.
            filename (str, optional): Name of the file. Defaults to "test".
            file_format (str, optional): File format for saving the plot. Defaults to "pdf".
        """
        plt.figure(figsize=self.figsize)
        dict_cases = {}
        for case_index, case in enumerate(self.cases):
            tmp = self.load_data(self.data[case_index], metric)
            auc_ls = []
            for row in tmp:
                auc_value = self.compute_auc(row)
                auc_ls.append(auc_value)
            dict_cases[case] = auc_ls
        df = pd.DataFrame(dict_cases)
        fig = sns.boxplot(data=df, palette=self.colors, boxprops=dict(alpha=0.8))
        fig.set(xlabel=x_label, ylabel=y_label)
        if save_plot:
            if self.plots_dir == None:
                folder = fm.create_folder(self.working_dir)
            folder = fm.create_folder(self.plots_dir)
            filename = fm.create_filename(
                folder, filename=filename, file_format=file_format, time_stampt=False
            )
            box_plot = fig.get_figure()
            box_plot.savefig(filename, bbox_inches="tight")
        else:
            plt.show()

    def create_scatter_plot(
        self,
        x_metric,
        y_metric,
        x_label="",
        y_label="",
        show_percentage=False,
        legend_title="",
        title=None,
        save_plot=False,
        filename="test",
        file_format="pdf",
        **kwargs,
    ):
        """
        Create a scatter plot for the given x and y metrics.

        Args:
            x_metric (str): Metric name for y.
            y_metric (str): Metric name.
            x_label (str, optional): X-axis label. Defaults to "".
            y_label (str, optional): Y-axis label. Defaults to "".
            show_mean (bool, optional): Whether to show mean. Defaults to False.
            show_median (bool, optional): Whether to show median. Defaults to False.
            show_percentage (bool, optional): Whether to show values as percentage. Defaults to False.
            legend_title (str, optional): Legend title. Defaults to "".
            title (str, optional): Title of the plot. Defaults to None.
            save_plot (bool, optional): Whether to save the plot. Defaults to False.
            filename (str, optional): Name of the file. Defaults to "test".
            file_format (str, optional): File format for saving the plot. Defaults to "pdf".
        """
        plt.figure(figsize=self.figsize)
        ax = plt.subplot()
        ax.set_xlabel(x_label, fontsize=self.label_fontsize)
        ax.set_ylabel(y_label, fontsize=self.label_fontsize)
        ax.tick_params(which="both", top=True, right=True, direction="in", labelsize=25)

        data = self.data.copy()

        if show_percentage:
            data[y_metric] *= 100

        sns.scatterplot(
            data=data, x=x_metric, y=y_metric, hue="Case", palette=self.colors
        )

        if title:
            ax.set_title(title, fontsize=30)

        legend_loc = kwargs.get("legend_loc", "best")
        legend = plt.legend(
            fontsize=self.legend_fontsize,
            loc=legend_loc,
            ncols=kwargs.get("legend_ncols", 1),
        )
        legend.set_title(title=legend_title, prop={"size": self.legend_fontsize})

        if save_plot:
            if self.plots_dir == None:
                folder = fm.create_folder(self.working_dir)
            folder = fm.create_folder(self.plots_dir)
            filename = fm.create_filename(
                folder, filename=filename, file_format=file_format, time_stampt=False
            )
            plt.savefig(filename, bbox_inches="tight")
        else:
            plt.show()

    # Define a method to export the plot to a file
    def export_plot(self, foldername=None, filename="", file_format="pdf"):
        if foldername == None:
            folder = fm.create_folder(self.working_dir)
        folder = fm.create_folder(self.plots_dir)
        filename = fm.create_filename(
            folder, filename=filename, file_format=file_format, time_stampt=False
        )
        plt.savefig(filename, bbox_inches="tight")
