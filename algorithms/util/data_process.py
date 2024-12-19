import pandas as pd
import numpy as np


def merge_single_run_data(data: list[pd.DataFrame]) -> pd.DataFrame:
    assert isinstance(data, list), "Data must be a list of dataframes"
    merged = {}
    for col in data[0].columns:
        merged[col] = [df[col].to_list() for df in data]
    result = pd.DataFrame(merged)
    return result


def merge_multiple_run_data(data: list[pd.DataFrame]) -> pd.DataFrame:
    assert isinstance(data, list), "Data must be a list of dataframes"
    result = pd.concat(data, axis=0, ignore_index=True)
    return result


def merge_single_and_multiple_run_data(
    single_run_data: pd.DataFrame | list[pd.DataFrame],
    multiple_run_data: pd.DataFrame | list[pd.DataFrame],
) -> pd.DataFrame:
    if isinstance(single_run_data, list) and isinstance(multiple_run_data, list):
        single_runs = merge_single_run_data(single_run_data)
        multiple_run_data.append(single_runs)
        return merge_multiple_run_data(multiple_run_data)
    if isinstance(single_run_data, pd.DataFrame) and isinstance(
        multiple_run_data, list
    ):
        multiple_runs = merge_multiple_run_data(multiple_run_data)
        merged = {}
        for key in multiple_runs.columns:
            column = multiple_runs[key].to_list()
            column.append(single_run_data[key].to_list())
            merged[key] = column
        return pd.DataFrame(merged)
    if isinstance(single_run_data, list) and isinstance(
        multiple_run_data, pd.DataFrame
    ):
        single_runs = merge_single_run_data(single_run_data)
        multiple_runs = [single_runs, multiple_run_data]
        return merge_multiple_run_data(multiple_runs)
    if isinstance(single_run_data, pd.DataFrame) and isinstance(
        multiple_run_data, pd.DataFrame
    ):
        merged = {}
        for key in multiple_run_data.columns:
            column = multiple_run_data[key].to_list()
            column.append(single_run_data[key].to_list())
            merged[key] = column
        return pd.DataFrame(merged)


def threshold_points(data, thresholds, add_best=False):
    return_val = []
    for i, threshold in enumerate(thresholds):
        for value in data:
            if value <= threshold:
                return_val.append(value)
                break
        if len(return_val) < i + 1:
            raise ValueError(f"No value below {threshold}% found in data.")
    if add_best:
        best_val = np.min(data)
        return_val.append(best_val)
    return return_val


def get_threshold_plot_data_frames(thresholds, data, cases, add_best=False):
    """
    Data has to be multiple run data.
    DataFrame should be looking like this:


    |                        Failure Rate                          |             Ask Likelihood       |
    |--------------------------------------------------------------|----------------------------------|
    |        [...] shape: (num_likelihood, ) => (11, ) here        |      [0.0, 0.1, ... 0.99]        |
    |        [...] shape: (num_likelihood, ) => (11, ) here        |      [0.0, 0.1, ... 0.99]        |
    |        [...] shape: (num_likelihood, ) => (11, ) here        |      [0.0, 0.1, ... 0.99]        |
    |        [...] shape: (num_likelihood, ) => (11, ) here        |      [0.0, 0.1, ... 0.99]        |
                                                ...
    row number: number of runs
    Ask Likelihood: use for x-axis

    """
    data_fr = [np.array(df["Failure Rate"].to_list()) * 100 for df in data]
    results = []
    final_results = []
    for d in data_fr:
        th_data = []
        for _data in d:
            th_vals = threshold_points(_data, thresholds, add_best=add_best)
            th_data.append(th_vals)
        results.append(th_data)
    num_df = len(thresholds) + 1 if add_best else len(thresholds)
    for i in range(num_df):
        new_results = []
        for res in results:
            tmp = []
            for res_ in res:
                tmp.append(res_[i])
            new_results.append(tmp)
        final_results.append(np.array(new_results).T)
    x_axis = [[float(ask) for ask in cases] for _ in range(len(final_results[0]))]
    th_plot_data = [
        pd.DataFrame(data=zip(ls, x_axis), columns=["Failure Rate", "Ask Likelihood"])
        for ls in final_results
    ]
    th_cases = [str(th) + "%" for th in thresholds]
    if add_best:
        th_cases.append("Best")
    return th_plot_data, th_cases
