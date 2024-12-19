import datetime
import os

import pandas as pd


def standardize_folder(folder):
    if folder != "":
        if folder[-1] != "/":
            folder += "/"
    return folder


def create_folder(folder="tmp/logs/"):
    standardize_folder(folder)
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder


def create_filename(
    folder="tmp/logs/",
    filename="logfile",
    prefix="",
    suffix="",
    file_format="",
    time_stampt=True,
):
    folder = standardize_folder(folder)
    if prefix != "":
        prefix = prefix + "-"

    if suffix != "":
        suffix = "-" + suffix

    if file_format != "":
        file_format = "." + file_format

    timeStamptText = ""
    if time_stampt:
        text = (datetime.datetime.now()).strftime("%m-%d_%H-%M")
        timeStamptText = text
        if (prefix != "") or (filename != "") or (suffix != ""):
            timeStamptText += "-"
    return folder + timeStamptText + prefix + filename + suffix + file_format


def export_data(
    data: dict[str, list] | list,
    columns=None,
    folder_name="tmp/log/",
    suffix="",
    file_name="",
    file_format="csv",
    time_stampt=False,
):
    # can do in the experiment script
    standardize_folder(folder_name)
    folder = create_folder(folder_name)
    filename = create_filename(
        folder,
        filename=file_name,
        suffix=suffix,
        file_format=file_format,
        time_stampt=time_stampt,
    )
    df = pd.DataFrame(data)
    if columns is not None:
        df.columns = columns
    if file_format == "pkl":
        df.to_pickle(filename)
    else:
        df.to_csv(filename)
