import os
import warnings
from collections import defaultdict, OrderedDict
import matplotlib.pyplot as plt

from .. import settings

DEFAULT_TIMEPLOT_KWARGS = {
    "train": {
        "c": "r",
        "ms": 3,
        "lw": 1,
    },
    "valid": {
        "c": "b",
        "lw": 1,
        "ms": 3,
    },
}
DEFAULT_TIMEPLOT_KWARGS = defaultdict(dict, DEFAULT_TIMEPLOT_KWARGS)


def plot_all_over_time(metric_list, best_metric_list, save_dir="plots/over_time"):
    transposed_metrics = defaultdict(lambda: defaultdict(list))
    transposed_best = defaultdict(lambda: defaultdict(list))
    for emet, ebest in zip(metric_list, best_metric_list):
        for eval_type, this_dict in emet.items():
            for metric_name, mval in this_dict.items():
                bmval = ebest[eval_type][metric_name]
                transposed_metrics[metric_name][eval_type].append(mval)
                transposed_best[metric_name][eval_type].append(bmval)

    if os.path.exists(save_dir):
        warnings.warn("May override plots in {} !".format(save_dir))
    else:
        os.makedirs(save_dir)

    for metric_name in transposed_metrics:
        this_met_data = OrderedDict()
        pltkw_info = dict()
        for type_name in transposed_metrics[metric_name]:
            this_met_data[type_name] = transposed_metrics[metric_name][type_name]
            pltkw_info[type_name] = {"ls": "-", "marker": "o"}
            pltkw_info[type_name].update(DEFAULT_TIMEPLOT_KWARGS[type_name])

            best_name = type_name + "-best"
            this_met_data[best_name] = transposed_best[metric_name][type_name]
            pltkw_info[best_name] = {"ls": ":"}
            pltkw_info[best_name].update(DEFAULT_TIMEPLOT_KWARGS[type_name])

        plot_over_time(metric_name, this_met_data, pltkw_info, save_dir=save_dir)


def plot_over_time(metric_name, metric_data, pltkwd_info, save_dir):
    fig, ax = plt.subplots()

    plt.title(metric_name)
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    for key, data in metric_data.items():
        plt.plot(data, **pltkwd_info[key], label=key)
    plt.legend()
    fname = os.path.join(save_dir, metric_name + settings.DEFAULT_PLOT_FILETYPE)
    # TODO: Check fname against characters meaningful to the filesystem, e.g. `/`.
    plt.savefig(fname)

    datamin = min(min(arr) for arr in metric_data.values())
    datamax = max(max(arr) for arr in metric_data.values())
    # if settings.TIMEPLOT_AUTOSCALING is set to True, only produce log-scale plots under certain conditions
    if (not settings.TIMEPLOT_AUTOSCALING) or (datamin > 0 and datamax / datamin > 2):
        plt.yscale("log")
        fname = os.path.join(save_dir, metric_name + "_logplot" + settings.DEFAULT_PLOT_FILETYPE)
        plt.savefig(fname)

        if (not settings.TIMEPLOT_AUTOSCALING) or (max(len(arr) for arr in metric_data.values()) > 10):
            plt.xscale("log")
            fname = os.path.join(save_dir, metric_name + "_loglogplot" + settings.DEFAULT_PLOT_FILETYPE)
            plt.savefig(fname)

    plt.close(fig)
