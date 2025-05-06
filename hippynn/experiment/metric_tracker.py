"""
Keep track of training metrics over the experiment
"""
import copy
import warnings
import torch
from hippynn.tools import add_safe_globals

try:
    from hippynn.plotting import plot_all_over_time
except ImportError:
    plot_all_over_time = lambda *args: warnings.warn(
        "Could not import plotting module. Make sure matplotlib is installed to enable plotting"
    )


class MetricTracker:
    """
    MetricTracker instances keep track of metrics and models for an experiment

    :ivar metric_names: the tracked metrics
    :ivar stopping_key: the metric used to determine what model is best
    :ivar best_metric_values: dictionary of metric keys to the best metric values observed so far.
          lower is assumed to be better.
    :ivar best_model: state dict for the best model so far.
    :ivar epoch_metric_values: list (index epoch) of
                               dictionary (key split, value of dictionary (key metric, value metric value)
    :ivar other_metric_values: dictionary of metric values at other times than after an epoch, for example
        the final test values.
    :ivar epoch_times: timing info for each epoch
    :ivar quiet: whether to print the values registered.

    """

    def __init__(self, metric_names, stopping_key, quiet=False):
        """

        :param metric_names:
        :param stopping_key:
        :param quiet:
        :param split_names: splits to track.
        """
        if stopping_key not in metric_names and stopping_key is not None:
            raise ValueError("Stopping key {} is not in metric names {}".format(stopping_key, metric_names))
        self.metric_names = metric_names
        self.stopping_key = stopping_key

        # metadata
        self.name_column_width = max(len(i) for i in self.metric_names)  # name column size
        self.n_metrics = len(metric_names)

        # State variables
        self.best_metric_values = {}
        self.other_metric_values = {}
        self.best_model = None
        self.epoch_times = []

        self.epoch_metric_values = []
        self.epoch_best_metric_values = []
        self.quiet = quiet

    @classmethod
    def from_evaluator(cls, evaluator):
        return cls(evaluator.metric_names, evaluator.stopping_key)

    @property
    def current_epoch(self):
        return len(self.epoch_best_metric_values)

    def register_metrics(self, metric_info, when):
        """
        :param metric_info: dictionary of metric names: metric values
        :param when: string or integer specifying epoch number.
        :return:
        """
        """
        Updates the metrics for this epoch and computers which ones are better.
        
        """
        better_metrics = {k: {} for k in self.best_metric_values}
        for split_type, typevals in metric_info.items():
            for mname, mval in typevals.items():
                try:
                    old_best = self.best_metric_values[split_type][mname]
                    better = old_best > mval
                    del old_best  # marking not needed.
                except KeyError:
                    if split_type not in self.best_metric_values:
                        # Haven't seen this split before!
                        self.best_metric_values[split_type] = {}
                        better_metrics[split_type] = {}
                    better = True  # old best was not found!

                if better:
                    self.best_metric_values[split_type][mname] = mval
                better_metrics[split_type][mname] = better
        if isinstance(when, int):
            self.epoch_metric_values.append(metric_info)
            self.epoch_best_metric_values.append(copy.deepcopy(self.best_metric_values))
        else:
            self.other_metric_values[when] = metric_info

        if self.stopping_key and "valid" in metric_info:
            better_model = better_metrics.get("valid", {}).get(self.stopping_key, False)
            stopping_key_metric = metric_info["valid"][self.stopping_key]
        else:
            better_model = None
            stopping_key_metric = None

        return better_metrics, better_model, stopping_key_metric

    def evaluation_print(self, evaluation_dict, quiet=None, _print=print):
        if quiet is None:
            quiet = self.quiet
        if quiet:
            return
        table_evaluation_print(evaluation_dict, self.metric_names, self.name_column_width, _print=_print)

    def evaluation_print_better(self, evaluation_dict, better_dict, quiet=None, _print=print):
        if quiet is None:
            quiet = self.quiet
        if quiet:
            return
        table_evaluation_print_better(evaluation_dict, better_dict, self.metric_names, self.name_column_width, _print=print)
        if self.stopping_key:
            _print(
                "Best {} so far: {:>8.5g}".format(
                    self.stopping_key, self.best_metric_values["valid"][self.stopping_key]
                )
            )

    def plot_over_time(self):
        plot_all_over_time(self.epoch_metric_values, self.epoch_best_metric_values)

add_safe_globals([MetricTracker])


# Driver for printing evaluation table results, with * for better entries.
# Decoupled from the estate in case we want to more easily change print formatting.
def table_evaluation_print_better(evaluation_dict, better_dict, metric_names, n_columns, _print=print):
    """
    Print metric results as a table, add a '*' character for metrics in better_dict.

    :param evaluation_dict: dict[eval type]->dict[metric]->value
    :param better_dict: dict[eval type]->dict[metric]->bool
    :param metric_names: Names
    :param n_columns: Number of columns for name fields.
    :return: None
    """
    type_names = evaluation_dict.keys()
    better_labels = {True: "*", False: " "}

    transposed_values_better = [
        [(better_labels[better_dict[tname][mname]], evaluation_dict[tname][mname]) for tname in type_names]
        for mname in metric_names
    ]

    n_types = len(type_names)

    header = " " * (n_columns + 2) + "".join("{:>14}".format(tn) for tn in type_names)
    rowstring = "{:<" + str(n_columns) + "}: " + "   {}{:>10.5g}" * n_types

    _print(header)
    _print("-" * len(header))
    for n, valsbet in zip(metric_names, transposed_values_better):
        rowoutput = [k for bv in valsbet for k in bv]
        _print(rowstring.format(n, *rowoutput))


# Driver for printing evaluation table results.
# Decoupled from the estate in case we want to more easily change print formatting.
def table_evaluation_print(evaluation_dict, metric_names, n_columns, _print=print):
    """
    Print metric results as a table.

    :param evaluation_dict: dict[eval type]->dict[metric]->value
    :param metric_names: Names
    :param n_columns: Number of columns for name fields.
    :return: None
    """

    type_names, type_values = zip(*evaluation_dict.items())
    transposed_values = [[t[m] for t in type_values] for m in metric_names]

    n_types = len(type_names)

    header = " " * (n_columns + 2) + "".join("{:>14}".format(tn) for tn in type_names)
    rowstring = "{:<" + str(n_columns) + "}: " + "    {:>10.5g}" * n_types

    _print(header)
    _print("-" * len(header))
    for n, vals in zip(metric_names, transposed_values):
        _print(rowstring.format(n, *vals))
    _print("-" * len(header))
