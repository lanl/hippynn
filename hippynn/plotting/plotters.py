import itertools

import matplotlib
from matplotlib import pyplot as plt

from ..graphs.indextypes.reduce_funcs import elementwise_compare_reduce
from .. import settings

import torch


class Plotter:
    """
    Base class for plotters.
    Can be used with external callbacks by specifying plt_fn.
    """

    def __init__(self, parents, plt_fn=None, saved=False, shown=False):
        """
        Base plotter arguments inherited by all plotters.

        :param parents: nodes reflecting the data required to make the plotter
        :param plt_fn: a function to use to make the plot
        :param saved: whether to save the plot to a file
        :param shown: whether to show the plot using ``plt.show``
        """
        self.parents = parents
        self.shown = shown
        self.saved = saved
        if callable(plt_fn):
            self.plt_fn = plt_fn

    def make_plot(self, data_args):
        fig, ax = plt.subplots()
        self.plt_fn(*data_args)
        return fig

    def plt_fn(self):
        return NotImplemented


def as_numpy(torch_tensor):
    return torch_tensor.data.cpu().numpy()


class ComposedPlotter(Plotter):
    def __init__(self, subplotters, **kwargs):
        self.subplotters = subplotters
        parents = list(itertools.chain.from_iterable((p.parents for p in subplotters)))
        super().__init__(parents, **kwargs)

    def plt_fn(self, *data_args):
        remaining_args = data_args
        for plotter in self.subplotters:
            n_args = len(plotter.parents)
            this_args = remaining_args[:n_args]
            remaining_args = remaining_args[n_args:]
            fig = plotter.plt_fn(*this_args)
        return fig


class _CompareableTruePred:
    @classmethod
    def compare(cls, node, **kwargs):
        node = node.main_output
        xlabel = "true " + node.db_name
        ylabel = "predicted " + node.db_name
        if kwargs.get("saved", False) is True:
            kwargs["saved"] = cls.__name__ + "-" + node.db_name + settings.DEFAULT_PLOT_FILETYPE
        reduced_true, reduced_pred = elementwise_compare_reduce(node.true, node.pred)
        return cls(reduced_true, reduced_pred, xlabel=xlabel, ylabel=ylabel, **kwargs)


class Hist2D(Plotter, _CompareableTruePred):
    def __init__(self, x_var, y_var, xlabel, ylabel, bins=200, norm=None, add_identity_line=True, **kwargs):
        super().__init__((x_var, y_var), **kwargs)
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.bins = bins
        self._norm = norm
        self.add_identity_line = add_identity_line

    def plt_fn(self, x_val, y_val):
        x_val = as_numpy(x_val).flatten()
        y_val = as_numpy(y_val).flatten()
        plt.hist2d(x_val, y_val, bins=self.bins, norm=self.norm)
        if self.add_identity_line:
            min_val = min(x_val.min(), y_val.min())
            max_val = max(x_val.max(), y_val.max())
            plt.plot((min_val, max_val), (min_val, max_val), c="r", lw=0.5)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.colorbar()

    @property
    def norm(self):
        return self._norm or matplotlib.colors.LogNorm()


class HierarchicalityPlot(Hist2D):
    def __init__(self, x_var, y_var, **kwargs):
        super().__init__(x_var, y_var, xlabel="log_10 R", ylabel="|ΔE|", add_identity_line=False, **kwargs)

    def plt_fn(self, x_val, y_val):
        with torch.autograd.no_grad():
            x_val = torch.log10(x_val + 1e-15)
            y_val = torch.abs(y_val)
        return super().plt_fn(x_val, y_val)


class Hist1DComp(Plotter, _CompareableTruePred):
    def __init__(self, x_var, y_var, xlabel, ylabel, bins=200, norm=None, **kwargs):
        super().__init__((x_var, y_var), **kwargs)
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.bins = bins
        self._norm = norm

    def plt_fn(self, x_val, y_val):
        x_val = as_numpy(x_val).flatten()
        y_val = as_numpy(y_val).flatten()
        plt.hist(x_val, bins=self.bins, label=self.xlabel, color=[1.0, 0, 0, 0.5])
        plt.hist(y_val, bins=self.bins, label=self.ylabel, color=[0, 0, 1.0, 0.5])
        plt.xlabel(self.xlabel + " and " + self.ylabel)
        plt.ylabel("Count")
        plt.legend()


class Hist1D(Plotter):
    def __init__(self, x_var, xlabel, bins=200, **kwargs):
        super().__init__((x_var,), **kwargs)
        self.xlabel = xlabel
        self.bins = bins

    def plt_fn(self, x_val):
        plt.hist(x_val, bins=self.bins)
        plt.xlabel(self.xlabel)
        plt.ylabel("Count")


class SensitivityPlot(Plotter):
    def __init__(self, sensitivity_module, r_min=0.1, r_max=None, n_r=500, **kwargs):
        self.sensitivity = sensitivity_module
        if r_max is None:
            r_max = self.sensitivity.hard_max_dist + 1
        self.r_params = r_min, r_max, n_r
        super().__init__((), **kwargs)

    def plt_fn(self):
        with torch.autograd.no_grad():
            mu = list(self.sensitivity.parameters())[0]
            r_range = torch.linspace(*self.r_params, dtype=mu.dtype, device=mu.device)
            # allow_warning=False to disable the false low distance warning
            sense_out = self.sensitivity(r_range, warn_low_distances=False).cpu().data.numpy()
            r_range = r_range.cpu().data.numpy()
        for sense_func in sense_out.transpose():
            plt.plot(r_range, sense_func, c="r")
        plt.plot(r_range, sense_out.sum(axis=1), c="b")
        plt.xlabel("Distance")
        plt.ylabel("Sensitivity")


class InteractionPlot(Plotter):
    def __init__(self, int_layer, r_min=0.1, r_max=None, n_r=500, **kwargs):
        raise NotImplementedError()  #  TODO Finish this.

        self.int_layer = int_layer
        if r_max is None:
            r_max = self.int_layer.sensitivity.hard_max_dist + 0.5
        self.r_params = r_min, r_max, n_r
        super().__init__((), **kwargs)

    def plt_fn(self):
        dtype, device = self.int_layer.int_weights.dtype, self.int_layer.int_weights.device
