import os
import warnings
import copy

from matplotlib import pyplot as plt

from ..graphs import GraphModule
from .. import settings


class PlotMaker:
    """
    The plot maker is responsible for collecting the data for the plotters and executing the plotters.
    """

    def __init__(self, *plotters, plot_every, save_dir="plots/"):
        """

        :param plotters: Individual plotters to use.
        :param plot_every: How often to make plots during training (in epochs)
        :param save_dir: What directory to store the plots in, relative to the
         training experiment path.
        """
        self.plotters = plotters
        self.save_dir = save_dir
        self.plot_every = plot_every
        self.torch_module = None

    def assemble_module(self, model_outputs, targets):
        model_outputs = [x.pred for x in model_outputs]
        targets = [x.true for x in targets]
        all_inputs = (*model_outputs, *targets)
        graph = GraphModule(all_inputs, self.required_nodes)
        # Plot maker runs on CPU
        self.torch_module = copy.deepcopy(graph)
        self.torch_module.cpu()

    @property
    def required_nodes(self):
        return [p for plotter in self.plotters for p in plotter.parents]

    def make_plots(self, outputs, targets, sub_location=None):
        computed_data = self.torch_module(*outputs, *targets)
        remaining_data = computed_data

        location = self.make_full_location(sub_location)
        print("Making plots. Saved location: {}".format(location))

        for plotter in self.plotters:
            n_args = len(plotter.parents)
            this_data = remaining_data[:n_args]
            remaining_data = remaining_data[n_args:]
            if plotter.shown or plotter.saved:
                fig = plotter.make_plot(this_data)
                if plotter.shown:
                    plt.show()
                if plotter.saved:
                    file = os.path.join(location, plotter.saved)
                    print("Saving plot at {}".format(file))
                    fig.savefig(file, transparent=settings.TRANSPARENT_PLOT)
                plt.close(fig)

    def make_full_location(self, sub_location):
        sub_location = sub_location if sub_location else "Unspecified"
        location = os.path.join(self.save_dir, sub_location)
        if os.path.exists(location):
            warnings.warn("May override plots in {} !".format(location))
        else:
            os.makedirs(location)
        return location

    def plot_phase(self, prediction_all_vals, target_all_vals, when, eval_type):
        if when is not None:
            # If integer, convert to epoch number string IF it is time to plot.
            if isinstance(when, int):
                if (when % self.plot_every) == 0:
                    when = "epochs/epoch{}".format(when)
                else:
                    return
            # If string, make the plots. Note that this CANNOT become an ELIF, the above will convert when to a str.
            if isinstance(when, str):
                location = os.path.join(when, eval_type)
                self.make_plots(prediction_all_vals, target_all_vals, location)
            else:
                warnings.warn(
                    "Evaluation description was not a string "
                    + "or integer (specifying epoch number): {}\n".format(when)
                    + "Not generating plots."
                )
