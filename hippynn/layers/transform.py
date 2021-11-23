"""
Layers that wrap other layers.
"""

import torch


# RESNET WRAPPER for the layers.
class ResNetWrapper(torch.nn.Module):
    """
    Resnet Wrapper Class
    """

    def __init__(self, base_layer, nf_in, nf_middle, nf_out, activation=torch.nn.functional.softplus):
        """
        Constructor

        :param base_layer: The ResLayer pytorch module
        :param nf_in: Input dimensions
        :param nf_out: Output dimensions
        :param activation: a nonlinearity function
        """
        super().__init__()

        self.activation = activation
        self.base_layer = base_layer
        self.res_layer = torch.nn.Linear(nf_middle, nf_out)

        if nf_in != nf_out:
            self.adjust_layer = torch.nn.Linear(nf_in, nf_out, False)
            self.needs_size_adjust = True
        else:
            self.needs_size_adjust = False

    def regularization_params(self):
        params = [self.res_layer.weight]
        if hasattr(self.base_layer, "regularization_params"):
            params.extend(self.base_layer.regularization_params())
        else:
            params.append(self.base_layer.weight)
        if self.needs_size_adjust:
            params.append(self.adjust_layer.weight)
        return params

    def forward(self, *input):
        """

        :param input: list of inputs for the layer;
            the first one is taken to be the features which will be computed as a residual
        :return: Pytorch module output
        """
        middle_activation = self.activation(self.base_layer(*input))
        difference_activation = self.res_layer(middle_activation)

        input_activation = input[0]
        if self.needs_size_adjust:
            input_activation = self.adjust_layer(input_activation)

        return difference_activation + input_activation
