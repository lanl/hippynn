"""
Layers for simple operations
"""
import torch


class LambdaModule(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def extra_repr(self):
        return self.fn.__name__

    def forward(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


class ListMod(torch.nn.Module):
    def forward(self, *features):
        return [x.to(torch.get_default_dtype()) for x in features]


class _WeightedLoss(torch.nn.Module):
    loss_func = None

    def forward(self, pred, true, weights):
        unweighted_loss = self.loss_func(pred, true, reduction='none')
        normalized_weights = weights/torch.mean(weights)
        weighted_loss = (normalized_weights*unweighted_loss).mean()
        return weighted_loss


class WeightedMSELoss(_WeightedLoss):
    loss_func = staticmethod(torch.nn.functional.mse_loss)


class WeightedMAELoss(_WeightedLoss):
    loss_func = staticmethod(torch.nn.functional.l1_loss)


class AtLeast2D(torch.nn.Module):
    def forward(self, item):
        if item.ndimension() == 1:
            item = item.unsqueeze(1)
        return item


class ValueMod(torch.nn.Module):
    def __init__(self, value, convert=True):
        super().__init__()
        if not isinstance(value, torch.Tensor) and convert:
            value = torch.tensor(value, dtype=torch.get_default_dtype())

        if isinstance(value, torch.Tensor):
            self.register_buffer("value", value)
        else:
            self.value = value

    def extra_repr(self):
        return str(self.value)

    def forward(self):
        return self.value


class Idx(torch.nn.Module):
    def __init__(self, index, repr_info=None):
        super().__init__()
        self.index = index
        self.repr_info = repr_info

    def extra_repr(self):
        if self.repr_info is None:
            return ""
        return "{parent_name}.{index}".format(**self.repr_info)

    def forward(self, bundled_inputs):
        return bundled_inputs[self.index]

class EnsembleTarget(torch.nn.Module):
    def forward(self,*input_tensors):
        n_members = len(input_tensors)

        all = torch.stack(input_tensors, dim=1)
        mean = torch.mean(all, dim=1)
        std = torch.std(all, dim=1)
        return mean, std, all