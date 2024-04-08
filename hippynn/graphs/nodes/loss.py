"""
Nodes for constructing loss functions.
"""
import torch
import torch.nn.functional

from ... import settings
from ..indextypes import IdxType, elementwise_compare_reduce
from .base import SingleNode
from ...layers import algebra as algebra_modules
from ...layers import regularization as reg_modules


class _DebugBroadCast(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def extra_repr(self):
        return repr(self.inner)

    def forward(self, *inputs):
        shapes = [x.shape for x in inputs]
        print("DEBUG BROADCASTING shapes:", shapes)
        for s in shapes:
            if not (s == shapes[0]):
                raise ValueError("Broadcasting in loss shapes:", shapes)
        return self.inner(*inputs)


class ReduceSingleNode(SingleNode):
    _index_state = IdxType.Scalar

    def __init__(self, parent):
        name = self._classname + "({})".format(parent.name)
        parent = elementwise_compare_reduce(parent.main_output)
        super().__init__(name, (parent,))

    @classmethod
    def of_node(cls, node):
        return cls(node.pred)

    def __init_subclass__(cls, op=None, **kwargs):
        if op is not None:
            cls._classname = op.__name__
            cls.torch_module = algebra_modules.LambdaModule(op)


class Mean(ReduceSingleNode, op=torch.mean):
    pass


def mean_sq(input):
    return torch.pow(input, 2).mean()


class MeanSq(ReduceSingleNode, op=mean_sq):
    pass


class Std(ReduceSingleNode, op=torch.std):
    pass


class Var(ReduceSingleNode, op=torch.var):
    pass


class _BaseCompareLoss(SingleNode):
    _index_state = IdxType.Scalar

    def __init__(self, predicted, true):
        name = "{}({},{})".format(self._classname, predicted.name, true.name)
        predicted, true = elementwise_compare_reduce(predicted, true)
        super().__init__(name, (predicted, true), module=None)

    @classmethod
    def of_node(cls, node):
        node = node.main_output
        return cls(node.pred, node.true)

    def __init_subclass__(cls, op=None, **kwargs):
        if op is not None:
            cls._classname = op.__name__
            if settings.DEBUG_LOSS_BROADCAST:
                cls.torch_module = _DebugBroadCast(op)
            else:
                # Note: as of now, we need to wrap raw operations as a loss module because
                # the graph module's ModuleList can't take non-torch operations.
                # Add the lambda to the graph module instead to put the problem nearer to the solution?
                cls.torch_module = algebra_modules.LambdaModule(op)

class _WeightedCompareLoss(SingleNode):
    _index_state = IdxType.Scalar
    def __init__(self, predicted,true, weight):
        name = "{}({},{},{})".format(self._classname, predicted.name, true.name, weight.name)
        predicted, true, weight = elementwise_compare_reduce(predicted, true, weight)
        super().__init__(name, (predicted, true, weight), module=None)
    @classmethod
    def of_node(cls, node, weight):
        node = node.main_output
        true = node.true
        predicted = node.pred
        weight = weight.true
        return cls(predicted, true, weight)

class WeightedMSELoss(_WeightedCompareLoss):
    _classname = "WeightedMSE"
    torch_module = algebra_modules.WeightedMSELoss()

class WeightedMAELoss(_WeightedCompareLoss):
    _classname = "WeightedMAE"
    torch_module = algebra_modules.WeightedMAELoss()

class RsqMod(torch.nn.Module):
    def forward(self, predicted, true):
        return 1 - (torch.mean(torch.pow(predicted - true, 2)) / true.var())


class Rsq(_BaseCompareLoss):
    torch_module = RsqMod()
    _classname = "Rsq"


class MSELoss(_BaseCompareLoss, op=torch.nn.functional.mse_loss):
    pass


class MAELoss(_BaseCompareLoss, op=torch.nn.functional.l1_loss):
    pass


class _LPReg(SingleNode):
    _index_state = IdxType.Scalar
    _auto_module_class = reg_modules.LPReg

    def __init__(self, network, p=2, module="auto"):
        name = "L^P_Reg({},p={})".format(network.name, p)
        parents = (network,)
        self.p = p
        super().__init__(name, parents, module=module)

    def auto_module(self):
        return self._auto_module_class(self.parents[0].torch_module, p=self.p)


def lpreg(network, p):
    return _LPReg(network, p=p).pred


def l2reg(network):
    return lpreg(network, p=2)


def l1reg(network):
    return lpreg(network, p=1)

# For loss functions with phases
def absolute_errors(predict: torch.Tensor, true: torch.Tensor):
    """Compute the absolute errors with phases between predicted and true values. In
    other words, prediction should be close to the absolute value of true, and the sign
    does not matter.

    :param predict: predicted values
    :type predict: torch.Tensor
    :param true: true values
    :type true: torch.Tensor
    :return: errors
    :rtype: torch.Tensor
    """

    return torch.minimum(torch.abs(true - predict), torch.abs(true + predict))


