Performance Tips
=================

Devices
-------

Using GPU resources will often speed up your training by more than a factor of ten.
In general hippynn will aim to use the "cuda" device if available, and then fall back on the default pytorch device.
To specify the device just pass a ``torch.device`` or compatible string to the :obj:`~hippynn.SetupParams` for a training run.
This will take care of devices for training.
At application time, if you're using a :func:`~hippynn.GraphModule`, :func:`~hippynn.Predictor`, :obj:`~hippynn.interfaces.ase_interface.HippynnCalculator`
called ``obj``, then you can use ``obj.to(torch.device('cuda'))`` to move these objects to the GPU.

Custom Kernels
--------------

hippynn has low-level implementations of the message passing functions for hip-nn.
These are almost always helpful, but the exact degree depends, mostly on the size of the batches and the size of the network.
In some cases with large models, there are multiple factors of improvement!
These improve both the memory usage and speed for evaluating the network.
You shouldn't need anything special for these to work on the GPU using triton, which is included with pytorch.
On CPU they require installing numba.
Custom kernels are described in more detail :doc:`here </user_guide/ckernels>` here.


Profiling
---------

Especially during development, it can be very insightful to produce a trace of the CPU and GPU activities to find bottlenecks exist in the training.
To analyze this, the :func:`~hippynn.experiment.setup_and_profile` functions as a drop-in replacement for the :func:`~hippynn.experiment.setup_and_train` function.
The function accepts the same kwargs as :func:`~hippynn.experiment.setup_and_train`,
so users may quickly receive the robust feature offerings of the PyTorch profiler API with one function call.
This will output a json file which is compatible with the google chrome tracing tools.


Scaling
-------

hipppynn models and loss functions are just pytorch modules,
so you should be able to interface them to any accelerator package of choice.

We recommend pytorch lightning for scaling.
hippynn includes an interface to `Pytorch Lightning`_ for scaling to multiple GPUs or multiple nodes.
See the :doc:`documentation page </examples/lightning>` for more infromation.

There is legacy compatibility with the simple ``torch.nn.DataParallel`` by passing a list of device indices to
the SetupParams for an experiment, but this is not a strong strategy for scaling training because it is limited by
single-process CPU dispatch.

.. _`Pytorch Lightning`: https://lightning.ai/docs/pytorch/stable/
