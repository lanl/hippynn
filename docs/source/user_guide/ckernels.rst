Custom Kernels
==============


Analogs of convolutional layers that apply to continously variable points in space, such as the
`HIP-NN` interaction layer, can be awkward to write in pure-pytorch.

The :mod:`~hippynn.custom_kernels` subpackage implements some more efficient kernels for both the forward
and backward pass of the sum over neighbors. This is implemented, more or less, as a CSR-type
sparse sum over the set of pairs of atoms, which, depending on the kernel, utilizes some
mixture of inner products and outer products on the remaining "feature" and "sensitivity" axes.
This behavior can be switched off (and is off by default if the dependencies are not installed)
to revert to a pure pytorch implementation.

The custom kernels provide `much` better memory footprint than the pure pytorch implementation,
and a decent amount of speedup on those core operations. The memory footprint of the pytorch
implementation is approximately:

.. math::

    O(N_\mathrm{pairs}N_\mathrm{sensitivities}N_\mathrm{features}),

whereas the memory footprint of the custom kernels is approximately

.. math::

    O(N_\mathrm{pairs}N_\mathrm{sensitivities} +
      N_\mathrm{atoms}N_\mathrm{features}N_\mathrm{sensitivities}).

The custom kernels are implemented using ``numba`` and/or ``cupy``, depending
on what is installed in your python environment.
However, there are certain overheads in using them.
In particular, if you are using a GPU and your batch size is small,
the pytorch implementations may actually be faster, because they launch more quickly.
This is especially true if you use a shallower model (one interaction layer) with
with a small number of elements, because the memory waste in a pure pytorch
implementation is proportional to the number of input features.
If you are using a CPU, the custom kernels are recommended at all times.

The three custom kernels correspond to the interaction sum in hip-nn:

For envsum, sensum, featsum:

.. math::

    e^{\nu}_{i,a} = \sum_p s^\nu_{p} z_{p_j,a}

    s_{p,\nu} = \sum_{a} e^{\nu}_{p_i,a} z_{p_j,a}

    f_{j,a} = \sum_{\nu,i} e_{p_i,\nu,a} s_{p_i,a}

Custom kernels can be set ahead of time using :doc:`/user_guide/settings` and dynamically
using :func:`~hippynn.custom_kernels.set_custom_kernels`.

