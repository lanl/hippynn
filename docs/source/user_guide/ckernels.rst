Custom Kernels
==============

Bottom line up front
--------------------

We use custom kernels in `hippynn` to accelerate the HIP-NN neural network message passing.
On the GPU, the best implementation to select is ``triton``, followed by ``cupy``,
followed by ``numba``. On the CPU, only ``numba`` is available. In general, these
custom kernels are very useful, and the only reasons for them to be off is if are
if the packages are not available for installation in your environment or if diagnosing
whether or not a bug could be related to potential misconfiguration of these additional packages.
``triton`` comes with recent versions of ``pytorch``, so optimistically you may already be
configured to use the custom kernels.

Detailed Explanation
--------------------
Analogs of convolutional layers that apply to continously variable points in space, such as the
HIP-NN interaction layer, can be awkward to write in pure-pytorch.

The :mod:`~hippynn.custom_kernels` subpackage implements some more efficient kernels for both the forward
and backward pass of the sum over neighbors. This is implemented, more or less, as a CSR-type
sparse sum over the set of pairs of atoms, which, depending on the kernel, utilizes some
mixture of inner products and outer products on the remaining "feature" and "sensitivity" axes.
This behavior can be switched off (and is off by default if the dependencies are not installed)
to revert to a pure pytorch implementation.

The custom kernels provide *much* better memory footprint than the pure pytorch implementation,
and a very good amount of speedup on those core operations. The memory footprint of the pytorch
implementation is approximately:

.. math::

    O(N_\mathrm{pairs}N_\mathrm{sensitivities}N_\mathrm{features}),

whereas the memory footprint of the custom kernels is approximately

.. math::

    O(N_\mathrm{pairs}N_\mathrm{sensitivities} +
      N_\mathrm{atoms}N_\mathrm{features}N_\mathrm{sensitivities}).

The custom kernels are implemented using ``triton``, ``cupy`` and/or ``numba``, depending
on what is installed in your python environment.
However, there are certain overheads in using them.
In particular, if you are using a GPU and your batch size is small,
the pytorch implementations may actually be faster, because they launch more quickly.
This is especially true if you use a shallow HIP-NN type model (one interaction layer) with
with a small number of elements, because the memory waste in a pure pytorch
implementation is proportional to the number of input features.
Nonetheless for most practical purposes, keeping custom kernels
on at all times is computationally recommended.
If you are using a CPU, the custom kernels are provided only using ``numba``, but they
do not come with any large overheads, and so provide computatonal benefits at all times.
The only reason to turn custom kernels off, in general, is to diagnose whether there are
issues with how they are being deployed; if ``numba`` or ``cupy`` is not correctly installed,
then we have found that sometimes the kernels may silently fail.

The three custom kernels correspond to the interaction sum in hip-nn:

.. math::

    a'_{i,a} = \sum_{\nu,b} V^\nu_{a,b} e^{\nu}_{i,b}

    e^{\nu}_{i,a} = \sum_p s^\nu_{p} z_{p_j,a}

Where :math:`a` is the pre-activation for an interaction layer using input features :math:`z`.

For envsum, sensesum, featsum:

.. math::

    e^{\nu}_{i,a} = \sum_p s^\nu_{p} z_{p_j,a}

    s_{p,\nu} = \sum_{a} e^{\nu}_{p_i,a} z_{p_j,a}

    f_{j,a} = \sum_{\nu,i} e_{p_i,\nu,a} s_{p_i,a}

These three functions form a closed system under automatic differentiation, and are linked to each
other in pytorch's autograd, thereby supporting custom kernels in backwards passes and in
double-backwards passes associated with Force training or similar features.

Custom kernels can be set ahead of time using :doc:`/user_guide/settings` and dynamically
using :func:`~hippynn.custom_kernels.set_custom_kernels`.

