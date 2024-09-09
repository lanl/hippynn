:tocdepth: 4

Welcome to hippynn's documentation!
===================================

We hope you enjoy your stay.

What is hippynn?
================

``hippynn`` is a python library for machine learning on atomistic systems
using `pytorch`_.
We aim to provide high-performance modular design so that different
components can be re-used, extended, or added to. You can find more information
about overall library features at the :doc:`/user_guide/features` page.
The development home is located at `the github github repository`_, which also contains `many example files`_.
Additionally, the :doc:`user guide </user_guide/index>` aims to describe abstract
aspects of the library, while the
:doc:`examples documentation section </examples/index>` aims to show
more concretely how to perform tasks with hippynn. Finally, the
:doc:`api documentation </api_documentation/hippynn>` contains a comprehensive
listing of the library components and their documentation.

The main components of hippynn are constructing models, loading databases,
training the models to those databases, making predictions on new databases,
and interfacing with other atomistic codes for operations such as molecular dynamics.
In particular, we provide interfaces to `ASE`_ (prediction),
`PYSEQM`_ (training/prediction), and `LAMMPS`_ (prediction).
hippynn is also used within `ALF`_ for generating machine learned potentials
along with their training data completely from scratch.

Multiple :doc:`database formats </user_guide/databases>` for training data are supported, including
Numpy arrays, `ASE`_-compatible formats, `FitSNAP`_ JSON format, and `ANI HDF5 files`_.

``hippynn`` includes many tools, such as an :doc:`ASE calculator</examples/ase_calculator>`,
a :doc:`LAMMPS MLIAP interface</examples/mliap_unified>`,
:doc:`batched prediction </examples/predictor>` and batched geometry optimization,
:doc:`automatic ensemble creation </examples/ensembles>`,
:doc:`restarting training from checkpoints </examples/restarting>`,
:doc:`sample-weighted loss functions </examples/weighted_loss>`,
:doc:`distributed training with pytorch lightning </examples/lightning>`,
and more.

``hippynn`` is highly modular, and if you are a model developer, interfacing your
pytorch model into the hippynn node/graph system will make it simple and easy for users
to build models of energy, charge, bond order, excited state energies, and more.

.. _`ASE`: https://wiki.fysik.dtu.dk/ase/
.. _`PYSEQM`: https://github.com/lanl/PYSEQM/
.. _`LAMMPS`: https://www.lammps.org
.. _`FitSNAP`: https://github.com/FitSNAP/FitSNAP
.. _`ANI HDF5 files`: https://doi.org/10.1038/s41597-020-0473-z
.. _`ALF`: https://github.com/lanl/ALF/

.. _`the github github repository`: https://github.com/lanl/hippynn/
.. _`many example files`: https://github.com/lanl/hippynn/tree/development/examples
.. _`pytorch`: https://pytorch.org


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   How to install hippynn <installation>
   Examples <examples/index>
   User Guide <user_guide/index>
   hippynn API documentation <api_documentation/hippynn>
   license

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

