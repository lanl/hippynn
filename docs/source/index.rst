:tocdepth: 4

Welcome to hippynn's documentation!
===================================

We hope you enjoy your stay.

What is hippynn?
================

`hippynn` is a python library for machine learning on atomistic systems.
We aim to provide high-performance modular design so that different
components can be re-used, extended, or added to. You can find more information
at the :doc:`/user_guide/features` page. The development home is located
at `the hippynn github repository`_, which also contains `many example files`_

The main components of hippynn are constructing models, loading databases,
training the models to those databases, making predictions on new databases,
and interfacing with other atomistic codes. In particular, we provide interfaces
to `ASE`_ (prediction), `PYSEQM`_ (training/prediction), and `LAMMPS`_ (prediction).
hippynn is also used within `ALF`_ for generating machine learned potentials
along with their training data completely from scratch.

Multiple formats for training data are supported, including
Numpy arrays, the ASE Database, `fitSNAP`_ JSON format, and `ANI HDF5 files`_.

.. _`ASE`: https://wiki.fysik.dtu.dk/ase/
.. _`PYSEQM`: https://github.com/lanl/PYSEQM/
.. _`LAMMPS`: https://www.lammps.org
.. _`fitSNAP`: https://github.com/FitSNAP/FitSNAP
.. _`ANI HDF5 files`: https://doi.org/10.1038/s41597-020-0473-z
.. _`ALF`: https://github.com/lanl/ALF/

.. _`the hippynn github repository`: https://github.com/lanl/hippynn/
.. _`many example files`: https://github.com/lanl/hippynn/tree/development/examples


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

