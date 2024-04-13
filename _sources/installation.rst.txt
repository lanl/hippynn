Installation
============



Requirements
^^^^^^^^^^^^

Requirements:
    * Python_ >= 3.9
    * pytorch_ >= 1.9
    * numpy_
Optional Dependencies:
    * numba_ (recommended, for accelerating performance)
    * cupy_ (also for accelerating performance)
    * ASE_ (for usage with ase)
    * matplotlib_ (for plotting)
    * tqdm_ (for progress bars)
    * graphviz_ (for viewing model graphs as figures)
    * h5py_ (for ani-h5 datasets)
    * pyanitools_ (for ani-h5 datasets)

Interfacing codes:
    * ASE_
    * PYSEQM_
    * LAMMPS_

.. _numpy: https://numpy.org/
.. _Python: http://www.python.org
.. _pytorch: http://www.pytorch.org
.. _numba: https://numba.pydata.org/
.. _cupy: https://cupy.dev/
.. _tqdm: https://tqdm.github.io/
.. _matplotlib: https://matplotlib.org
.. _graphviz: https://github.com/xflr6/graphviz
.. _h5py:  https://www.h5py.org/
.. _pyanitools: https://github.com/atomistic-ml/ani-al/tree/master/readers/lib
.. _ASE: https://wiki.fysik.dtu.dk/ase/
.. _LAMMPS: https://www.lammps.org/
.. _PYSEQM: https://github.com/lanl/PYSEQM


Installation Instructions
^^^^^^^^^^^^^^^^^^^^^^^^^

Conda
-----
Install using conda::

    conda install -c conda-forge hippynn

Pip
---
Install using pip::

    pip install hippynn

Install from source:
--------------------

Clone the hippynn_ repository and navigate into it, e.g.::

    $ git clone https://github.com/lanl/hippynn.git
    $ cd hippynn

.. _hippynn: https://github.com/lanl/hippynn/

.. note::
  If you wish to do a cpu-only install, you may need to comment
  out ``cupy`` from the conda_requirements.txt file.

Dependencies using conda
........................

Install dependencies from conda using recommended channels::

    $ conda install -c pytorch -c conda-forge --file conda_requirements.txt

Dependencies using pip
.......................

Minimum dependencies using pip::

    $ pip install -e .

If you feel like tinkering, do an editable install::

    $ pip install -e .

If you would like to get all optional dependencies from pip::

    $ pip install -e .[full]


Notes
-----

- Install dependencies with pip from requirements.txt .
- Install dependencies with conda from conda_requirements.txt .
- If you don't want pip to install them, conda install from file before installing ``hippynn``.
  You may want to use -c pytorch for the pytorch channel.
  For ase and cupy, you probably want to use -c conda-forge.
- Optional dependencies are in optional_dependencies.txt

