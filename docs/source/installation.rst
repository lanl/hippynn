Installation
============


Requirements
^^^^^^^^^^^^

Requirements:
    * Python_ >= 3.9
    * pytorch_ >= 2.0
    * numpy_

Optional Dependencies:
    * triton_ (recommended, for improved GPU performance)
    * numba_ (recommended, for improved CPU/GPU performance)
    * cupy_ (alternative for accelerating GPU performance)
    * ASE_ (for usage with ase and other misc. features)
    * matplotlib_ (for plotting)
    * tqdm_ (for progress bars)
    * graphviz_ (for visualizing model graphs)
    * h5py_ (for loading ani-h5 datasets)
    * pytorch-lightning_ (for distributed training)
    * opt_einsum_ (backend for accelerating some pytorch expressions)

Interfacing codes:
    * ASE_
    * PYSEQM_
    * LAMMPS_

.. _triton: https://triton-lang.org/
.. _numpy: https://numpy.org/
.. _Python: http://www.python.org
.. _pytorch: http://www.pytorch.org
.. _numba: https://numba.pydata.org/
.. _cupy: https://cupy.dev/
.. _tqdm: https://tqdm.github.io/
.. _matplotlib: https://matplotlib.org
.. _graphviz: https://github.com/xflr6/graphviz
.. _h5py:  https://www.h5py.org/
.. _ASE: https://wiki.fysik.dtu.dk/ase/
.. _LAMMPS: https://www.lammps.org/
.. _PYSEQM: https://github.com/lanl/PYSEQM
.. _pytorch-lightning: https://github.com/Lightning-AI/pytorch-lightning
.. _hippynn: https://github.com/lanl/hippynn/
.. _opt_einsum: https://github.com/dgasmith/opt_einsum

Installation Instructions
^^^^^^^^^^^^^^^^^^^^^^^^^

We recommend installation from source, although you can also get hippynn
from conda and pypi, see below. A base installation of hippynn only requires
pytorch (`installation instructions <pytorch_install_>`_) and
numpy (`installation instructions <numpy_install_>`_).
We recommend you install these first using your package manager of choice,
then proceed to install hippynn.


Installing hippynn from source
------------------------------

Installing hippynn from source is an easy process.
hippynn is written in pure python, so you will not need to worry about
having compilers or interpreters for additional languages.

.. _pytorch_install: https://pytorch.org/get-started/locally/
.. _numpy_install: https://numpy.org/install/

Clone the hippynn_ repository and navigate into it, e.g.::

    $ git clone https://github.com/lanl/hippynn.git
    $ cd hippynn
    $ pip install -e .

The ``-e`` specifies an editable install, that is, python will import hippynn from
the current directory, which will allow you to tinker with hippynn if you so choose.

If numpy and pytorch are not currently installed, this command will install them using `pip`.

Once hippynn is installed, you can proceed to add optional packages as needed for
various extended functionality.

Install Optional Dependencies using conda
******************************************
Install dependencies from conda using recommended channels::

    $ conda install -c pytorch -c conda-forge --file conda_requirements.txt

.. note::
  If you wish to do a cpu-only install, you may need to comment
  out ``cupy`` from the conda_requirements.txt file.

Install Optional Dependencies using pip
******************************************

Pip will have already checked for the minimum requirements when you performed::

    $ pip install -e .


If you would like to get all optional dependencies from pip::

    $ pip install -e .[full]

On a recent mac with zsh instead of bash, this command must be escaped as::

    % pip install -e .\[full\]

Install hippynn from Conda
--------------------------
Install using conda::

    $ conda install -c conda-forge hippynn

Install hippynn from Pip
------------------------
Minimal Install using pip::

    $ pip install hippynn

All optional dependencies with pip::

    $ pip install hippynn[full]

On a recent mac with zsh instead of bash, this command must be escaped as::

    % pip install hippynn\[full\]


Note on installing cupy with conda
-----------------------------------

When using conda, sometimes both pytorch and cupy try to install cuda toolkit components,
but using different and conflicting mechanisms. If you encounter difficulties with using
a GPU after installing cupy, we recommend trying uninstalling it from conda and installing
it from pypi. `Link to cupy installation instructions <cupy_install>`_

.. _cupy_install: https://docs.cupy.dev/en/stable/install.html#installing-cupy-from-pypi

Misc. Notes
-----------

- Install dependencies with pip from requirements.txt .
- Install dependencies with conda from conda_requirements.txt .
- If you don't want pip to install them, conda install from file before installing ``hippynn``.
  You may want to use -c pytorch for the pytorch channel.
  For ase and cupy, you probably want to use -c conda-forge.
- Optional dependencies are in optional_dependencies.txt

