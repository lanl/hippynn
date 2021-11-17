Installation
============



Requirements
^^^^^^^^^^^^

Requirements
    * Python_ >= 3.7
    * pytorch_ >= 1.9
Optional Dependencies:
    * numba_ (recommended, for accelerating network performance)
    * ASE_ (for usage with ase)
    * matplotlib_ (for plotting)
    * tqdm_ (for progress bars)
    * graphviz_ (for viewing models as graphs)
    * h5py_ (for ani-h5 datasets)
    * pyanitools_ (for ani-h5 datasets)

.. _Python: http://www.python.org
.. _pytorch: http://www.pytorch.org
.. _numba: https://numba.pydata.org/
.. _ASE: https://wiki.fysik.dtu.dk/ase/
.. _tqdm: https://tqdm.github.io/
.. _matplotlib: https://matplotlib.org
.. _graphviz: https://github.com/xflr6/graphviz
.. _h5py:  https://www.h5py.org/
.. _pyanitools: https://github.com/atomistic-ml/ani-al/tree/master/readers/lib

Installation
^^^^^^^^^^^^

Clone this repository and navigate into it::

    $ pip install .

If you feel like tinkering, do an editable install::

    $ pip install -e .

If you would like to get all dependencies from pip::

    $ pip install -e .[full]

Notes
^^^^^

- Install dependencies with pip from requirements.txt .
- Install dependencies with conda from conda_requirements.txt .
- If you don't want pip to install them, conda install from file before installing hippynn. You may want to use -c pytorch for the pytorch channel.
- Optional dependencies are in optional_dependencies.txt
