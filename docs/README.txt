
Building the documentation
--------------------------

Get `sphinx` and `sphinx-rtd-theme` from pip.

Navigate to this directory.

If running for the first time, run `make apidpc` to build the
sphinx autodoc api documentation source files.

Next, use `make` to build the documentation, e.g. `make html` for
html files.

You can clean the build directory with `make clean`, and you can clean the apidoc
files (if you add or remove a file or change apidoc settings) with make `cleanapi`.

