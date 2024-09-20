Building the documentation
--------------------------

These instructions are mostly for developers who would want to
modify and test their changes to documentation. For live
documentation on the web, see https://lanl.github.io/hippynn/

To install documentation dependencies,
run `pip install -e .[docs]` from the root hippynn directory.

Next, navigate to this directory.

If running for the first time, run `make apidoc` to build the
sphinx autodoc api documentation source files. `make cleanapi` will clean them.

Next, use `make` to build the documentation, e.g. `make html` for
html files.

You can clean the build directory with `make clean`, and you can clean the apidoc
files (if you add or remove a file or change apidoc settings) with make `cleanapi`.

You can clean both API files and build files with `make clean_all`, and you
can build both API files and then HTML using `make html_all`.




