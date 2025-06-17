Example files for HIPPYNN 
-------------------------

The files in this directory provide examples for hippynn.
Because hippynn is for training and running machine learning models,
it requires data to run! Our example files typically contain
a header which describes where to get the data in order to run the example.
Occasionally an example file for running a model depends on running a
different example for training that model.


Suggested Starting Points
-------------------------

- If you're looking to get a basic picture of how hippynn works, see
``barebones.py``, which is about as simple of an example as can function.
If you have a basic understanding but want to see more details,
check out the jupyter notebooks ``graph_exploration.ipynb``.

- If you're looking to customize a nicely constructed training script,
 look into ``ani1x_training.py``. This trains to a large dataset
 known as ani1x, and separates out the various aspects of the 
 library as well as demonstrating how they interact together.
 It has many optional features included. 

- If you want to see how to use a model with ase or lammps,
first train a model with ``ani_aluminum_example.py``. 
Then check out ``ase_example.py`` or the ``examples/lammps/``
directory respectively.