"""
This example is intended to illustrate things people commonly want to do after training a hippynn model. 
This includes:
    1. Loading a saved model from a directory
    2. Creating a 'Predictor' object from the model that allows easy evaluation of the model
    3. Appling the Predictor directly to a database and extracting predicted values
    4. Appling the Predictor object to pytorch Tensors and extracting the predicted values
    5. Making a parity plot to visualize the predicted vs. true values

Also see https://lanl.github.io/hippynn/examples/predictor.html for additional information about how to use
the Predictor object. 
"""

import os

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from hippynn.experiment.serialization import load_checkpoint_from_cwd
from hippynn.tools import active_directory
from hippynn.graphs.predictor import Predictor

model_dir = "../TEST_MY_FIRST_QM7_MODEL"

if not os.path.exists(model_dir):
    raise ValueError(f"Please run the example script 'QM7_example.py' first to train the model used in this example.")

# RELOAD CHECKPOINT
with active_directory(model_dir):
    check = load_checkpoint_from_cwd(model_device='cpu', restart_db=True) 
    # alternatively, one can use 'restart_db=False' and manually load a different database with the same db_name for each input variable

# EXTRACT MODEL AND DATABASE
model = check['training_modules'].model
database = check['database'] # available if restart_db=True

# CREATE PREDICTOR OBJECT
predictor = Predictor.from_graph(model)

# APPLY MODEL TO GET PREDICTED VALUES
# ====METHOD 1=====
out = predictor.apply_to_database(database)
pred_T = out["test"]["T"] # 'test' refers to the data split, 'T' is the db_name of the output
# =================

# ====METHOD 2=====
# Alternatively, one can call the predictor function directly. The arguments are the db_names used for inputs when the model was created
R = database.splits["test"]["R"] # These can even be a tensor you construct yourself of shape (batch_size, n_atoms_max, 3)
Z = database.splits["test"]["Z"] # Shape (batch_size, n_atoms_max)
# If the database has not been split, use instead use: R = database.arr_dict["R"]; Z = database.arr_dict["Z"]
out = predictor(R=R, Z=Z)
pred_T = out["T"]
# =================

# EXTRACT TRUE VALUES FOR COMPARISON
true_T = database.splits["test"]["T"] 

# PLOT RESULTS
# Now we can recreate the parity plot if we wish
true_T = true_T.data.cpu().numpy().flatten()
pred_T = pred_T.data.cpu().numpy().flatten()
plt.hist2d(true_T, pred_T, bins=200, norm=mcolors.LogNorm())

min_val = min(true_T.min(), pred_T.min())
max_val = max(true_T.max(), pred_T.max())
plt.plot((min_val, max_val), (min_val, max_val), c="r", lw=0.5)

plt.xlabel("true_T")
plt.ylabel("pred_T")
plt.colorbar()

plt.show()

