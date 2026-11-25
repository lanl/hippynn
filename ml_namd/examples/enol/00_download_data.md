# Download the enol data (Harvard Dataverse)

The datasets are not stored in this repository. Download them from:

> Khanna, Ajay, 2026, *Replication Data for: Machine Learning for Non Adiabatic Excited State Molecular
> Dynamics*, https://doi.org/10.7910/DVN/HCL8EK, Harvard Dataverse.

From that record, fetch the **enol BALF** training pool and the **enol 1 ps initial conditions**, and place
them like this (relative to this example directory):

```
data/enol/balf/        # the BALF pool: acn_R.npy, acn_Z.npy, acn_sE*.npy, acn_F*.npy,
                       #   acn_dENACR{12,13,23}.npy, acn_GAP{12,13,23}.npy, acn_GDV{12,13,23}.npy
ics/enol1ps_R0.npy     # initial coordinates
ics/enol1ps_V0.npy     # initial velocities
ics/enol1ps_Z.npy      # atomic numbers
```

See `../../data_manifests/DATA_MANIFEST.md` for the array schema and conventions. Once the files are in
place, continue with `02_train_model_A.py`.
