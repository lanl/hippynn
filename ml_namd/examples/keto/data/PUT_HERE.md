# Place the downloaded keto data here

Download the keto BALF pool from the Harvard Dataverse DOI (see `../00_download_data.md`) and place it at:

```
keto/balf/
```

so that this file's directory contains `keto/balf/acn_R.npy`, `acn_Z.npy`, the per-state `acn_sE*`/`acn_F*`,
and the per-pair `acn_dENACR{12,13,23}` / `acn_GAP{12,13,23}` / `acn_GDV{12,13,23}` arrays. The array
schema is in `../../../data_manifests/DATA_MANIFEST.md`.
