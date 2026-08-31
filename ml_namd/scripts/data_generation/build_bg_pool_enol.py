import numpy as np, os, shutil
L="/lustre/scratch5/akhanna2/ml_namd_project/data_gen/enol_balf_galf"
PAIRS={"12":(1,2),"13":(1,3),"23":(2,3)}
COPY=["acn_R","acn_Z"]+[f"acn_sE{s}" for s in range(4)]+[f"acn_F{s}" for s in range(4)]+[f"acn_dENACR{p}" for p in PAIRS]
for t in ["BALF","GALF"]:
    src=f"{L}/{t}/extracted/50K"; dst=f"{L}/{t}/pool_s0s3"; os.makedirs(dst,exist_ok=True)
    for k in COPY: shutil.copy(f"{src}/{k}.npy",f"{dst}/{k}.npy")
    F={s:np.load(f"{src}/acn_F{s}.npy") for s in range(4)}
    sE={s:np.load(f"{src}/acn_sE{s}.npy") for s in range(4)}
    N=F[0].shape[0]
    for p,(i,j) in PAIRS.items():
        np.save(f"{dst}/acn_GDV{p}.npy",(F[i]-F[j]).reshape(N,45).astype(np.float32))
        np.save(f"{dst}/acn_GAP{p}.npy",(sE[j]-sE[i]).astype(np.float64))
    g=np.load(f"{dst}/acn_GAP12.npy")
    print(f"{t}: pool_s0s3 N={N}  files={len(os.listdir(dst))}  GAP12 median={np.median(g):.3f} min={g.min():.4f} pct_lt0.1eV={100*(g<0.1).mean():.1f}")
