# Model B mathematical background: learning scaled nonadiabatic couplings (dENACR)

This document derives the quantity Model B learns, why it is defined the way it is, the explicit
equation the v3 network head uses to build it, how the training loss handles the electronic-phase
ambiguity, and how the learned quantity is turned back into a surface-hopping coupling at inference. It
is the theory companion to `../scripts/training/direct_nacr_head_v3.py` (the head),
`../scripts/training/joint_state_nacr_loss.py` (the sign-consistent loss),
`../scripts/training/unified_training.py` (the training driver, Block D), and the composed A to B
predictor used by the FSSH@MLIP driver in `../scripts/fssh_mlip/`.

Notation: excited-state indices `i, j` in `{1, ..., n_exc}` (S0 is index 0); atom indices `A, B`;
nuclear coordinates `R` of shape `(n_at, 3)`; adiabatic electronic states $\psi_k(R)$ with energies
`E_k(R)`; the electronic Hamiltonian $\hat H(R)$. For acetylacetone `n_at = 15`, so a per-atom vector
flattens to length `3 n_at = 45`.

---

## 1. The object of interest and its singularity

Nonadiabatic dynamics is driven by the **derivative coupling vector** between two adiabatic states,

```math
\mathbf{d}_{ij}(R) = \langle \psi_i(R) \mid \nabla_R \psi_j(R) \rangle \in \mathbb{R}^{n_\text{at} \times 3}.
```

Differentiating the eigenvalue equation $\hat H \psi_j = E_j \psi_j$ and projecting onto $\psi_i$ (the
off-diagonal Hellmann-Feynman relation) gives, for `i != j`,

```math
\mathbf{d}_{ij}(R) = \frac{\langle \psi_i \mid (\nabla_R \hat H) \mid \psi_j \rangle}{E_j(R) - E_i(R)}.
```

The numerator is a smooth matrix element of the (smooth) Hamiltonian gradient. The denominator is the
**energy gap** $\Delta_{ij} = E_j - E_i$, which vanishes at a conical intersection (CI). Therefore
$\mathbf{d}_{ij}$ **diverges as $1/\Delta_{ij}$ at the exact places nonadiabatic transitions happen**. A
network asked to regress $\mathbf{d}_{ij}$ directly must reproduce an unbounded, sharply peaked function
— the hardest possible target, and one no single smooth representation fits together with the smooth
energies.

## 2. The learnable target: the scaled coupling (dENACR)

Multiply the singular coupling by the gap to cancel the pole:

```math
\mathbf{D}_{ij}(R)  \equiv  \Delta_{ij}  \mathbf{d}_{ij}  =  \langle \psi_i \mid (\nabla_R \hat H) \mid \psi_j \rangle .
```

We call $\mathbf{D}_{ij}$ the **scaled nonadiabatic coupling**, or **dENACR** (`acn_dENACR{ij}` in the
data, flattened to length 45). It is the numerator above — a matrix element of the Hamiltonian gradient
— so it is **finite and smooth through a CI**, where $\mathbf{d}_{ij}$ itself blows up. Model B learns
$\mathbf{D}_{ij}$, not $\mathbf{d}_{ij}$; the singular $1/\Delta$ factor is reintroduced analytically at
inference (Section 6) rather than fit.

This is the central design decision: learn the regular quantity, divide by an independently predicted
gap, and keep the network target bounded and matched in character to the energies and forces the same
pipeline already predicts.

## 3. The branching plane and the physical inputs (GDV, GAP)

Near a CI the topography is controlled by two vectors spanning the **branching plane**:

- the **gradient-difference vector** $\mathbf{g}_{ij} = \tfrac{1}{2}\nabla_R (E_i - E_j)$,
- the **derivative-coupling (interstate) vector** $\mathbf{h}_{ij}$, proportional to $\mathbf{D}_{ij}$.

Two facts make $\mathbf{g}$ a powerful **input** for predicting $\mathbf{D}$:

1. It is available cheaply from state forces. With $\mathbf{F}_k = -\nabla_R E_k$,

```math
\mathbf{F}_i - \mathbf{F}_j  =  -\nabla E_i + \nabla E_j  =  \nabla (E_j - E_i)  =  -2 \mathbf{g}_{ij}.
```

   The difference of state forces is stored directly as the **GDV** input
   (`acn_GDV{ij}` = $\mathbf{F}_i - \mathbf{F}_j$, flattened to 45, float32) — the gradient of the gap, up
   to sign, i.e. the $\mathbf{g}$ direction of the branching plane.
2. The **GAP** (`acn_GAP{ij}` = $\Delta_{ij} = E_j - E_i$, eV) is the invariant scalar that sets the
   magnitude scale (through the $1/\Delta$ reconstruction) and localizes the near-degeneracy.

Model B is therefore the map

```math
\mathbf{D}_{ij}  =  \mathcal{B}(Z,\ R,\ \mathbf{GDV}_{ij},\ \mathrm{GAP}_{ij}),
```

taking the geometry plus the two branching-plane quantities the energy/force model already produces. At
training time GDV and GAP come from the reference labels; at inference they come from **Model A** (the
E+F network), which is why the deployed system is a *composed* A to B stack (Section 6).

## 4. How the v3 head builds dENACR (the construction equation)

The head is equivariant: a rotation of the molecule rotates the predicted coupling identically. It
assembles a per-atom vector from atom-pair (edge) contributions, then conditions it on the gap. Two
scalar, invariant coefficients are produced for each network edge (atoms `A`, `B` within a cutoff) from
the message-passing features and the pair distance $r_{AB}$: a GDV-mixing coefficient $c^{g}_{AB}$ and a
geometry coefficient $c^{r}_{AB}$ (symmetric `HBondSymmetric` layers).

The production mode `gdv_pair_mixing_residual` builds the per-atom vector $\mathbf{v}_A$ from a sum over
that atom's edges of a **GDV-difference** term and a **geometry** (relative-position) term:

```math
\mathbf{v}_A  =  s \sum_{B \sim A} [  c^{g}_{AB} (\mathbf{GDV}_A - \mathbf{GDV}_B)  +  c^{r}_{AB} (R_A - R_B)  ],
```

with edge scale `s = 0.5`, and each edge added antisymmetrically to its two endpoints (so the pair
`(A, B)` contributes `+` to `A` and `-` to `B`). This is the equation the user shorthand
`c_alpha (R_A - R_B) + c_beta (F_i - F_j)` refers to: `c_alpha = s c^{r}` multiplies the geometry pair
vector, and `c_beta = s c^{g}` multiplies the GDV (`GDV = F_i - F_j`), here entering as per-atom GDV
differences along network edges. Using **differences** on edges makes both terms translationally
covariant and ties the coupling direction to the local branching-plane geometry. (The `gdv_pair_mixing`
mode drops the geometry term; `gdv_local_gate` replaces the edge mixing with a per-atom gate on each
atom's own GDV — see the head's `V3_MODES`.)

The assembled vector is then **gap-conditioned** by a multiplicative gate and **zero-sum projected**:

```math
\mathbf{v}_A  \leftarrow  [ 1 + \mathrm{MLP}(\Delta_{ij}) ]  \mathbf{v}_A , \qquad
\mathbf{v}_A  \leftarrow  \mathbf{v}_A - \frac{1}{n_\text{at}} \sum_{B} \mathbf{v}_B .
```

The MLP's final layer is **zero-initialized**, so at initialization the gate equals 1 and v3 is
identical to its gapless predecessor (v2); it then learns a gap-dependent magnitude correction. Because
the gap is a per-system invariant, multiplying by it preserves rotational equivariance. The zero-sum
projection removes any net translational component (a true interstate coupling has zero total).

Finally the per-atom vectors are flattened into the predicted scaled coupling:

```math
\mathbf{D}_{ij}  =  \mathrm{flatten}_A(\mathbf{v}_A)  \in  \mathbb{R}^{3 n_\text{at}} .
```

This $\mathbf{D}_{ij}$ is exactly the quantity trained against the label `acn_dENACR{ij}` (Section 5).

## 5. The electronic-phase ambiguity and the training loss

Adiabatic states are defined only up to a sign (more generally a phase): replacing
$\psi_k \to \pm \psi_k$ leaves all physical observables unchanged but flips the sign of every coupling
that touches state `k`. Concretely a per-state sign set $s_k \in \{+1, -1\}$ acts as

```math
\mathbf{D}_{ij}  \to  s_i  s_j  \mathbf{D}_{ij}.
```

The reference labels come from independent single points, so their signs are arbitrary and **not
consistent from geometry to geometry**. A naive MSE would penalize a sign the network cannot know. Two
losses address this; both are in `unified_training.py` Block D.

**Per-pair phaseless loss (baseline).** Resolve each pair's sign independently, per sample, taking the
better of the two signs:

```math
\ell^{\text{phaseless}}_{ij}  =  \frac{\mathrm{RMSE}_{ij}}{\sqrt{3 n_\text{at}}} + \mathrm{MAE}_{ij},
\qquad
\mathrm{err}_{ij}  =  \min_{\sigma \in \{+1,-1\}} \lVert  \mathbf{D}^{\text{ML}}_{ij} - \sigma  \mathbf{D}^{\text{QC}}_{ij}  \rVert,
```

with the per-sample sign chosen from the 45-vector norm (hippynn `MSEPhaseLoss` / `MAEPhaseLoss`).
Robust and simple, but it grants **each pair its own sign**. For `n_exc` states there are
`n_exc (n_exc - 1) / 2` pairs but only `n_exc - 1` independent state signs, so the per-pair solution can
land in a set of signs that corresponds to **no single per-state gauge** (`r_ij != s_i s_j`) — a
"pair-scramble." The surface-hopping driver can only re-align a consistent per-state gauge, so a
pair-scramble survives into the dynamics.

**Joint per-state phase loss (sign-consistent).** Enforce a single global gauge shared by all pairs:

```math
\mathcal{L}^{\text{joint}}  =  \frac{1}{n_\text{pairs}}  \mathbb{E}_\text{batch}[  \min_{\{ s_k \in \{\pm 1\},\ s_1 = +1 \}} \sum_{i < j} \lVert  \mathbf{D}^{\text{ML}}_{ij} - s_i s_j  \mathbf{D}^{\text{QC}}_{ij}  \rVert^2  ].
```

The bilinear factor $s_i s_j$ ties every pair to one of $2^{ n_\text{exc}-1}$ candidate per-state
gauges (the first sign fixed), re-imposing the cocycle constraints the per-pair loss drops. This is
exactly the construction the FSSH driver applies at inference (`_phase_match`), here applied against the
label during training (`joint_state_nacr_loss.py`).

**A measured caveat (loss form vs gauge).** The joint loss above is a **pure sum of squares**, whereas
the phaseless blend is `RMSE / sqrt(3 n_at) + MAE` (about 70% L1). A pure-L2 objective weights the large
near-CI $\lVert \mathbf{D} \rVert$ values quadratically and was observed to inflate the S1-S2 coupling
magnitude relative to the L1-leaning blend. When comparing sign strategies, match the loss *form* (or
the magnitude difference is attributed to the gauge when it is really the norm). A matched-form
factorial (per-pair vs joint gauge, at equal loss form) isolates the two effects.

## 6. Inference: reconstructing the hopping driver

Deployment is a **composed A to B** evaluation (`modelb_composed_predictor` in
`../scripts/fssh_mlip/enol_ef_nacr_developer_handoff/src/`, driven by `hippynn_fssh_composed.py`):

1. **Model A** (frozen E+F network) predicts state energies `E_k` and forces $\mathbf{F}_k$, giving
   $\mathrm{GAP}_{ij} = E_j - E_i$ and $\mathbf{GDV}_{ij} = \mathbf{F}_i - \mathbf{F}_j$.
2. **Model B** predicts the scaled coupling
   $\mathbf{D}_{ij} = \mathcal{B}(Z, R, \mathbf{GDV}_{ij}, \mathrm{GAP}_{ij})$.
3. The **derivative coupling** is reconstructed by dividing by the *same* gap:

```math
\mathbf{d}_{ij}  =  \frac{\mathbf{D}_{ij}}{\Delta_{ij}}  =  \frac{\mathbf{D}_{ij}}{E_j - E_i}.
```

4. Signs are made temporally consistent by the per-state `_phase_match` against the previous step, and
   the **time-derivative coupling** that drives hops is the velocity projection

```math
\sigma_{ij}  =  \mathbf{d}_{ij} \cdot \mathbf{v}  =  \sum_{A} \sum_{\alpha \in \{x,y,z\}} d_{ij,A\alpha}  v_{A\alpha}.
```

**Critical numerical rule (no gap floor).** The division in step 3 uses the **raw** gap, with no floor
and no clamp. Flooring $\Delta$ (an earlier `max(Delta, 1e-8)`) inverts small or slightly-negative MLIP
gaps, explodes $\mathbf{d}_{ij}$ to about `1e8` and $\sigma_{ij}$ to about `1e6`, and produces amplitude
NaNs that freeze hopping. The physical gap magnitude near the seam is finite (about `1e-2` eV), so the
raw division is well posed; see the driver comments in `../scripts/fssh_mlip/hippynn_fssh.py`.

## 7. Units and shapes

| Symbol | Name | Array | Shape | Units |
|---|---|---|---|---|
| `R` | positions | `acn_R` | `(N, n_at, 3)` | Angstrom |
| `Z` | species | `acn_Z` | `(N, n_at)` | — |
| `E_k` | state energy (shifted) | `acn_sE{k}` | `(N,)` | eV |
| `F_k` | state force | `acn_F{k}` | `(N, n_at, 3)` | eV/Angstrom |
| `GAP_ij` | gap `E_j - E_i` | `acn_GAP{ij}` | `(N,)` | eV |
| `GDV_ij` | `F_i - F_j` | `acn_GDV{ij}` | `(N, 45)` | eV/Angstrom |
| `D_ij` | scaled coupling (target) | `acn_dENACR{ij}` | `(N, 45)` | eV/Angstrom |
| `d_ij` | derivative coupling (reconstructed) | — | `(N, n_at, 3)` | 1/Angstrom |
| `sigma_ij` | time-derivative coupling | — | `(N,)` | 1/fs |

The pairs trained are `{12, 13, 23}` (S1-S2, S1-S3, S2-S3); S0 does not couple in this manifold.

## 8. Summary of the design logic

1. The derivative coupling is singular at conical intersections; **scaling by the gap removes the pole**
   and yields a smooth, bounded target ($\mathbf{D}_{ij}$, dENACR).
2. The gap and the gradient-difference vector **span the branching plane**; supplying GAP and GDV as
   inputs feeds the network the physics that sets the coupling's scale and direction.
3. The **v3 head** builds an equivariant, zero-sum per-atom vector as `s * sum_edges [ c_g (GDV_A - GDV_B)
   + c_r (R_A - R_B) ]`, then applies a zero-initialized **gap gate** (so v3 reduces to v2 at init).
4. The electronic phase is arbitrary; the **joint per-state loss** enforces a single consistent gauge the
   surface-hopping driver can honor, unlike the per-pair phaseless baseline.
5. At inference, **Model A supplies GAP and GDV**, Model B supplies $\mathbf{D}_{ij}$, and the coupling is
   reconstructed as $\mathbf{D}_{ij}/\Delta_{ij}$ with the **raw gap (no floor)**, then dotted with the
   velocity to drive fewest-switches surface hopping.

See `model_b_usage_guide.md` for the practical how-to (training a Model B, wiring it into the composed
FSSH driver, and the input/output contract).
