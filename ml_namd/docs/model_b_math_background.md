# Model B mathematical background: learning scaled nonadiabatic couplings (dENACR)

This document derives the quantity Model B learns, why it is defined the way it is, how the v3
network head constructs it, how the training loss handles the electronic-phase ambiguity, and how the
learned quantity is turned back into a surface-hopping coupling at inference. It is the theory
companion to `../scripts/training/direct_nacr_head_v3.py` (the head),
`../scripts/training/joint_state_nacr_loss.py` (the sign-consistent loss),
`../scripts/training/unified_training.py` (the training driver, Block D), and the composed A→B
predictor used by the FSSH@MLIP driver in `../scripts/fssh_mlip/`.

Notation: excited-state indices $i, j \in \{1,\dots,n_\text{exc}\}$ (S0 is index $0$); nuclear
coordinates $\mathbf{R}\in\mathbb{R}^{n_\text{at}\times 3}$; adiabatic electronic states
$|\psi_k(\mathbf{R})\rangle$ with energies $E_k(\mathbf{R})$; the electronic Hamiltonian
$\hat H(\mathbf{R})$. For acetylacetone $n_\text{at}=15$, so a per-atom vector flattens to length
$3 n_\text{at}=45$.

---

## 1. The object of interest and its singularity

Nonadiabatic dynamics is driven by the **derivative coupling vector** between two adiabatic states,

$$
\mathbf{d}_{ij}(\mathbf{R}) \;=\; \big\langle \psi_i(\mathbf{R}) \,\big|\, \nabla_{\mathbf{R}} \,\psi_j(\mathbf{R}) \big\rangle
\;\in\; \mathbb{R}^{n_\text{at}\times 3}.
$$

Differentiating the eigenvalue equation $\hat H|\psi_j\rangle = E_j|\psi_j\rangle$ and projecting onto
$\langle\psi_i|$ (the off-diagonal Hellmann–Feynman relation) gives, for $i\neq j$,

$$
\mathbf{d}_{ij}(\mathbf{R}) \;=\; \frac{\big\langle \psi_i \,\big|\, (\nabla_{\mathbf{R}}\hat H)\, \big|\, \psi_j \big\rangle}{E_j(\mathbf{R}) - E_i(\mathbf{R})}.
$$

The numerator is a smooth matrix element of the (smooth) Hamiltonian gradient. The denominator is the
**energy gap** $\Delta_{ij} = E_j - E_i$, which vanishes at a conical intersection (CI). Therefore
$\mathbf{d}_{ij}$ **diverges as $1/\Delta_{ij}$ at the exact places nonadiabatic transitions happen**.
A network asked to regress $\mathbf{d}_{ij}$ directly must reproduce a function that is unbounded and
sharply peaked near seams — the hardest possible target, and one that no single smooth representation
fits well together with the smooth energies.

## 2. The learnable target: the scaled coupling (dENACR)

Multiply the singular coupling by the gap to cancel the pole:

$$
\boxed{\;\mathbf{D}_{ij}(\mathbf{R}) \;\equiv\; \Delta_{ij}\,\mathbf{d}_{ij} \;=\; \big\langle \psi_i \,\big|\, (\nabla_{\mathbf{R}}\hat H)\, \big|\, \psi_j \big\rangle \;}
$$

We call $\mathbf{D}_{ij}$ the **scaled nonadiabatic coupling**, or **dENACR** (`acn_dENACR{ij}` in the
data, flattened to length $45$). It is the numerator above — a matrix element of the Hamiltonian
gradient — so it is **finite and smooth through a CI**, where $\mathbf{d}_{ij}$ itself blows up. Model
B learns $\mathbf{D}_{ij}$, not $\mathbf{d}_{ij}$; the singular $1/\Delta$ factor is reintroduced
analytically at inference (Section 6) rather than fit.

This is the central design decision. Learning the regular quantity and dividing by an
independently-predicted gap keeps the network target bounded and matched in character to the energies
and forces the same pipeline already predicts.

## 3. The branching plane and the physical inputs (GDV, GAP)

Near a CI the topography is controlled by two vectors that span the **branching plane**:

- the **gradient-difference vector** $\mathbf{g}_{ij} = \tfrac12 \nabla_{\mathbf{R}}(E_i - E_j)$,
- the **derivative-coupling (interstate) vector** $\mathbf{h}_{ij}$, proportional to $\mathbf{D}_{ij}$.

Two facts make $\mathbf{g}$ a powerful **input** for predicting $\mathbf{D}$:
1. $\mathbf{g}_{ij}$ is available cheaply from state forces. With $\mathbf{F}_k = -\nabla_{\mathbf{R}} E_k$,

$$
\mathbf{F}_i - \mathbf{F}_j \;=\; -\nabla E_i + \nabla E_j \;=\; \nabla(E_j - E_i) \;=\; -2\,\mathbf{g}_{ij}.
$$

   In this project the difference of state forces is stored directly as the **GDV** input
   (`acn_GDV{ij}` $=\mathbf{F}_i-\mathbf{F}_j$, flattened to $45$, float32); it is the gradient of the
   gap up to sign, i.e. the $\mathbf{g}$ direction of the branching plane.
2. The gap itself, **GAP** (`acn_GAP{ij}` $=\Delta_{ij}=E_j-E_i$, eV), is the invariant scalar that
   sets the magnitude scale (through the $1/\Delta$ reconstruction) and localizes the near-degeneracy.

Model B is therefore a map

$$
\mathbf{D}_{ij} \;=\; \mathcal{B}\big(\mathbf{Z}, \mathbf{R}, \; \mathbf{GDV}_{ij}, \; \mathrm{GAP}_{ij}\big),
$$

taking the geometry plus the two branching-plane quantities the energy/force model already produces.
At training time GDV and GAP come from the reference labels; at inference they come from **Model A**
(the E+F network), which is why the deployed system is a *composed* A→B stack (Section 6).

## 4. The v3 network head: an equivariant, gap-conditioned construction

The head (`DirectScaledNACRV3Layer`) builds $\mathbf{D}_{ij}$ as an **equivariant per-atom vector**,
so that a rotation of the molecule rotates the predicted coupling identically. It is assembled from
edge (atom-pair) contributions and then conditioned on the gap. The production mode is
`gdv_pair_mixing_residual`; the pieces are:

**(a) Edge mixing of the GDV.** For each network edge (atom pair $a\!-\!b$ within a cutoff) a scalar,
invariant coefficient $c_{ab}$ is produced from the message-passing features and the pair distance (a
symmetric `HBondSymmetric` layer). It multiplies the **difference** of the two atoms' GDV vectors and
is added antisymmetrically to the two endpoints:

$$
\mathbf{v}_a \mathrel{+}= s\, c_{ab}\,(\mathbf{GDV}_b - \mathbf{GDV}_a), \qquad
\mathbf{v}_b \mathrel{-}= s\, c_{ab}\,(\mathbf{GDV}_b - \mathbf{GDV}_a),
$$

with edge scale $s=0.5$. Using GDV *differences* on edges makes the term translationally covariant and
ties the coupling direction to the local branching-plane geometry.

**(b) Geometry residual.** A second symmetric coefficient multiplies the relative pair coordinate
$\mathbf{r}_{ab}=\mathbf{r}_b-\mathbf{r}_a$ and is added the same antisymmetric way, giving the head a
geometry-only channel alongside the GDV channel.

**(c) Zero-sum projection.** The per-system atom mean is removed,
$\mathbf{v}_a \leftarrow \mathbf{v}_a - \tfrac{1}{n_\text{at}}\sum_b \mathbf{v}_b$, so the coupling
carries no net translational component (a true interstate coupling has zero total).

**(d) Gap conditioning.** The invariant scalar gap passes through a small MLP whose final layer is
**zero-initialized**, giving a multiplicative gate

$$
\text{gate}(\Delta_{ij}) = 1 + \mathrm{MLP}(\Delta_{ij}), \qquad \mathbf{v}_a \leftarrow \text{gate}(\Delta_{ij})\,\mathbf{v}_a.
$$

At initialization $\text{gate}\equiv 1$, so v3 is **identical to its gapless predecessor (v2) at step 0**
and only learns a gap-dependent magnitude correction during training. Because the gap is a per-system
invariant, multiplying by it preserves rotational equivariance.

The per-atom vectors are scattered back to a dense $(\text{n\_systems}, 3 n_\text{at})$ output — the
predicted flattened $\mathbf{D}_{ij}$. (Modes `gdv_local_gate` and `gdv_pair_mixing` drop the residual
and/or replace the edge mixing with a per-atom gate; see the head's `V3_MODES`.)

## 5. The electronic-phase ambiguity and the training loss

Adiabatic states are defined only up to a sign (more generally a phase): replacing
$|\psi_k\rangle \to \pm|\psi_k\rangle$ leaves all physical observables unchanged but flips the sign of
every coupling that touches state $k$. Concretely a per-state sign set $s_k\in\{+1,-1\}$ acts as

$$
\mathbf{D}_{ij} \;\longrightarrow\; s_i\, s_j\, \mathbf{D}_{ij}.
$$

The reference labels come from independent single points, so their signs are arbitrary and **not
consistent from geometry to geometry**. A naive MSE would penalize the network for a sign it cannot
know. Two losses address this; both are implemented in `unified_training.py` Block D.

**Per-pair phaseless loss (baseline).** Resolve each pair's sign independently, per sample, by taking
the better of the two signs:

$$
\ell^{\text{phaseless}}_{ij} \;=\; \frac{\mathrm{RMSE}_{ij}}{\sqrt{3 n_\text{at}}} + \mathrm{MAE}_{ij},
\qquad
\mathrm{MSE/MAE}_{ij} = \min_{\sigma\in\{+1,-1\}} \big\| \mathbf{D}^{\text{ML}}_{ij} - \sigma\,\mathbf{D}^{\text{QC}}_{ij} \big\|,
$$

with the per-sample sign chosen from the $45$-vector norm (hippynn `MSEPhaseLoss`/`MAEPhaseLoss`). This
is robust and simple, but it grants **each pair its own sign**. For $n_\text{exc}$ states there are
$\binom{n_\text{exc}}{2}$ pairs but only $n_\text{exc}-1$ independent state signs, so the per-pair
solution can land in a set of signs that corresponds to **no single per-state gauge**
($r_{ij}\neq s_i s_j$) — a "pair-scramble." The surface-hopping driver can only re-align a consistent
per-state gauge, so a pair-scramble survives into the dynamics.

**Joint per-state phase loss (sign-consistent).** Enforce a single global gauge shared by all pairs:

$$
\mathcal{L}^{\text{joint}} \;=\; \frac{1}{n_\text{pairs}}\;\mathbb{E}_\text{batch}\Big[\; \min_{\{s_k\in\{\pm1\},\,s_1=+1\}} \sum_{i<j} \big\| \mathbf{D}^{\text{ML}}_{ij} - s_i s_j\, \mathbf{D}^{\text{QC}}_{ij} \big\|^2 \;\Big].
$$

The bilinear factor $s_i s_j$ ties every pair to one of $2^{\,n_\text{exc}-1}$ candidate per-state
gauges (the first sign is fixed), re-imposing the cocycle constraints the per-pair loss drops. This is
exactly the construction the FSSH driver applies at inference (`_phase_match`), here applied against the
label during training (`joint_state_nacr_loss.py`).

**A measured caveat (loss form vs gauge).** The joint loss above is a **pure sum of squares**, whereas
the phaseless blend is $\mathrm{RMSE}/\sqrt{3 n_\text{at}} + \mathrm{MAE}$ (about 70% L1). A pure-L2
objective weights the large near-CI $\|\mathbf{D}\|$ values quadratically and was observed to inflate
the S1–S2 coupling magnitude relative to the L1-leaning blend. When comparing sign strategies, match
the loss *form* (or the magnitude difference is attributed to the gauge when it is really the norm). A
matched-form factorial (per-pair vs joint gauge, at equal loss form) isolates the two effects.

## 6. Inference: reconstructing the hopping driver

Deployment is a **composed A→B** evaluation (`modelb_composed_predictor` in
`../scripts/fssh_mlip/enol_ef_nacr_developer_handoff/src/`, driven by `hippynn_fssh_composed.py`):

1. **Model A** (frozen E+F network) predicts state energies $E_k$ and forces $\mathbf{F}_k$, from which
   $\mathrm{GAP}_{ij} = E_j - E_i$ and $\mathbf{GDV}_{ij} = \mathbf{F}_i - \mathbf{F}_j$.
2. **Model B** predicts the scaled coupling $\mathbf{D}_{ij} = \mathcal{B}(\mathbf{Z},\mathbf{R},\mathbf{GDV}_{ij},\mathrm{GAP}_{ij})$.
3. The **derivative coupling** is reconstructed by dividing by the *same* gap:

$$
\mathbf{d}_{ij} \;=\; \frac{\mathbf{D}_{ij}}{\Delta_{ij}} \;=\; \frac{\mathbf{D}_{ij}}{E_j - E_i}.
$$

4. Signs are made temporally consistent by the per-state `_phase_match` against the previous step, and
   the **time-derivative coupling** that drives hops is the velocity projection

$$
\sigma_{ij} \;=\; \mathbf{d}_{ij}\cdot \mathbf{v} \;=\; \sum_{a}\sum_{\alpha\in\{x,y,z\}} d_{ij,a\alpha}\, v_{a\alpha}.
$$

**Critical numerical rule (no gap floor).** The division in step 3 uses the **raw** gap, with no floor
and no clamp. Flooring $\Delta$ (an earlier `max(\Delta, 10^{-8})`) inverts small or slightly-negative
MLIP gaps, explodes $\mathbf{d}_{ij}\sim 10^8$ and $\sigma_{ij}\sim 10^6$, and produces amplitude NaNs
that freeze hopping. The physical gap magnitude near the seam is finite (~$10^{-2}$ eV), so the raw
division is well posed; see the driver comments in `../scripts/fssh_mlip/hippynn_fssh.py`.

## 7. Units and shapes

| Symbol | Name | Array | Shape | Units |
|---|---|---|---|---|
| $\mathbf{R}$ | positions | `acn_R` | $(N, n_\text{at}, 3)$ | Å |
| $\mathbf{Z}$ | species | `acn_Z` | $(N, n_\text{at})$ | — |
| $E_k$ | state energy (shifted) | `acn_sE{k}` | $(N,)$ | eV |
| $\mathbf{F}_k$ | state force | `acn_F{k}` | $(N, n_\text{at}, 3)$ | eV/Å |
| $\mathrm{GAP}_{ij}$ | gap $E_j-E_i$ | `acn_GAP{ij}` | $(N,)$ | eV |
| $\mathbf{GDV}_{ij}$ | $\mathbf{F}_i-\mathbf{F}_j$ | `acn_GDV{ij}` | $(N, 45)$ | eV/Å |
| $\mathbf{D}_{ij}$ | scaled coupling (target) | `acn_dENACR{ij}` | $(N, 45)$ | eV/Å |
| $\mathbf{d}_{ij}$ | derivative coupling (reconstructed) | — | $(N, n_\text{at}, 3)$ | Å$^{-1}$ |
| $\sigma_{ij}$ | time-derivative coupling | — | $(N,)$ | fs$^{-1}$ |

The pairs trained are $\{12, 13, 23\}$ (S1–S2, S1–S3, S2–S3); S0 does not couple in this manifold.

## 8. Summary of the design logic

1. The derivative coupling is singular at conical intersections; **scaling by the gap removes the pole**
   and yields a smooth, bounded target ($\mathbf{D}_{ij}$, dENACR).
2. The gap and the gradient-difference vector **span the branching plane**; supplying GAP and GDV as
   inputs feeds the network the physics that sets the coupling's scale and direction.
3. The **v3 head** builds an equivariant, zero-sum per-atom vector from GDV edge-differences and a
   geometry residual, then applies a zero-initialized **gap gate** (so v3 reduces to v2 at init).
4. The electronic phase is arbitrary; the **joint per-state loss** enforces a single consistent gauge
   that the surface-hopping driver can honor, unlike the per-pair phaseless baseline.
5. At inference, **Model A supplies GAP and GDV**, Model B supplies $\mathbf{D}_{ij}$, and the coupling
   is reconstructed as $\mathbf{D}_{ij}/\Delta_{ij}$ with the **raw gap (no floor)**, then dotted with
   the velocity to drive fewest-switches surface hopping.

See `model_b_usage_guide.md` for the practical how-to (training a Model B, wiring it into the composed
FSSH driver, and the input/output contract).
