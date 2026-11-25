## The stratification procedure

Both split types use the same 80/10/10 fractions and a single seeded RNG (rng = np.random.default_rng(seed)). The difference is how indices are assigned:

random: one global shuffle — perm = rng.permutation(N), then test = first 10%, valid = next 10%, train = rest.
stratified: balance on the reference S1–S2 gap:
Load |GAP12| = |sE2 − sE1| straight from the pool labels (the reference/oracle gap, not A-predicted).
Bin all N geometries into 20 equal-count quantile bins by |GAP12|: edges = np.quantile(gap, linspace(0,1,21)), bin_id = np.digitize(...). So each bin holds ~5% of frames, ordered from smallest gap (near-CI apex) to largest.
Within each bin, shuffle (rng.shuffle(ib)) and split that bin 80/10/10.
Concatenate the per-bin slices → global train/valid/test.

The effect: because every gap bin donates the same 80/10/10 proportion, each set ends up with the same gap distribution — crucially the same share of the rare near-CI apex frames. Random splitting lets that apex share fluctuate; stratified pins it. (We saw this directly: keto near-CI |GAP12|<10 meV is ~14.8% globally → stratified gives each set ≈0.147–0.148, while random wandered ~0.1552 vs 0.1562 across seeds. The script’s _report() prints exactly this per-set apex fraction as the equitable-representation check.)

Seed dependence — yes, the same seeds, with one important nuance

Yes. Stratified uses the identical five seeds (42, 101726, 102593, 112793, 112891) as the random arm — the same default_rng(seed) drives both. Random consumes it for the global permutation; stratified consumes it for the per-bin shuffles. So each seed produces a distinct stratified partition, which is exactly what makes the 5-seed error bar meaningful for the stratified arm too (a seed-independent stratified split would give 5 identical partitions → no variance to measure).

The nuance worth knowing:

Deterministic (not seeded): the bin edges — they’re quantiles of the fixed |GAP12| distribution, so the gap-balanced structure is the same for all 5 stratified seeds.
Seeded (varies by seed): only which specific frames within each bin land in train/valid/test.

So across the 5 stratified seeds, the near-CI-balance property is held fixed while the exact membership varies — meaning the stratified error bar isolates the init + within-stratum membership variance at constant gap-balance, and the random-vs-stratified contrast is a clean like-for-like comparison (same seeds, same fractions, same reference gap; only the balancing differs).

Two connected points: this |GAP12| is the reference gap (so the split is defined on truth), and Model A and Model B share one partition per (cell, split, seed) via --reuse-masks-from — stratification is applied once, by A, and B inherits it. For enol the same procedure runs on the enol pool’s |GAP12| (median ~1.08 eV vs keto ~0.08 eV), so the bin edges are much wider — same recipe, different gap scale.
