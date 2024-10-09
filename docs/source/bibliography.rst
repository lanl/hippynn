Research articles using hippynn
===============================

hippynn implements a variety of methods from the research literature. Some of the earlier
research was created with an older, internal implementation
of HIP-NN using theano. However, the capabilities are available in hippynn.

One of the main components of hippynn is the implementaiton of HIP-NN,
or the Hierarchical Interacting Particle Neural Network, was introduced in
:cite:t:`lubbers2018hierarchical` for the modeling of molecular energies and forces
from atomistic configuration data. HIP-NN was also used to help validate results for potential energy surfaces
in :cite:t:`suwa2019machine` and :cite:t:`smith2021automated`, and was
later extended to a more flexible functional form, HIP-NN with Tensor Sensitivities,
or HIP-NN-TS, in :cite:t:`chigaev2023lightweight`.
:cite:t:`fedik2024challenges` critically examined the performance of this improved
functional form for transitions states and transition path sampling.
:cite:t:`matin2024machine` demonstrated a method for improving the performance
of potentials with respect to experiment by incorporating experimental structural
data.  :cite:t:`burrill2024mltb` showed how a linear combination of semi-empirical
and machine learning models can be more powerful than either model alone.
:cite:t:`shinkle2024thermodynamic` demonstrated that HIP-NN can model free energies
for coarse-grained models using force-matching, and that these many-body models provide
improved transferability between thermodynamic states.

HIP-NN is also useful for modeling properties aside from energy/forces.
It was adapted to learn charges in :cite:t:`nebgen2018transferable`
and to learn charge predictions from dipole information in :cite:t:`sifain2018discovering`.
Bond order regression to predict two-body quantities was explored in :cite:t:`magedov2021bond`.
The atom (charge) and two-body (bond) regressions were combined to build Huckel-type
quantum Hamiltonians in :cite:t:`zubatiuk2021machine`. This was extended to
semi-empirical Hamiltonians in :cite:t:`zhou2022deep` by combining the facilities
of hippynn with another pytorch code, PYSEQM, developed by :cite:t:`zhou2020graphics`,
which provides quantum calculations that are differentiable by pytorch.

Another avenue of work has been to model excited state dynamics with HIP-NN.
In :cite:t:`sifain2021predicting`, a localization layer was used to predict
both the energy and location of singlet-triplet excitations in organic materials.
In :cite:t:`habib2023machine`, HIP-NN was used in a dynamical setting to learn
the dynamics of excitons in nanoparticles. In this mode, the predictions of
a model produce inputs for the next time step, and training takes place by
backpropagating through multiple steps of prediction. :cite:t:`li2024machine` used
the framework to predict several excited state properties; energy,
transition dipole, and non-adiabatic coupling vectors were predicted for several
excited states in a molecular system.


.. bibliography::