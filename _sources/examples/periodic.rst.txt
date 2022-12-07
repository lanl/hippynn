Periodic Boundary Conditions
============================


Periodic boundary conditions require a cell node. Triclinic cells are fully supported.

Example::

    species = inputs.SpeciesNode(db_name="species")
    positions = inputs.PositionsNode(db_name="coordinates")
    cell = inputs.CellNode(db_name="cell")
    network = networks.Hipnn("HIPNN",
                            (species, positions,cell),
                             periodic=True, module_kwargs = network_params)


This will Generate a :class:`~hippynn.graphs.nodes.pairs.PeriodicPairIndexer`
that searches image cells surrounding the data. It includes wrapping of coordinates
to within the unit cell. Because the nearest images (27 replicates of the cell at
search radius 1) are numerous, periodic pair finding is noticeably more costly in terms of
memory and time than open boundary conditions. The less skewed your cells are, as well as
are the larger cells are compared to the cutoff distance required,
the fewer images needed to be searched in finding pairs.


Dynamic Pair Finder
-------------------
For highly complex datasets, there is a more flexible pairfinder which
can be built as such::

    enc, padidxer = indexers.acquire_encoding_padding(
                    species, species_set=network_params['possible_species'])

    pairfinder = pairs.DynamicPeriodicPairs('PairFinder', (positions, species, cell),
                                            dist_hard_max=network_params['dist_hard_max'])
    network = networks.Hipnn("HIPNN", (padidxer, pairfinder), periodic=True,
                             module_kwargs=network_params)

The :class:`~hippynn.graphs.nodes.pairs.DynamicPeriodicPairs` object uses an algorithm to
determine how many image cells need to be searched for each system, and iterates through
the systems one by one. The upshot of this is that less memory is required.
However, the cost is that each system is evaluated independently in serial,
and as such the pair finding can be a rather slow operation. This algorithm is
more likely to show benefits when the number of atoms in a training system is highly
variable.

Caching Pre-computed Pairs
--------------------------

To mitigate the cost of periodic pairfinding (with either of the above methods),
the pairs for each system in the training database can be cached. To do this,
first assemble your modules for training. Then run
:func:`~hippynn.experiment.assembly.precompute_pairs` on the training module.
This produces a cache of the pairs in the database, and replaces the
pair finder with an input node that gets the information from this cache.
After that, you'll have to re-assemble a new model for training,
and set the database to use these new inputs.

Example::

        training_modules, db_info = \
            assemble_for_training(train_loss, validation_losses, plot_maker=plot_maker)
        from hippynn.experiment.assembly import precompute_pairs
        precompute_pairs(training_modules.model, database,n_images=4)
        training_modules, db_info = assemble_for_training(train_loss,
                                                          validation_losses,plot_maker=plot_maker)
        database.inputs = db_info['inputs']

Caching pairs has several caveats. By default it produces a sparse output,
this sparse tensor cannot be used with the ``n_workers`` argument on a dataloader,
due to current limitations in pytorch. As such we recommend you move the
dataset to a GPU and use ``n_workers=None`` when caching pairs.

What's not yet supported
------------------------
During training, we don't yet have support for mixed PBCs where
some directions are periodic and others are open.
However, the ASE interface can almost (but not quite yet) handle
simulations for such systems, because in this case ASE handles neighbor finding.

We also don't have support for mixed datasets of open and closed boundaries.
To deal with this, you can embed your open systems in a very large box as
a pre-processing step.



