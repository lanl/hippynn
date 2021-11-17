Periodic Boundary Conditions
============================


Periodic boundary conditions require a cell node.

Example::

    species = inputs.SpeciesNode(db_name="species")
    positions = inputs.PositionsNode(db_name="coordinates")
    cell = inputs.CellNode(db_name="cell")
    network = networks.Hipnn("HIPNN",
                            (species, positions,cell),
                             periodic=True, module_kwargs = network_params)


Because the default is to search only the nearest images (27 replicates of the cell),
the smallest length of the box should be larger than the interaction length of the network.
If you want to go to smaller boxes, you will need to search further periodic images.
This can be done by explicitly constructing a :class:`~hippynn.graphs.nodes.pairs.PeriodicPairIndexer`
with the keyword arguments ``module_kwargs=dict(n_images=k)`` for a search depth ``k``.

Triclinic cells have preliminary support; because the number off images to
search depends on the skew of the cell, you'll need to figure out how many images to search.
The less skewed your cells are, the fewer are needed.

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
determine how many image cells need to be searched for each system.
The upside of this pair finder is that it is extremely robust and can handle,
for example, highly skewed cells or single-atom cells.
However, the cost is that each system is evaluated independently in serial,
and as such the pair finding can be a rather expensive operation.

Caching Pre-computed Pairs
--------------------------

To mitigate the cost of pairfinding (either with the static image search or dynamic search),
the pairs for each system in the training database can be cached. To do this,
first assemble your modules for training. Then run
:func:`~hippynn.experiment.assembly.precompute_pairs` on the training module.
This produces a cache of the pairs in the database, and replaces the
pair finder with a lookup node that gets the information from this cache.
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

What's not yet supported
------------------------
During training, we don't yet have support for mixed PBCs where
some directions are periodic and others are open.
However, the ASE interface can handle simulation for such systems,
because ASE handles neighbor finding.

We also don't have support for mixed datasets of open and closed boundaries.
To deal with this, you could embed your open systems in a very large box as
a pre-processing step.



