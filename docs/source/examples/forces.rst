Force Training
==============


Example::

    force = physics.GradientNode("gradients", (sys_energy, positions), sign=1,db_name="grad")

The "sign" argument specifies whether or not the arrays
are gradients (sign=1) or forces (sign=-1).

You can then make losses such as MSE or MAE from the force node
the same way you might any other Node.