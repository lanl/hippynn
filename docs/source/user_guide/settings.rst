Library Settings
================

``hippynn`` has several settings located as variables stored in :obj:``~hippynn.settings``.

There are four possible sources for settings.

1. Default values
2. The file `~/.hippynnrc`, which is a standard python config file which contains
   variables under the section [GLOBAL].
3. A file specified by the environment variable `HIPPYNN_LOCAL_RC_FILE`
   which is treated the same as the user rc file.
4. Environment variables prefixed by ``HIPPYNN_``, e.g. ``HIPPYNN_DEFAULT_PLOT_FILETYPE``.

These three sources are checked in order, so that values in later sources overwrite values
found in earlier sources.

Some settings can be directly changed during script execution.
If possible, this is indicated in the `dynamic` column in the following table.

The following settings are available:

.. list-table:: Hippynn Settings Summary
   :widths: 10 10 10 10 10
   :header-rows: 1

   * - Variable
     - Meaning
     - Values
     - Default
     - Dynamic
   * - PROGRESS
     - | Progress bars function during
       | training, evaluation, and prediction
     - tqdm, none
     - tqdm
     - | Yes, but assign this to a
       | generator-wrapper such as ``tqdm.tqdm``,
       | or with a python ``None`` to disable.
       | The wrapper must accept ``tqdm`` arguments,
       | although it technically doesn't have to
       | do anything with them.
   * - DEFAULT_PLOT_FILETYPE
     - | File type to use for plots when
       | not explicitly specified
     - | Filetypes supported
       | by matplotlib e.g.
       | '.pdf', '.png', '.jpg'
     - .pdf
     - Yes
   * - USE_CUSTOM_KERNELS
     - | Use custom kernels with numba or cupy.
       | Auto tries to detect the
       | installation of numba or cupy.
       | for more info see :doc:`/user_guide/ckernels`.
     - auto, true, false, pytorch, numba, cupy
     - auto
     - Not directly, use :func:`~hippynn.custom_kernels.set_custom_kernels`
   * - WARN_LOW_DISTANCES
     - | Warn if atom distances are low
       | compared to current radial
       | sensitivity parameters.
     - true, false
     - true
     - yes
   * - DEBUG_LOSS_BROADCAST
     - | Warn if quantities in the loss
       | broadcast badly against each
       | other.
     - true, false
     - false
     - no
   * - DEBUG_GRAPH_BROADCAST
     - | Print high verbose information about
       | the execution of a graph module.
       | Don't turn this on unless something
       | is going very wrong.
     - true, false
     - false
     - no
